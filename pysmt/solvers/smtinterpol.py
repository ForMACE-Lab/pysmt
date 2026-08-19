#
# This file is part of pySMT.
#
#   Copyright 2014 Andrea Micheli and Marco Gario
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
"""SMTInterpol solver and interpolation support through its SMT-LIB API."""

import os
import re
import shlex
import shutil
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import pysmt.smtlib.commands as smtcmd
from pysmt.decorators import clear_pending_pop
from pysmt.environment import Environment
from pysmt.exceptions import (SolverReturnedUnknownResultError,
                              InternalSolverError,
                              ConvertExpressionError,
                              PysmtValueError,
                              SolverAPINotFound,
                              UnsupportedOperatorError)
from pysmt.fnode import FNode
from pysmt.logics import (Logic, QF_ALIA, QF_AUFLIA, QF_AX, QF_LIA, QF_LRA,
                          QF_UF, QF_UFLIA, QF_UFLIRA, QF_UFLRA)
from pysmt.oracles import get_logic
from pysmt.smtlib.script import SmtLibCommand
from pysmt.smtlib.solver import SmtLibOptions, SmtLibSolver
from pysmt.solvers.eager import EagerModel
from pysmt.solvers._smtlib_bridge import (collect_symbols,
                                          declaration_script,
                                          make_safe_symbol_map,
                                          node_to_smtlib,
                                          normalize_logic_name,
                                          repair_quoted_symbols,
                                          rename_symbols,
                                          restore_mapped_symbols,
                                          restore_original_symbols,
                                          script_logic,
                                          symbol_declaration,
                                          term_to_pysmt)
from pysmt.solvers.interpolation import Interpolator


_BASE_LOGICS = [QF_UF, QF_LIA, QF_LRA, QF_UFLIA, QF_UFLRA, QF_AX, QF_ALIA,
                QF_AUFLIA]
SMTINTERPOL_QF_AUFLIRA = Logic(
    name="QF_AUFLIRA",
    description=(
        "Quantifier-free arrays and uninterpreted functions combined with "
        "linear integer and real arithmetic."
    ),
    quantifier_free=True,
    arrays=True,
    integer_arithmetic=True,
    real_arithmetic=True,
    linear=True,
    uninterpreted=True,
)
SMTINTERPOL_SOLVER_LOGICS = _BASE_LOGICS + [SMTINTERPOL_QF_AUFLIRA]
SMTINTERPOL_INTERPOLATION_LOGICS = _BASE_LOGICS


def _jar_candidates() -> Iterator[Path]:
    configured = os.environ.get("PYSMT_SMTINTERPOL_JAR")
    if configured:
        yield Path(configured).expanduser()

    for entry in sys.path:
        if entry:
            yield Path(entry) / "smtinterpol.jar"

    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        yield parent / "solvers" / "smtinterpol" / "dist"
        yield parent / "smtinterpol" / "dist"

    yield Path.home() / ".smt_solvers" / "smtinterpol"


def _resolve_jar() -> Path:
    for candidate in _jar_candidates():
        if candidate.is_file() and candidate.suffix == ".jar":
            return candidate.resolve()
        if candidate.is_dir():
            jars = sorted(candidate.glob("smtinterpol*.jar"))
            if jars:
                return jars[-1].resolve()
    raise SolverAPINotFound


def _java_command() -> List[str]:
    java = os.environ.get("PYSMT_SMTINTERPOL_JAVA") or shutil.which("java")
    if java is None:
        raise SolverAPINotFound
    java_options = shlex.split(
        os.environ.get("PYSMT_SMTINTERPOL_JAVA_OPTIONS", "")
    )
    return [java] + java_options + ["-jar", str(_resolve_jar())]


# Resolve eagerly so Factory can treat a missing Java runtime/JAR exactly like
# a missing native solver binding.
SMTINTERPOL_COMMAND = _java_command()

_ERROR_RESPONSE = re.compile(r'\(error "([^"]*)"\)')


class SMTInterpolOptions(SmtLibOptions):
    """Options for the SMT-LIB subprocess used by SMTInterpol."""

    def __init__(self, **base_options):
        debug_interaction = base_options.pop("debug_interaction", False)
        SmtLibOptions.__init__(self, **base_options)
        self.debug_interaction = debug_interaction

    def __call__(self, solver):
        SmtLibOptions.__call__(self, solver)
        solver.set_option(":verbosity", "0")

# EOC SMTInterpolOptions


class SMTInterpolSolver(SmtLibSolver):
    """Incremental SMTInterpol solver through the standard SMT-LIB wrapper."""

    LOGICS = SMTINTERPOL_SOLVER_LOGICS
    OptionsClass = SMTInterpolOptions

    def __init__(self, environment: Environment, logic, **options):
        self._symbol_map: Dict[FNode, FNode] = {}
        SmtLibSolver.__init__(
            self,
            list(SMTINTERPOL_COMMAND),
            environment,
            logic,
            LOGICS=self.LOGICS,
            **options
        )

    def _send_command(self, cmd):
        """Send SMT-LIB while preserving names already written as ``|...|``."""
        serialized = repair_quoted_symbols(cmd.serialize_to_string())
        self._debug("Sending: %s", serialized)
        self.solver_stdin.write(serialized)
        self.solver_stdin.write("\n")
        self.solver_stdin.flush()

    def _read_value_response(self):
        """Read one response without consuming output from the next command."""
        response = ""
        while True:
            line = self.solver_stdout.readline()
            if line == "":
                raise InternalSolverError(
                    "SMTInterpol closed stdout while returning a value"
                )
            response += line
            start = response.find("(")
            if start < 0:
                continue
            try:
                end = _matching_parenthesis(response, start)
            except InternalSolverError:
                continue
            if response[end + 1:].strip():
                raise InternalSolverError(
                    "Unexpected SMTInterpol value response: %s"
                    % response[:4000]
                )
            break

        self._debug("Read: %s", response.strip())
        return response

    @clear_pending_pop
    def add_assertion(self, formula, named=None):
        """Declare constants and function symbols before asserting a formula."""
        self._assert_is_boolean(formula)
        original_formula = formula
        try:
            detected = get_logic(formula, self.environment)
            if not detected <= self.logic:
                raise ConvertExpressionError(
                    "SMTInterpol logic %s cannot represent %s"
                    % (self.logic, detected),
                    expression=formula,
                )
            formula = formula.simplify()
        except UnsupportedOperatorError as ex:
            raise ConvertExpressionError(
                str(ex),
                expression=formula,
            ) from ex

        self._symbol_map.update(
            make_safe_symbol_map(
                [original_formula],
                self.environment,
                "smtinterpol_solver",
            )
        )
        formula = rename_symbols(
            formula,
            self._symbol_map,
            self.environment,
        )

        sorts = self.to.get_types(original_formula, custom_only=True)
        for sort in sorts:
            if all(sort not in scope for scope in self.declared_sorts):
                self._declare_sort(sort)

        for symbol in collect_symbols([original_formula]):
            native_symbol = self._symbol_map[symbol]
            if all(
                native_symbol not in scope
                for scope in self.declared_vars
            ):
                self._declare_variable(native_symbol)

        self._send_silent_command(SmtLibCommand(smtcmd.ASSERT, [formula]))

    @clear_pending_pop
    def solve(self, assumptions=None):
        has_assumptions = assumptions is not None
        if has_assumptions:
            self.push()
            try:
                for assumption in assumptions:
                    self.add_assertion(assumption)
            except Exception:
                self.pop()
                raise

        self._send_command(SmtLibCommand(smtcmd.CHECK_SAT, []))
        answer = self._get_answer()
        if answer == "sat":
            result = True
        elif answer == "unsat":
            result = False
        elif answer == "unknown":
            if has_assumptions:
                self.pop()
            raise SolverReturnedUnknownResultError()
        else:
            if has_assumptions:
                self.pop()
            raise InternalSolverError(
                "Unexpected SMTInterpol satisfiability result: %s" % answer
            )

        if has_assumptions:
            # Keep the temporary scope alive so get_model()/get_value() refer
            # to the model produced under these assumptions. The standard
            # clear_pending_pop decorator removes it before the next mutation.
            self.pending_pop = True
        return result

    def get_value(self, item):
        self._assert_no_function_type(item)
        native_item = rename_symbols(
            item,
            self._symbol_map,
            self.environment,
        )
        self._send_command(SmtLibCommand(smtcmd.GET_VALUE, [native_item]))
        response = self._read_value_response().strip()
        pairs = _split_top_level(response[1:-1])
        if len(pairs) != 1 or not pairs[0].startswith("("):
            raise InternalSolverError(
                "Unexpected SMTInterpol value response: %s" % response[:4000]
            )
        assignment = _split_top_level(pairs[0][1:-1])
        if len(assignment) != 2:
            raise InternalSolverError(
                "Unexpected SMTInterpol assignment: %s" % pairs[0][:4000]
            )
        value_text = assignment[1]

        symbols = set().union(*self.declared_vars)
        declarations = "\n".join(
            symbol_declaration(symbol)
            for symbol in sorted(
                symbols,
                key=lambda symbol: symbol.symbol_name(),
            )
        )
        item_type = item.get_type()
        try:
            type_text = item_type.as_smtlib(False)
        except TypeError:
            type_text = item_type.as_smtlib()
        dummy_name = "__pysmt_smtinterpol_value_%x" % id(item)
        parser_logic = (
            QF_UFLIRA
            if self.logic == SMTINTERPOL_QF_AUFLIRA
            else self.logic
        )
        script = "(set-logic %s)\n" % normalize_logic_name(parser_logic)
        if declarations:
            script += declarations + "\n"
        script += "(declare-fun %s () %s)\n" % (dummy_name, type_text)
        script += "(assert (= %s %s))\n" % (dummy_name, value_text)

        parsed = self.parser.__class__().get_script(StringIO(script))
        equality = parsed.get_last_formula(self.environment.formula_manager)
        if len(equality.args()) != 2:
            raise InternalSolverError(
                "Could not parse SMTInterpol value: %s" % value_text[:4000]
            )
        value = restore_original_symbols(equality.arg(1), symbols)
        return restore_mapped_symbols(
            value,
            self._symbol_map,
            self.environment,
        )

    def get_model(self):
        assignment = {}
        symbols = set().union(*self.declared_vars)
        reverse_map = {
            native: original
            for original, native in self._symbol_map.items()
        }
        for symbol in symbols:
            if symbol.is_term():
                original = reverse_map.get(symbol, symbol)
                assignment[original] = self.get_value(original)
        return EagerModel(
            assignment=assignment,
            environment=self.environment,
        )

    @clear_pending_pop
    def reset_assertions(self):
        self._send_silent_command(
            SmtLibCommand(smtcmd.RESET_ASSERTIONS, [])
        )
        # SMTInterpol removes declarations together with assertions.
        self.declared_vars = [set()]
        self.declared_sorts = [set()]

    def _exit(self):
        """Close the Java process and reap it before releasing the pipes."""
        if self.solver.poll() is None:
            try:
                self._send_command(SmtLibCommand(smtcmd.EXIT, []))
            except (BrokenPipeError, OSError, ValueError):
                pass

        for stream in (
            self.solver_stdin,
            self.solver_stdout,
            self.solver.stderr,
        ):
            try:
                stream.close()
            except (OSError, ValueError):
                pass

        try:
            self.solver.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.solver.terminate()
            try:
                self.solver.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.solver.kill()
                self.solver.wait()

# EOC SMTInterpolSolver


def _matching_parenthesis(text: str, start: int) -> int:
    depth = 0
    quoted_symbol = False
    string = False
    comment = False
    index = start

    while index < len(text):
        char = text[index]
        if comment:
            if char == "\n":
                comment = False
            index += 1
            continue
        if string:
            if char == '"':
                if index + 1 < len(text) and text[index + 1] == '"':
                    index += 2
                    continue
                string = False
            index += 1
            continue
        if quoted_symbol:
            if char == "|":
                quoted_symbol = False
            index += 1
            continue

        if char == ";":
            comment = True
        elif char == '"':
            string = True
        elif char == "|":
            quoted_symbol = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1

    raise InternalSolverError("Incomplete SMTInterpol S-expression response")


def _split_top_level(text: str) -> List[str]:
    terms = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break

        start = index
        if text[index] == "(":
            index = _matching_parenthesis(text, index) + 1
        elif text[index] == "|":
            index += 1
            while index < len(text) and text[index] != "|":
                index += 1
            if index >= len(text):
                raise InternalSolverError(
                    "Incomplete quoted symbol in SMTInterpol response"
                )
            index += 1
        elif text[index] == '"':
            index += 1
            while index < len(text):
                if text[index] == '"':
                    if index + 1 < len(text) and text[index + 1] == '"':
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
        else:
            while index < len(text) and not text[index].isspace():
                index += 1
        terms.append(text[start:index])
    return terms


def _interpolant_terms(output: str) -> List[str]:
    status = re.search(r"(?m)^(sat|unsat|unknown)\s*$", output)
    error = _ERROR_RESPONSE.search(output)
    # An error *before* the status means SMTInterpol refused part of the
    # script, so the assertions never reached the stack -- and an empty stack
    # answers `sat`, which the caller would otherwise read as "this query has
    # no interpolant". An error *after* it is the expected complaint about
    # asking for interpolants of a satisfiable query.
    if error is not None and (status is None or error.start() < status.start()):
        raise InternalSolverError(
            "SMTInterpol rejected the interpolation script: %s"
            % error.group(1)[:4000]
        )
    if status is None:
        raise InternalSolverError(
            "SMTInterpol did not return a satisfiability status: %s"
            % output[:4000]
        )
    if status.group(1) == "sat":
        return []
    if status.group(1) == "unknown":
        raise SolverReturnedUnknownResultError()

    response_start = output.find("(", status.end())
    if response_start < 0:
        raise InternalSolverError(
            "SMTInterpol did not return interpolants: %s" % output[:4000]
        )
    response_end = _matching_parenthesis(output, response_start)
    response = output[response_start + 1:response_end]
    return _split_top_level(response)


class SMTInterpolInterpolator(Interpolator):
    """Craig and sequence interpolation backed by SMTInterpol."""

    LOGICS = SMTINTERPOL_INTERPOLATION_LOGICS

    def __init__(self, environment: Environment, logic=None):
        Interpolator.__init__(self)
        self.environment = environment
        self.logic = logic

    def _check_logic(self, formulas: Sequence[FNode]) -> List[Logic]:
        """The formulas' logics, refusing any SMTInterpol cannot interpolate."""
        detected = []
        for formula in formulas:
            logic = get_logic(formula, self.environment)
            if not any(logic <= supported for supported in self.LOGICS):
                raise PysmtValueError(
                    "Logic not supported by SMTInterpol interpolation. "
                    "(detected logic is: %s)" % logic
                )
            detected.append(logic)
        return detected

    def _script(self, formulas: Sequence[FNode], logic: Logic):
        symbol_map = make_safe_symbol_map(
            formulas,
            self.environment,
            "smtinterpol_itp",
        )
        native_formulas = [
            rename_symbols(formula, symbol_map, self.environment)
            for formula in formulas
        ]
        declarations, symbols = declaration_script(native_formulas)
        partition_names = [
            "__pysmt_smtinterpol_partition_%d" % index
            for index in range(len(formulas))
        ]

        lines = [
            "(set-option :print-success false)",
            "(set-option :produce-interpolants true)",
            "(set-option :verbosity 0)",
            "(set-logic %s)" % normalize_logic_name(logic),
        ]
        if declarations:
            lines.append(declarations)
        for name, formula in zip(partition_names, native_formulas):
            lines.append(
                "(assert (! %s :named %s))"
                % (node_to_smtlib(formula), name)
            )
        lines.extend([
            "(check-sat)",
            "(get-interpolants %s)" % " ".join(partition_names),
            "(exit)",
        ])
        return (
            "\n".join(lines) + "\n",
            declarations,
            symbols,
            symbol_map,
        )

    def _interpolate(self, formulas: Sequence[FNode]):
        formulas = list(formulas)
        if len(formulas) < 2:
            raise PysmtValueError("interpolation needs at least 2 formulae")
        logic = script_logic(self._check_logic(formulas), self.LOGICS,
                             self.logic)
        script, declarations, symbols, symbol_map = self._script(formulas,
                                                                 logic)

        timeout_text = os.environ.get("PYSMT_SMTINTERPOL_TIMEOUT")
        timeout = float(timeout_text) if timeout_text else None
        try:
            completed = subprocess.run(
                SMTINTERPOL_COMMAND,
                input=script,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as ex:
            raise InternalSolverError(
                "SMTInterpol interpolation timed out"
            ) from ex
        except OSError as ex:
            raise InternalSolverError(
                "Could not execute SMTInterpol: %s" % ex
            ) from ex

        if completed.returncode != 0:
            raise InternalSolverError(
                "SMTInterpol failed with exit code %d: %s"
                % (completed.returncode, completed.stderr[:4000])
            )

        terms = _interpolant_terms(completed.stdout)
        if not terms:
            return None
        expected = len(formulas) - 1
        if len(terms) != expected:
            raise InternalSolverError(
                "SMTInterpol returned %d interpolants; expected %d. "
                "Output: %s"
                % (len(terms), expected, completed.stdout[:4000])
            )
        return [
            term_to_pysmt(
                term,
                logic,
                declarations,
                self.environment,
                symbols,
                symbol_map,
            )
            for term in terms
        ]

    def binary_interpolant(self, a: FNode, b: FNode):
        interpolants = self._interpolate([a, b])
        if interpolants is None:
            return None
        return interpolants[0]

    def sequence_interpolant(self, formulas: Sequence[FNode]):
        return self._interpolate(formulas)

    def _exit(self):
        pass

# EOC SMTInterpolInterpolator
