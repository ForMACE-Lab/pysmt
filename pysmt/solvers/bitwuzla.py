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
"""Native Bitwuzla solver and interpolation support."""

from typing import Any, Dict, List, Optional, Sequence, Set

from pysmt.exceptions import SolverAPINotFound
from pysmt.environment import Environment
from pysmt.fnode import FNode

try:
    from bitwuzla import (BitwuzlaException, Option, # type: ignore[import]
                          Options, Parser, Result, TermManager)
except ImportError:
    raise SolverAPINotFound

from pysmt.decorators import catch_conversion_error, clear_pending_pop
from pysmt.exceptions import (SolverReturnedUnknownResultError,
                              InternalSolverError,
                              ConvertExpressionError,
                              PysmtValueError)
from pysmt.logics import (Logic, QF_ABV, QF_AUFBV, QF_BV, QF_UFBV,
                          get_logic_by_name)
from pysmt.oracles import get_logic
from pysmt.solvers._array_rewriter import (expand_array_equalities,
                                           make_array_expander,
                                           shared_array_expander)
from pysmt.solvers._smtlib_bridge import (collect_symbols,
                                          interpolation_state,
                                          make_renamer,
                                          make_safe_symbol_map,
                                          node_to_smtlib,
                                          normalize_logic_name,
                                          repair_quoted_symbols,
                                          rename_symbols,
                                          script_logic,
                                          symbol_declaration,
                                          term_to_pysmt, terms_to_pysmt)
from pysmt.solvers.eager import EagerModel
from pysmt.solvers.interpolation import Interpolator
from pysmt.solvers.smtlib import SmtLibBasicSolver
from pysmt.solvers.solver import (Converter, IncrementalTrackingSolver,
                                  Solver, SolverOptions)

# Bitwuzla decides quantifier-free bit-vectors, arrays and uninterpreted
# functions, including the constant-array variants of those logics.
BITWUZLA_LOGICS = [QF_BV, QF_UFBV, QF_ABV, QF_AUFBV,
                   get_logic_by_name("QF_ABV*"),
                   get_logic_by_name("QF_AUFBV*")]


class BitwuzlaOptions(SolverOptions):
    """Options for the native Bitwuzla backend."""

    def __init__(self, **base_options):
        SolverOptions.__init__(self, **base_options)
        if self.unsat_cores_mode is not None:
            raise PysmtValueError(
                "'unsat_cores_mode' is not supported by the Bitwuzla backend."
            )

    def __call__(self, solver):
        solver._native_options.set(Option.PRODUCE_MODELS, self.generate_models)

        if self.random_seed is not None:
            solver._native_options.set(Option.SEED, self.random_seed)

        for name, value in self.solver_options.items():
            option_name = str(name).upper().replace("-", "_").replace(".", "_")
            try:
                option = Option.__members__[option_name]
            except KeyError:
                raise PysmtValueError(
                    "Unknown Bitwuzla option '%s'. Accepted names are: %s"
                    % (name, ", ".join(sorted(Option.__members__)))
                )
            try:
                solver._native_options.set(option, value)
            except (RuntimeError, TypeError, ValueError) as ex:
                raise PysmtValueError(
                    "Error setting Bitwuzla option '%s=%s': %s"
                    % (name, value, ex)
                )

# EOC BitwuzlaOptions


class BitwuzlaConverter(Converter):
    """SMT-LIB-backed converter for Bitwuzla's native parser."""

    def __init__(self, solver):
        self.solver = solver

    @catch_conversion_error
    def convert(self, formula):
        try:
            return self.solver._convert(formula)
        except BitwuzlaException as ex:
            raise ConvertExpressionError(
                message="Bitwuzla could not convert the expression: %s" % ex,
                expression=formula,
            )

    def back(self, term, *args, **kwargs):
        return self.solver._back(term)

# EOC BitwuzlaConverter


class BitwuzlaSolver(IncrementalTrackingSolver, SmtLibBasicSolver):
    """Incremental pySMT solver backed by Bitwuzla."""

    LOGICS = BITWUZLA_LOGICS
    OptionsClass = BitwuzlaOptions

    def __init__(self, environment: Environment, logic, **options):
        IncrementalTrackingSolver.__init__(
            self,
            environment=environment,
            logic=logic,
            **options
        )
        self._new_context()
        self.converter = BitwuzlaConverter(self)

    def _new_context(self):
        self._term_manager = TermManager()
        self._native_options = Options()
        self.options(self)
        self._parser = Parser(self._term_manager, self._native_options)
        self._parser.parse(
            "(set-logic %s)" % normalize_logic_name(self.logic),
            True,
            False,
        )
        self.bitwuzla = self._parser.bitwuzla()
        self._declared_symbols: Set[FNode] = set()
        self._declaration_lines: List[str] = []
        self._symbol_map: Dict[FNode, FNode] = {}
        # One renamer per context, sharing the (growing) symbol map: the
        # assertion stack of an incremental solve repeats most of its subterms,
        # and a per-call walker rebuilt all of them each time.
        self._renamer = make_renamer(self._symbol_map, self.environment)
        self._array_expander = make_array_expander(self.environment)
        # Native terms already built for a pySMT node. Bitwuzla terms belong
        # to the term manager, not to the assertion stack, so one stays valid
        # for the whole context and across any number of push/pop levels --
        # and it dies with the context, which is why this is reset here rather
        # than anywhere else. Without it, re-asserting a formula the solver has
        # seen before re-serializes and re-parses its entire cone; the
        # incremental callers do exactly that, repeatedly, with the unchanged
        # half of a query.
        self._term_cache: Dict[FNode, Any] = {}

    def _ensure_declarations(self, formulas: Sequence[FNode]):
        new_symbols = [
            symbol
            for symbol in collect_symbols(formulas)
            if symbol not in self._declared_symbols
        ]
        if not new_symbols:
            return

        self._symbol_map.update(
            make_safe_symbol_map(
                formulas,
                self.environment,
                "bitwuzla",
                symbols=new_symbols,
            )
        )
        native_symbols = [self._symbol_map[symbol] for symbol in new_symbols]
        declarations = "\n".join(
            symbol_declaration(symbol) for symbol in native_symbols
        )
        self._parser.parse(repair_quoted_symbols(declarations), True, False)
        self._declared_symbols.update(new_symbols)
        self._declaration_lines.extend(
            symbol_declaration(symbol) for symbol in native_symbols
        )

    def _convert(self, formula: FNode):
        term = self._term_cache.get(formula)
        if term is not None:
            return term

        # Bitwuzla is incomplete on equalities involving constant arrays and
        # answers `unknown` for them; stating those equalities elementwise is
        # an equivalence over a finite index sort and keeps the query decidable.
        expanded = expand_array_equalities(
            formula,
            self.environment,
            self._array_expander,
        )
        self._ensure_declarations([expanded])
        native_formula = rename_symbols(
            expanded,
            self._symbol_map,
            self.environment,
            self._renamer,
        )
        term = self._parser.parse_term(node_to_smtlib(native_formula))
        self._term_cache[formula] = term
        return term

    def _back(self, term):
        return term_to_pysmt(
            term,
            self.logic,
            "\n".join(self._declaration_lines),
            self.environment,
            self._declared_symbols,
            self._symbol_map,
        )

    @clear_pending_pop
    def _reset_assertions(self):
        self._new_context()

    @clear_pending_pop
    def _add_assertion(self, formula: FNode, named: Optional[str] = None):
        self._assert_is_boolean(formula)
        if named is not None:
            raise PysmtValueError(
                "Named assertions are not supported by the Bitwuzla backend."
            )
        self.bitwuzla.assert_formula(self.converter.convert(formula))
        return formula

    @clear_pending_pop
    def _solve(self, assumptions=None):
        native_assumptions = []
        if assumptions is not None:
            native_assumptions = [
                self.converter.convert(assumption) for assumption in assumptions
            ]

        try:
            result = self.bitwuzla.check_sat(*native_assumptions)
        except RuntimeError as ex:
            raise InternalSolverError(str(ex))

        if result == Result.SAT:
            return True
        if result == Result.UNSAT:
            return False
        raise SolverReturnedUnknownResultError()

    @clear_pending_pop
    def _push(self, levels=1):
        self.bitwuzla.push(levels)

    @clear_pending_pop
    def _pop(self, levels=1):
        self.bitwuzla.pop(levels)

    def get_value(self, item):
        self._assert_no_function_type(item)
        try:
            value = self.bitwuzla.get_value(self.converter.convert(item))
        except RuntimeError as ex:
            raise InternalSolverError(str(ex))
        return self.converter.back(value)

    def get_model(self):
        assignment = {}
        for symbol in self._declared_symbols:
            if not symbol.symbol_type().is_function_type():
                assignment[symbol] = self.get_value(symbol)
        return EagerModel(assignment=assignment, environment=self.environment)

    def declare_variable(self, var):
        self._ensure_declarations([var])

    def exit(self):
        # SmtLibIgnoreMixin.exit() models the SMT-LIB ``exit`` *command*, and
        # ignoring it is right for a solver driven over SMT-LIB. It is not
        # the resource release Solver.exit() stands for, which a native
        # context does need; the mixin only shadows it through the MRO.
        Solver.exit(self)

    def _exit(self):
        # The cache holds one native term per node of every query the solver
        # was given, and the converter closes a reference cycle back to this
        # solver, so neither goes away on its own. Dropping them here keeps
        # the native terms from outliving the context they were built in.
        self._term_cache.clear()
        self.converter.solver = None
        del self.converter
        del self.bitwuzla
        del self._parser
        del self._term_manager

# EOC BitwuzlaSolver


class BitwuzlaInterpolator(Interpolator):
    """Craig and sequence interpolation backed by Bitwuzla."""

    LOGICS = BITWUZLA_LOGICS

    def __init__(self, environment: Environment, logic=None):
        Interpolator.__init__(self)
        self.environment = environment
        self.logic = logic

    def _check_logic(self, formulas: Sequence[FNode]) -> List[Logic]:
        """The formulas' logics, refusing any Bitwuzla cannot interpolate."""
        detected = []
        for formula in formulas:
            logic = get_logic(formula, self.environment)
            if not any(logic <= supported for supported in self.LOGICS):
                raise PysmtValueError(
                    "Logic not supported by Bitwuzla interpolation. "
                    "(detected logic is: %s)" % logic
                )
            detected.append(logic)
        return detected

    def _context(self, formulas: Sequence[FNode], logic: Logic):
        manager = TermManager()
        options = Options()
        options.set(Option.PRODUCE_INTERPOLANTS, True)
        options.set(Option.INTERPOLANTS_SIMP, True)
        parser = Parser(manager, options)
        parser.parse(
            "(set-logic %s)" % normalize_logic_name(logic),
            True,
            False,
        )
        # Equalities between constant arrays make Bitwuzla give up with
        # `unknown` before it ever reaches interpolation. Stating them
        # elementwise is an equivalence and introduces no new symbols, so the
        # interpolant comes back already expressed in the original signature.
        expander = shared_array_expander(self.environment)
        formulas = [
            expand_array_equalities(formula, self.environment, expander)
            for formula in formulas
        ]
        # The blocks of a sequence interpolation overlap heavily, and so do the
        # calls: the naming state lives on the environment, so the renamer's
        # memoization survives this interpolator being discarded.
        state = interpolation_state(self.environment, "bitwuzla_itp")
        symbols = collect_symbols(formulas)
        native_formulas = state.rename(formulas, symbols)
        # Renaming is injective and total over the collected symbols, so this
        # is exactly what collecting from the renamed formulas would yield;
        # sorted the same way to keep the emitted script unchanged.
        native_symbols = sorted(
            (state.symbol_map[symbol] for symbol in symbols),
            key=lambda symbol: symbol.symbol_name(),
        )
        declarations = state.declarations(native_symbols)
        if declarations:
            parser.parse(repair_quoted_symbols(declarations), True, False)
        return (
            parser,
            parser.bitwuzla(),
            declarations,
            native_symbols,
            state,
            native_formulas,
        )

    def binary_interpolant(self, a: FNode, b: FNode):
        logic = script_logic(self._check_logic([a, b]), self.LOGICS,
                             self.logic)
        (parser, solver, declarations, symbols,
         state, native_formulas) = self._context([a, b], logic)
        a_term = parser.parse_term(state.smtlib(native_formulas[0]))
        b_term = parser.parse_term(state.smtlib(native_formulas[1]))
        solver.assert_formula(a_term, b_term)

        result = solver.check_sat()
        if result == Result.SAT:
            return None
        if result == Result.UNKNOWN:
            raise SolverReturnedUnknownResultError()

        interpolant = solver.get_interpolant([a_term])
        return term_to_pysmt(
            interpolant,
            logic,
            declarations,
            self.environment,
            symbols,
            state.symbol_map,
            state,
        )

    def sequence_interpolant(self, formulas: Sequence[FNode]):
        formulas = list(formulas)
        if len(formulas) < 2:
            raise PysmtValueError("interpolation needs at least 2 formulae")
        logic = script_logic(self._check_logic(formulas), self.LOGICS,
                             self.logic)

        (parser, solver, declarations, symbols,
         state, native_formulas) = self._context(formulas, logic)
        native_formulas = [
            parser.parse_term(state.smtlib(formula))
            for formula in native_formulas
        ]
        solver.assert_formula(*native_formulas)

        result = solver.check_sat()
        if result == Result.SAT:
            return None
        if result == Result.UNKNOWN:
            raise SolverReturnedUnknownResultError()

        native_interpolants = solver.get_interpolants(
            [[formula] for formula in native_formulas[:-1]]
        )
        return terms_to_pysmt(
            native_interpolants,
            logic,
            declarations,
            self.environment,
            symbols,
            state.symbol_map,
            state,
        )

    def _exit(self):
        pass

# EOC BitwuzlaInterpolator
