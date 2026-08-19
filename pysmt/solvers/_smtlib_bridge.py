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
"""SMT-LIB conversion helpers for native solver APIs.

Some native Python APIs provide a complete SMT-LIB parser but no public
term-to-term converter that pySMT can reuse.  This module keeps the small
amount of serialization glue needed by those APIs in one place.
"""

import re
import sys
from collections import OrderedDict
from io import StringIO
from threading import RLock
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pysmt.smtlib.commands as smtcmd
from pysmt.environment import Environment
from pysmt.exceptions import PysmtValueError
from pysmt.fnode import FNode
from pysmt.logics import Logic
from pysmt.smtlib.parser import SmtLibParser
from pysmt.typing import PySMTType, _FunctionType
from pysmt.walkers import IdentityDagWalker


_SIMPLE_SYMBOL = re.compile(
    r"^[A-Za-z_~!@$%^&*\-+=<>.?/][A-Za-z0-9_~!@$%^&*\-+=<>.?/@]*$"
)
_NESTED_QUOTED_SYMBOL = re.compile(r"\|\\\|([^|]*)\\\|\|")
_INT_DIGIT_LIMIT_LOCK = RLock()


def normalize_logic_name(logic) -> str:
    """Return the SMT-LIB spelling of a pySMT logic."""
    name = str(logic)
    if name.endswith("*"):
        name = name[:-1]
    if "t" in name:
        name = name.replace("t", "")
    if name == "QF_BOOL":
        return "QF_BV"
    if name == "BOOL":
        return "BV"
    return name


def integer_to_decimal(value: int) -> str:
    """Convert an arbitrary-size integer without Python's digit-limit guard."""
    if value == 0:
        return "0"

    sign = ""
    if value < 0:
        sign = "-"
        value = -value

    base = 10 ** 9
    chunks = []
    while value:
        value, remainder = divmod(value, base)
        chunks.append(remainder)

    return sign + str(chunks[-1]) + "".join(
        "%09d" % chunk for chunk in reversed(chunks[:-1])
    )


def repair_quoted_symbols(text: str) -> str:
    """Repair nested quoting emitted for names already wrapped in ``|...|``."""
    return _NESTED_QUOTED_SYMBOL.sub(r"|\1|", text)


def node_to_smtlib(node: FNode) -> str:
    """Serialize a pySMT node as a standalone SMT-LIB term."""
    try:
        text = node.to_smtlib(daggify=True)
    except TypeError:
        text = node.to_smtlib()
    return repair_quoted_symbols(text)


def _quote_symbol(name: str) -> str:
    if "|" in name or "\\" in name:
        raise ValueError(
            "Cannot SMT-LIB-quote symbol containing '|' or '\\\\': %r" % name
        )
    return "|%s|" % name


def escape_symbol(name: str) -> str:
    """Return a valid SMT-LIB identifier for a pySMT symbol name."""
    if len(name) >= 2 and name[0] == "|" and name[-1] == "|":
        return _quote_symbol(name[1:-1])
    if _SIMPLE_SYMBOL.match(name) is not None:
        return name
    return _quote_symbol(name)


def _sort_to_smtlib(tp: PySMTType) -> str:
    try:
        return tp.as_smtlib(False)
    except TypeError:
        return tp.as_smtlib()


def _get_int_max_str_digits() -> int:
    """The interpreter's int/str conversion limit, or 0 when there is none."""
    getter = getattr(sys, "get_int_max_str_digits", None)
    return getter() if getter is not None else 0


def _set_int_max_str_digits(limit: int) -> None:
    setter = getattr(sys, "set_int_max_str_digits", None)
    if setter is not None:
        setter(limit)


def script_logic(detected: Sequence[Logic],
                 logics: Sequence[Logic],
                 requested: Optional[Logic] = None) -> Logic:
    """The logic to announce in the ``(set-logic ...)`` of a native script.

    ``requested`` is what the caller asked the backend for, and it is kept
    whenever it covers the formulas -- a backend set up for a wider logic
    goes on using it.  It does not always cover them: ``Factory`` falls back
    to ``DEFAULT_INTERPOLATION_LOGIC`` when the caller names no logic, so a
    query over integers can arrive announced as ``QF_UFLRA``.  A backend then
    rejects every declaration in the script, and that is worse than an error:
    an empty assertion stack is satisfiable, which the caller reads as "there
    is no interpolant".  Announcing the logic the formulas are actually in
    keeps the script well formed.
    """
    if requested is not None and all(d <= requested for d in detected):
        return requested
    candidates = [logic for logic in logics
                  if all(d <= logic for d in detected)]
    if not candidates:
        raise PysmtValueError("No supported logic covers %s"
                              % ", ".join(str(d) for d in detected))
    best = candidates[0]
    for candidate in candidates[1:]:
        if candidate <= best:
            best = candidate
    return best


def collect_symbols(formulas: Iterable[FNode]) -> List[FNode]:
    """Collect free constants and uninterpreted function symbols.

    ``FreeVarsOracle.walk_function`` already yields the function symbol
    alongside the free variables of its arguments, so the oracle's result is
    exactly the set of names that need declaring. The oracle lives on the
    environment and memoizes per DAG node, which matters a great deal here:
    the model-checking engines re-serialize a conjunction that grows by one
    transition per bound, and an unmemoized traversal made that quadratic in
    the size of the unrolling.
    """
    symbols: Set[FNode] = set()
    for formula in formulas:
        symbols.update(formula.get_free_variables())
    return sorted(symbols, key=lambda symbol: symbol.symbol_name())


def symbol_declaration(symbol: FNode) -> str:
    """Build an SMT-LIB declaration for a pySMT symbol."""
    name = escape_symbol(symbol.symbol_name())
    tp = symbol.symbol_type()

    if tp.is_function_type():
        assert isinstance(tp, _FunctionType)
        args = " ".join(_sort_to_smtlib(arg) for arg in tp.param_types)
        return "(declare-fun %s (%s) %s)" % (
            name,
            args,
            _sort_to_smtlib(tp.return_type),
        )

    return "(declare-fun %s () %s)" % (name, _sort_to_smtlib(tp))


def declaration_script(
    formulas: Sequence[FNode],
    symbols: Optional[Sequence[FNode]] = None,
) -> Tuple[str, List[FNode]]:
    """Return declarations and their corresponding pySMT symbols.

    ``symbols`` lets a caller that has already collected them skip the
    traversal; the two are always collected from the same formulas.
    """
    if symbols is None:
        symbols = collect_symbols(formulas)
    else:
        symbols = list(symbols)
    return "\n".join(symbol_declaration(symbol) for symbol in symbols), symbols


def make_safe_symbol_map(
    formulas: Sequence[FNode],
    environment: Environment,
    prefix: str,
    symbols: Optional[Sequence[FNode]] = None,
) -> Dict[FNode, FNode]:
    """Map arbitrary pySMT names to injective, SMT-LIB-safe identifiers.

    ``symbols`` overrides the set to map, so a caller that already knows which
    symbols are new does not pay for a second traversal of ``formulas``. The
    encoding is per-symbol, so mapping a subset yields exactly the entries that
    mapping everything would have produced for those symbols.
    """
    manager = environment.formula_manager
    if symbols is None:
        symbols = collect_symbols(formulas)
    mapping = {}
    for symbol in symbols:
        encoded_name = symbol.symbol_name().encode("utf-8").hex()
        safe_name = "__pysmt_%s_%s" % (prefix, encoded_name)
        mapping[symbol] = manager.Symbol(safe_name, symbol.symbol_type())
    return mapping


class SymbolRenamer(IdentityDagWalker):
    """Rewrites symbols through a mapping, memoizing across calls.

    The mapping is read at walk time rather than captured per call, so a
    caller may keep one renamer alive and extend the mapping as new symbols
    are declared. Entries already present never change, and the memoization is
    keyed by DAG node, so previously rewritten nodes stay valid -- which is the
    point: consecutive bounds of an unrolling share nearly all of their
    subterms, and a fresh walker per call rebuilt all of them every time.
    """

    def __init__(self, symbol_map: Dict[FNode, FNode], env=None):
        IdentityDagWalker.__init__(self, env=env)
        self.symbol_map = symbol_map

    def walk_symbol(self, symbol, args, **kwargs):
        return self.symbol_map.get(symbol, symbol)

# EOC SymbolRenamer


def make_renamer(
    symbol_map: Dict[FNode, FNode],
    environment: Environment,
) -> SymbolRenamer:
    """Build a reusable renamer over a (possibly growing) symbol mapping."""
    return SymbolRenamer(symbol_map, env=environment)


def rename_symbols(
    formula: FNode,
    symbol_map: Dict[FNode, FNode],
    environment: Environment,
    renamer: Optional[SymbolRenamer] = None,
) -> FNode:
    """Rename constants and function symbols without term-substitution limits.

    Passing ``renamer`` reuses its memoization across formulas; omitting it
    keeps the original one-shot behaviour.
    """
    if not symbol_map:
        return formula
    if renamer is None:
        renamer = SymbolRenamer(symbol_map, env=environment)
    return renamer.walk(formula)


#: Attribute holding the per-environment interpolation states, keyed by the
#: backend's name prefix.
_INTERPOLATION_STATES = "_pysmt_interpolation_states"

#: How many serialized formulas an :class:`InterpolationState` keeps. Small on
#: purpose -- see :meth:`InterpolationState.smtlib`.
_SMTLIB_CACHE_ENTRIES = 4


class InterpolationState:
    """Naming state shared by every interpolation call on one environment.

    ``Factory.binary_interpolant`` builds a fresh :class:`Interpolator` for
    each call and drops it again, so anything an interpolator memoizes on
    itself is thrown away before the next call can use it.  The model-checking
    engines call those shortcuts in a loop over an unrolling whose blocks
    overlap almost entirely -- precisely the workload memoization exists for.
    This state therefore hangs off the *environment*, which outlives the
    individual interpolators, and is keyed by backend prefix so two backends
    never share a signature.

    Nothing here changes what a caller sees, only what it pays:

    * a safe name is a per-symbol hex encoding, so the entry a symbol gets
      from the shared map is the entry a one-shot call would have given it;
    * entries are only ever added, never rewritten, so the renamers' DAG
      memoization stays valid as the map grows.  That needs every walk to be
      preceded by the extension covering its symbols, which :meth:`rename`
      guarantees -- and it is enough, because a node memoized earlier can only
      mention symbols that were already free in the formula it was memoized
      from, hence already mapped;
    * a declaration is a function of one symbol alone.
    """

    def __init__(self, environment: Environment, prefix: str):
        self.environment = environment
        self.prefix = prefix
        self.symbol_map: Dict[FNode, FNode] = {}
        self.reverse_map: Dict[FNode, FNode] = {}
        self._renamer = SymbolRenamer(self.symbol_map, env=environment)
        #: Maps native-safe names back to the originals, for the interpolant.
        self.reverse_renamer = SymbolRenamer(self.reverse_map, env=environment)
        self._declarations: Dict[FNode, str] = {}
        self._smtlib: "OrderedDict[FNode, str]" = OrderedDict()

    def smtlib(self, formula: FNode) -> str:
        """The SMT-LIB text of ``formula``, remembering the last few.

        A binary interpolation re-serializes its unchanged half on every inner
        iteration of the engine, and the halves are memoized nodes, so the
        repeat is an exact hit.  The cache stays deliberately tiny: these
        strings run to megabytes on a deep unrolling, and only the most recent
        ones are ever asked for again.
        """
        cached = self._smtlib.get(formula)
        if cached is not None:
            self._smtlib.move_to_end(formula)
            return cached
        text = node_to_smtlib(formula)
        self._smtlib[formula] = text
        while len(self._smtlib) > _SMTLIB_CACHE_ENTRIES:
            self._smtlib.popitem(last=False)
        return text

    def rename(
        self,
        formulas: Sequence[FNode],
        symbols: Iterable[FNode],
    ) -> List[FNode]:
        """Map ``symbols`` to safe names and rewrite ``formulas`` with them."""
        manager = self.environment.formula_manager
        symbol_map = self.symbol_map
        for symbol in symbols:
            if symbol in symbol_map:
                continue
            safe = manager.Symbol(
                "__pysmt_%s_%s"
                % (self.prefix, symbol.symbol_name().encode("utf-8").hex()),
                symbol.symbol_type(),
            )
            symbol_map[symbol] = safe
            self.reverse_map[safe] = symbol
        return [self._renamer.walk(formula) for formula in formulas]

    def declarations(self, symbols: Sequence[FNode]) -> str:
        """The declaration block for ``symbols``, one line built per symbol."""
        cache = self._declarations
        lines = []
        for symbol in symbols:
            line = cache.get(symbol)
            if line is None:
                line = symbol_declaration(symbol)
                cache[symbol] = line
            lines.append(line)
        return "\n".join(lines)

# EOC InterpolationState


def interpolation_state(
    environment: Environment,
    prefix: str,
) -> InterpolationState:
    """The :class:`InterpolationState` for ``prefix`` on ``environment``."""
    states = getattr(environment, _INTERPOLATION_STATES, None)
    if states is None:
        states = {}
        setattr(environment, _INTERPOLATION_STATES, states)
    state = states.get(prefix)
    if state is None:
        state = InterpolationState(environment, prefix)
        states[prefix] = state
    return state


def restore_mapped_symbols(
    formula: FNode,
    symbol_map: Dict[FNode, FNode],
    environment: Environment,
) -> FNode:
    """Replace native-safe symbols in a parsed term with original symbols."""
    if not symbol_map:
        return formula
    reverse_map = {safe: original for original, safe in symbol_map.items()}
    return rename_symbols(formula, reverse_map, environment)


def _quoted_originals(
    original_symbols: Iterable[FNode],
) -> Dict[Tuple[str, PySMTType], FNode]:
    """Index the explicitly ``|quoted|`` symbols by their unquoted name."""
    originals: Dict[Tuple[str, PySMTType], FNode] = {}
    for symbol in original_symbols:
        name = symbol.symbol_name()
        if len(name) >= 2 and name[0] == "|" and name[-1] == "|":
            originals[(name[1:-1], symbol.symbol_type())] = symbol
    return originals


def _restore_originals(
    formula: FNode,
    originals: Dict[Tuple[str, PySMTType], FNode],
) -> FNode:
    substitutions = {}
    for symbol in formula.get_free_variables():
        original = originals.get((symbol.symbol_name(), symbol.symbol_type()))
        if original is not None and original != symbol:
            substitutions[symbol] = original

    if substitutions:
        return formula.substitute(substitutions)
    return formula


def restore_original_symbols(
    formula: FNode,
    original_symbols: Iterable[FNode],
) -> FNode:
    """Restore pySMT names that were explicitly represented as ``|name|``."""
    return _restore_originals(formula, _quoted_originals(original_symbols))


def terms_to_pysmt(
    terms: Sequence,
    logic,
    declarations: str,
    environment: Environment,
    original_symbols: Iterable[FNode] = (),
    symbol_map: Optional[Dict[FNode, FNode]] = None,
    state: Optional["InterpolationState"] = None,
) -> List[FNode]:
    """Parse several native terms back into ``environment``, in one pass.

    The declaration block is proportional to the size of the whole unrolling,
    and a sequence interpolation of bound k hands back k terms. Parsing the
    block once per term therefore made the round trip quadratic in the bound --
    on a QF_BV ISMC run it dominated everything but the native solving itself.
    Asserting all the terms in a single script parses the declarations once,
    and the per-term work (which is genuinely per-term) is unchanged.

    The reverse symbol maps are likewise built once rather than per term.
    ``state`` supplies them from the caller's shared naming state instead,
    which also carries the renamer's memoization across calls. The parsed term
    only mentions symbols from ``declarations``, i.e. exactly the ones
    ``symbol_map`` covers, so the wider shared map rewrites it identically.
    """
    terms = list(terms)
    if not terms:
        return []

    parts = ["(set-logic %s)\n" % normalize_logic_name(logic)]
    if declarations:
        parts.append(declarations + "\n")
    for term in terms:
        parts.append("(assert %s)\n" % repair_quoted_symbols(str(term)))
    script = "".join(parts)

    # Python 3.11+ limits decimal-to-int conversions to 4300 digits by
    # default. Native interpolators can legitimately synthesize larger
    # arbitrary-precision numerals. Without this scoped adjustment,
    # SmtLibParser mistakes such numerals for string constants and later
    # produces ill-typed formulae.
    with _INT_DIGIT_LIMIT_LOCK:
        previous_limit = _get_int_max_str_digits()
        raised_limit = 0 < previous_limit < len(script)
        if raised_limit:
            _set_int_max_str_digits(len(script))
        try:
            parser = SmtLibParser(environment)
            parsed = parser.get_script(StringIO(script))
        finally:
            if raised_limit:
                _set_int_max_str_digits(previous_limit)

    # The parser was built on `environment`, so the assertions are already
    # nodes of its formula manager and need no contextualizing walk.
    formulas = [
        command.args[0]
        for command in parsed.filter_by_command_name([smtcmd.ASSERT])
    ]
    if len(formulas) != len(terms):
        raise PysmtValueError(
            "expected %d assertions from the parsed term script, got %d"
            % (len(terms), len(formulas))
        )

    originals = _quoted_originals(original_symbols)
    renamer: Optional[SymbolRenamer]
    if state is not None:
        reverse_map = state.reverse_map
        renamer = state.reverse_renamer
    else:
        reverse_map = {safe: original
                       for original, safe in (symbol_map or {}).items()}
        renamer = (make_renamer(reverse_map, environment)
                   if reverse_map else None)

    restored = []
    for formula in formulas:
        formula = _restore_originals(formula, originals)
        if reverse_map:
            formula = rename_symbols(formula, reverse_map, environment, renamer)
        restored.append(formula)
    return restored


def term_to_pysmt(
    term,
    logic,
    declarations: str,
    environment: Environment,
    original_symbols: Iterable[FNode] = (),
    symbol_map: Optional[Dict[FNode, FNode]] = None,
    state: Optional["InterpolationState"] = None,
) -> FNode:
    """Parse a native solver term back into the requested pySMT environment."""
    return terms_to_pysmt(
        [term],
        logic,
        declarations,
        environment,
        original_symbols,
        symbol_map,
        state,
    )[0]
