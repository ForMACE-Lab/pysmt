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
"""Extensional expansion of array equalities over finite index sorts.

Both Bitwuzla and cvc5 are incomplete on *equalities between arrays* when a
constant array is involved.  Bitwuzla answers ``unknown`` outright ("Equality
over constant arrays not fully supported yet") and cvc5 raises "Array theory
solver does not yet support write-chains connecting two different constant
arrays".  Neither is a resource problem: both give up in a fraction of a
second, which in a model-checking loop turns straight into an ``UNKNOWN``
verdict for the whole task.

Selects and stores over those same arrays are handled fine by both solvers.
Only the equality is the sore point, and when the index sort is finite -- as it
always is for QF_ABV, where indices are bit-vectors -- an array equality can be
stated without ever comparing two arrays:

    a = b   <=>   /\\_{i in indices} select(a, i) = select(b, i)

which is just extensionality instantiated over the whole (finite) index
domain.  The rewrite is therefore an equivalence, not an approximation: it
preserves satisfiability exactly, so a verdict reached through the rewritten
formula is the verdict of the original.

Two properties make this cheap and safe to apply unconditionally:

* it introduces no new symbols, so an interpolant computed over rewritten
  formulas is already expressed in the original signature and needs no
  translation back;
* it only fires on array-sorted equalities, so a formula without arrays --
  every QF_BV task -- is skipped outright on the environment's theory oracle,
  and never pays for a walk it could only return unchanged.

Expansion is refused when the index domain is larger than ``max_entries``, so
a wide index sort degrades to the previous behaviour instead of blowing up the
formula.
"""

from typing import Dict, List, Optional

from pysmt.environment import Environment
from pysmt.exceptions import PysmtException
from pysmt.fnode import FNode
from pysmt.typing import PySMTType, _ArrayType, _BVType
from pysmt.walkers import IdentityDagWalker


# Index domains up to this size are expanded.  Bit-vector indices in practice
# are a handful of bits wide, so this leaves ample headroom while still
# refusing anything that would grow the formula unreasonably.
DEFAULT_MAX_ENTRIES = 256


class ArrayEqualityExpander(IdentityDagWalker):
    """Replace array equalities by elementwise equalities.

    The walker memoizes per DAG node like any ``IdentityDagWalker``, so one
    instance can be kept alive across the successive bounds of an unrolling:
    consecutive queries share nearly all of their subterms and each shared node
    is rewritten once.
    """

    def __init__(self, env=None, max_entries: int = DEFAULT_MAX_ENTRIES):
        IdentityDagWalker.__init__(self, env=env)
        self.max_entries = max_entries

    def _index_domain(self, index_type: PySMTType) -> Optional[List[FNode]]:
        """Enumerate an index sort, or ``None`` if that is unreasonable."""
        if not index_type.is_bv_type():
            return None
        assert isinstance(index_type, _BVType)
        size = 1 << index_type.width
        if size > self.max_entries:
            return None
        return [self.mgr.BV(value, index_type.width) for value in range(size)]

    def _expand(self, left: FNode, right: FNode) -> Optional[FNode]:
        """Expand an array equality, or return ``None`` to leave it alone."""
        array_type = self.env.stc.get_type(left)
        if not array_type.is_array_type():
            return None
        assert isinstance(array_type, _ArrayType)
        indices = self._index_domain(array_type.index_type)
        if indices is None:
            return None

        conjuncts = []
        for index in indices:
            left_element = self._select(left, index)
            right_element = self._select(right, index)
            # Arrays of arrays expand recursively; EqualsOrIff covers the
            # Boolean-element case, where SMT-LIB equality is Iff.
            nested = self._expand(left_element, right_element)
            if nested is None:
                nested = self.mgr.EqualsOrIff(left_element, right_element)
            conjuncts.append(nested)
        return self.mgr.And(conjuncts)

    def walk_equals(self, formula, args, **kwargs):
        expanded = self._expand(args[0], args[1])
        if expanded is None:
            return IdentityDagWalker.walk_equals(self, formula, args, **kwargs)
        return expanded

    def _select(self, array: FNode, index: FNode) -> FNode:
        """Build ``select(array, index)``, resolving it where possible.

        Reads are folded through constant arrays and write chains so that the
        array term itself drops out of the result.  That matters beyond
        tidiness: Bitwuzla decides these formulas but then *crashes* inside
        ``get_interpolant`` while an array term is still present, so folding
        the read is what keeps interpolation usable, not merely faster.
        """
        while True:
            if array.is_array_value():
                assignments = array.array_value_assigned_values_map()
                if not assignments:
                    # Constant everywhere: the index is irrelevant.
                    return array.array_value_default()
                if index.is_constant():
                    return assignments.get(index, array.array_value_default())
                # Symbolic index into an array with explicit entries: pick them
                # out with an if-then-else chain over the constant part.
                result = array.array_value_default()
                for key, value in sorted(assignments.items(), key=str):
                    result = self.mgr.Ite(
                        self.mgr.EqualsOrIff(index, key), value, result
                    )
                return result

            if array.is_store():
                base, written_index, written_value = array.args()
                if index.is_constant() and written_index.is_constant():
                    # Both concrete: the read either hits this write or skips
                    # past it, with no case split needed.
                    if written_index == index:
                        return written_value
                    array = base
                    continue
                return self.mgr.Ite(
                    self.mgr.EqualsOrIff(index, written_index),
                    written_value,
                    self._select(base, index),
                )

            return self.mgr.Select(array, index)

    def walk_array_select(self, formula, args, **kwargs):
        return self._select(args[0], args[1])

# EOC ArrayEqualityExpander


def make_array_expander(
    environment: Environment,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> ArrayEqualityExpander:
    """Build a reusable expander bound to ``environment``."""
    return ArrayEqualityExpander(env=environment, max_entries=max_entries)


# Attribute holding the per-environment expanders, keyed by ``max_entries``.
_SHARED_EXPANDERS = "_pysmt_shared_array_expanders"


def shared_array_expander(
    environment: Environment,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> ArrayEqualityExpander:
    """An expander whose memoization outlives its individual callers.

    The rewrite is a pure function of the node and the entry bound, so one
    expander per ``(environment, max_entries)`` returns exactly what a private
    one would.  Sharing matters because the interpolation shortcuts build --
    and drop -- an interpolator per call: a private expander is cold every
    time, and re-walks an unrolling that grew by one transition.
    """
    expanders: Optional[Dict[int, ArrayEqualityExpander]] = getattr(
        environment, _SHARED_EXPANDERS, None)
    if expanders is None:
        expanders = {}
        setattr(environment, _SHARED_EXPANDERS, expanders)
    expander = expanders.get(max_entries)
    if expander is None:
        expander = ArrayEqualityExpander(env=environment,
                                        max_entries=max_entries)
        expanders[max_entries] = expander
    return expander


def expand_array_equalities(
    formula: FNode,
    environment: Environment,
    expander: Optional[ArrayEqualityExpander] = None,
) -> FNode:
    """Return ``formula`` with array equalities stated elementwise.

    Passing ``expander`` reuses its memoization across formulas.  The result is
    logically equivalent to the input and over the same symbols.

    The rewrite is an enabler, never a requirement: a formula it cannot walk --
    one carrying a custom operator with no registered handler, say -- is
    returned untouched so the backend's own converter reports the problem in
    its own terms.  Swallowing the error here is safe precisely because the
    output is only ever an equivalent of the input; declining to rewrite costs
    completeness on constant arrays and nothing else.
    """
    try:
        # An array-free formula is returned unchanged by the walk anyway, so
        # this guard is only about cost -- but that cost is not negligible.
        # The interpolation shortcuts build a fresh Interpolator (hence a cold
        # expander) per call, so without the guard every interpolant of every
        # bound rebuilt the entire unrolling DAG. The theory oracle instead
        # lives on the environment and memoizes for the whole process, so a
        # QF_BV query only ever walks nodes it has not seen before.
        if not environment.theoryo.get_theory(formula).arrays:
            return formula
        if expander is None:
            expander = ArrayEqualityExpander(env=environment)
        return expander.walk(formula)
    except PysmtException:
        return formula
