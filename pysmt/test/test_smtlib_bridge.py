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
"""Tests for the glue the SMT-LIB-based backends share.

Nothing here needs a solver: these pin the properties the native backends
rely on when they hand terms to a parser and read them back.
"""

from pysmt.shortcuts import And, Array, BV, Equals, Not, Symbol, is_unsat
from pysmt.logics import QF_LIA
from pysmt.solvers._array_rewriter import (ArrayEqualityExpander,
                                           expand_array_equalities)
from pysmt.solvers._smtlib_bridge import term_to_pysmt, terms_to_pysmt
from pysmt.environment import get_env
from pysmt.test import TestCase, skipIfSolverNotAvailable, main
from pysmt.typing import ArrayType, BVType, INT


class TestNativeTermBridge(TestCase):

    def test_arbitrary_precision_numeral(self):
        """A numeral too long for Python's int/str guard must survive.

        Python 3.11 caps decimal-to-int conversions at 4300 digits. Native
        interpolators do synthesize larger numerals, and without raising the
        cap the parser reads one as a string constant and builds an ill-typed
        formula out of it.
        """
        x = Symbol("huge_integer_x", INT)
        digits = "9" * 5000
        formula = term_to_pysmt(
            "(= huge_integer_x %s)" % digits,
            QF_LIA,
            "(declare-fun huge_integer_x () Int)",
            get_env(),
            [x],
        )

        value = formula.arg(1)
        self.assertTrue(value.is_int_constant())
        self.assertEqual(value.constant_value() % (10 ** 20), (10 ** 20) - 1)

    @skipIfSolverNotAvailable('cvc5')
    def test_arbitrary_precision_numeral_reaches_cvc5(self):
        """The same numeral has to survive the trip back to a backend."""
        x = Symbol("huge_integer_x", INT)
        digits = "9" * 5000
        formula = term_to_pysmt(
            "(= huge_integer_x %s)" % digits,
            QF_LIA,
            "(declare-fun huge_integer_x () Int)",
            get_env(),
            [x],
        )
        self.assertTrue(is_unsat(And(formula, Not(formula)),
                                 solver_name="cvc5", logic=QF_LIA))


class TestBatchedTermParsing(TestCase):
    """Parsing several terms at once must match parsing them one by one.

    The declaration block is re-read for every term when they are parsed
    separately, which makes an interpolant round trip quadratic in the length
    of the sequence. Batching is only safe if it is indistinguishable from the
    singular path, including the order of the results.
    """

    def test_batch_matches_one_by_one(self):
        x = Symbol("batch_x", INT)
        y = Symbol("|batch quoted y|", INT)
        declarations = "\n".join("(declare-fun %s () Int)" % name
                                 for name in ("batch_x", "|batch quoted y|"))
        terms = ["(= batch_x 1)",
                 "(< batch_x |batch quoted y|)",
                 "(= |batch quoted y| 7)"]

        batched = terms_to_pysmt(terms, QF_LIA, declarations, get_env(), [x, y])
        one_by_one = [term_to_pysmt(term, QF_LIA, declarations, get_env(),
                                    [x, y])
                      for term in terms]

        self.assertEqual(batched, one_by_one)
        self.assertEqual(len(batched), 3)
        # The quoted name survives the round trip rather than being split.
        self.assertIn(y, batched[2].get_free_variables())

    def test_empty_batch(self):
        self.assertEqual(terms_to_pysmt([], QF_LIA, "", get_env()), [])


class TestArrayEqualityExpansion(TestCase):
    """The rewrite that keeps constant arrays out of the native backends.

    These pin the two properties the backends rely on: that an array-free
    formula is returned untouched, and that nothing array-sorted survives in
    one that is rewritten.
    """

    def test_array_free_formula_is_unchanged(self):
        x = Symbol("expand_x", BVType(8))
        formula = And(Equals(x, BV(0, 8)), Not(Equals(x, BV(1, 8))))
        self.assertEqual(expand_array_equalities(formula, get_env()), formula)

    def test_array_free_formula_is_not_walked(self):
        """A formula without arrays must not pay for a walk at all.

        The interpolation shortcuts build a fresh expander per call, so a walk
        here is a full rebuild of the query's DAG on every interpolant. The
        environment's theory oracle answers instead, and it memoizes for the
        whole process.
        """
        x = Symbol("nowalk_x", BVType(8))
        formula = And(Equals(x, BV(0, 8)), Not(Equals(x, BV(1, 8))))

        expander = ArrayEqualityExpander(env=get_env())
        walked = []
        original_walk = expander.walk
        expander.walk = lambda f, **kw: (walked.append(f),
                                         original_walk(f, **kw))[1]

        self.assertEqual(expand_array_equalities(formula, get_env(), expander),
                         formula)
        self.assertEqual(walked, [])

    def test_constant_array_equality_loses_its_array_terms(self):
        index_type = BVType(2)
        array = Symbol("expand_a", ArrayType(index_type, BVType(2)))
        formula = Equals(array, Array(index_type, BV(0, 2)))

        expanded = expand_array_equalities(formula, get_env())

        # The equality becomes one conjunct per index, and each read of the
        # constant array folds to the constant itself, so no array-sorted
        # subterm is left for the backend to choke on.
        stc = get_env().stc
        for node in expanded.get_atoms():
            for sub in node.args():
                self.assertFalse(stc.get_type(sub).is_array_type())
        self.assertEqual(expanded.get_free_variables(), {array})

    def test_wide_index_sort_is_left_alone(self):
        """Refusing to expand keeps a wide index sort from blowing up."""
        index_type = BVType(32)
        array = Symbol("expand_wide", ArrayType(index_type, BVType(2)))
        formula = Equals(array, Array(index_type, BV(0, 2)))
        self.assertEqual(expand_array_equalities(formula, get_env()), formula)


if __name__ == '__main__':
    main()
