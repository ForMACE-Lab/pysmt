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
from pysmt.shortcuts import *
from pysmt.typing import ArrayType, BVType, REAL, INT
from pysmt.test import TestCase, skipIfSolverNotAvailable, main
from pysmt.exceptions import NoSolverAvailableError
from pysmt.logics import QF_ABV, QF_BV, UFLIRA


class TestInterpolation(TestCase):
    def test_selection(self):
        with self.assertRaises(NoSolverAvailableError):
            Interpolator(logic=UFLIRA)

        with self.assertRaises(NoSolverAvailableError):
            Interpolator(name="nonexistent")

    @skipIfSolverNotAvailable('msat')
    def test_binary_interpolant_msat(self):
        self._test_binary_interpolant('msat')

    @skipIfSolverNotAvailable('msat')
    def test_sequence_interpolant_msat(self):
        self._test_sequence_interpolant('msat')

    @skipIfSolverNotAvailable('cvc5')
    def test_binary_interpolant_cvc5(self):
        self._test_binary_interpolant('cvc5')

    @skipIfSolverNotAvailable('cvc5')
    def test_sequence_interpolant_cvc5(self):
        self._test_sequence_interpolant('cvc5')

    @skipIfSolverNotAvailable('smtinterpol')
    def test_binary_interpolant_smtinterpol(self):
        self._test_binary_interpolant('smtinterpol')

    @skipIfSolverNotAvailable('smtinterpol')
    def test_sequence_interpolant_smtinterpol(self):
        self._test_sequence_interpolant('smtinterpol')

    def _test_binary_interpolant(self, name):
        itp = Interpolator(name=name)
        self._bool_example(itp, True)
        self._real_example(itp, True)
        self._int_example(itp, True)

    def _test_sequence_interpolant(self, name):
        itp = Interpolator(name=name)
        self._bool_example(itp, False)
        self._real_example(itp, False)
        self._int_example(itp, False)

    @skipIfSolverNotAvailable('msat')
    def test_context(self):
        with Interpolator() as itp:
            self._bool_example(itp, False)

    def _bool_example(self, itp, binary):
        # Bool Example
        x, y, z = Symbol("bx"), Symbol("by"), Symbol("bz")

        a = And(x, Not(y))
        b = And(Implies(z, y), z)

        if binary:
            i = itp.binary_interpolant(a, b)
        else:
            i = itp.sequence_interpolant([a, b])

        self.assertTrue(i is not None)
        if not binary:
            self.assertTrue(hasattr(i, '__len__'))
            self.assertTrue(len(i) == 1)
            i = i[0]

        self.assertTrue(i.get_free_variables() == set([y]))
        self.assertValid(Implies(a, i))
        self.assertUnsat(And(i, b))


    def _real_example(self, itp, binary):
        # Real Example
        x, y, z = Symbol("rx", REAL), Symbol("ry", REAL), Symbol("rz", REAL)

        a = And(LE(x, Real(0)), LE(y, x))
        b = And(GE(y, z), Equals(z, Real(1)))

        if binary:
            i = itp.binary_interpolant(a, b)
        else:
            i = itp.sequence_interpolant([a, b])

        self.assertTrue(i is not None)
        if not binary:
            self.assertTrue(hasattr(i, '__len__'))
            self.assertTrue(len(i) == 1)
            i = i[0]

        self.assertTrue(i.get_free_variables() == set([y]))
        self.assertValid(Implies(a, i))
        self.assertUnsat(And(i, b))


    def _int_example(self, itp, binary):
        # Int Example
        x, y, z = Symbol("ix", INT), Symbol("iy", INT), Symbol("iz", INT)

        a = And(LE(x, Int(1)), LT(y, x))
        b = And(GE(y, z), GT(z, Int(0)))

        if binary:
            i = itp.binary_interpolant(a, b)
        else:
            i = itp.sequence_interpolant([a, b])

        self.assertTrue(i is not None)
        if not binary:
            self.assertTrue(hasattr(i, '__len__'))
            self.assertTrue(len(i) == 1)
            i = i[0]

        self.assertTrue(i.get_free_variables() == set([y]))
        self.assertValid(Implies(a, i))
        self.assertUnsat(And(i, b))


class TestBVInterpolation(TestCase):
    """Interpolation over bit-vectors and arrays.

    Two backends reach these theories, and by different means: Bitwuzla reads
    an interpolant off its own proof, cvc5 synthesizes one. Running both
    through the same examples turns a divergence into a failure rather than
    into a gap in the coverage.
    """

    @skipIfSolverNotAvailable('bitwuzla')
    def test_bv_interpolant_bitwuzla(self):
        self._bv_example('bitwuzla')

    @skipIfSolverNotAvailable('cvc5')
    def test_bv_interpolant_cvc5(self):
        self._bv_example('cvc5')

    @skipIfSolverNotAvailable('bitwuzla')
    def test_satisfiable_has_no_interpolant_bitwuzla(self):
        self._satisfiable_example('bitwuzla')

    @skipIfSolverNotAvailable('cvc5')
    def test_satisfiable_has_no_interpolant_cvc5(self):
        self._satisfiable_example('cvc5')

    @skipIfSolverNotAvailable('bitwuzla')
    def test_array_interpolant_bitwuzla(self):
        self._array_example('bitwuzla')

    @skipIfSolverNotAvailable('cvc5')
    def test_array_interpolant_cvc5(self):
        self._array_example('cvc5')

    @skipIfSolverNotAvailable('bitwuzla')
    def test_quoted_symbol_bitwuzla(self):
        self._quoted_symbol_example('bitwuzla')

    @skipIfSolverNotAvailable('cvc5')
    def test_quoted_symbol_cvc5(self):
        self._quoted_symbol_example('cvc5')

    @skipIfSolverNotAvailable('bitwuzla')
    def test_constant_array_equality_is_decided_bitwuzla(self):
        self._constant_array_is_decided('bitwuzla')

    @skipIfSolverNotAvailable('cvc5')
    def test_constant_array_equality_is_decided_cvc5(self):
        self._constant_array_is_decided('cvc5')

    @skipIfSolverNotAvailable('bitwuzla')
    def test_constant_array_interpolant_bitwuzla(self):
        """Interpolate across an array initialized to a constant.

        A state variable set to a constant array, stepped by a store and read
        back in the query. Bitwuzla answered `unknown` here before the
        equality was stated elementwise and, once it could decide the formula,
        crashed inside `get_interpolant` while an array term was still
        present, so this covers both halves of that fix.

        Bitwuzla only: cvc5's interpolator does not advertise the
        constant-array logics.
        """
        index_type = BVType(2)
        value_type = BVType(2)
        state = Symbol("bzla_ca_state", ArrayType(index_type, value_type))
        step = Symbol("bzla_ca_step", ArrayType(index_type, value_type))

        a = And(Equals(state, Array(index_type, BV(0, 2))),
                Equals(step, Store(state, BV(1, 2), BV(3, 2))))
        b = Not(Equals(Select(step, BV(1, 2)), BV(3, 2)))

        i = binary_interpolant(a, b, solver_name='bitwuzla', logic=QF_ABV)
        self._assert_interpolant(a, b, i, QF_ABV, 'bitwuzla')

    def _assert_interpolant(self, a, b, i, logic, name):
        self.assertIsNotNone(i)
        self.assertTrue(is_unsat(And(a, Not(i)), solver_name=name,
                                 logic=logic))
        self.assertTrue(is_unsat(And(i, b), solver_name=name, logic=logic))

    def _bv_example(self, name):
        x = Symbol("%s_x" % name, BVType(8))
        a = Equals(x, BV(0, 8))
        middle = Equals(x, BV(0, 8))
        b = Equals(x, BV(1, 8))

        i = binary_interpolant(a, b, solver_name=name, logic=QF_BV)
        self._assert_interpolant(a, b, i, QF_BV, name)

        seq = sequence_interpolant([a, middle, b], solver_name=name,
                                   logic=QF_BV)
        self.assertIsNotNone(seq)
        self.assertEqual(len(seq), 2)
        self._assert_interpolant(a, And(middle, b), seq[0], QF_BV, name)
        self._assert_interpolant(And(seq[0], middle), b, seq[1], QF_BV, name)

    def _satisfiable_example(self, name):
        x = Symbol("%s_sat_x" % name, BVType(8))
        f = Equals(x, BV(0, 8))
        self.assertIsNone(binary_interpolant(f, f, solver_name=name,
                                             logic=QF_BV))

    def _array_example(self, name):
        index_type = BVType(8)
        array = Symbol("%s_array" % name, ArrayType(index_type, BVType(8)))
        index = Symbol("%s_index" % name, index_type)
        selected = Select(array, index)
        a = Equals(selected, BV(0, 8))
        b = Equals(selected, BV(1, 8))

        i = binary_interpolant(a, b, solver_name=name, logic=QF_ABV)
        self._assert_interpolant(a, b, i, QF_ABV, name)

    def _quoted_symbol_example(self, name):
        """A name that is already ``|quoted|`` must survive the round trip."""
        state = Symbol("|%s_state|" % name, BVType(8))
        a = Equals(state, BV(0, 8))
        b = Equals(state, BV(1, 8))

        i = binary_interpolant(a, b, solver_name=name, logic=QF_BV)
        self._assert_interpolant(a, b, i, QF_BV, name)
        self.assertIn(state, i.get_free_variables())

    def _constant_array_is_decided(self, name):
        """An equality involving constant arrays must not defeat the backend.

        Both used to give up rather than answer: Bitwuzla returned `unknown`
        ("Equality over constant arrays not fully supported yet") and cvc5
        raised an internal error about write-chains connecting two different
        constant arrays. Over a finite index sort the equality is expressible
        elementwise, which both decide.

        The formula is unsatisfiable: four indices, but the left side can
        differ from its 0-default at only two of them and the right side from
        its 1-default at only one, so some index has 0 on the left and 1 on
        the right.
        """
        index_type = BVType(2)
        value_type = BVType(2)
        i = Symbol("%s_ca_i" % name, index_type)
        j = Symbol("%s_ca_j" % name, index_type)
        k = Symbol("%s_ca_k" % name, index_type)
        x = Symbol("%s_ca_x" % name, value_type)
        y = Symbol("%s_ca_y" % name, value_type)
        z = Symbol("%s_ca_z" % name, value_type)

        left = Store(Store(Array(index_type, BV(0, 2)), i, x), j, y)
        right = Store(Array(index_type, BV(1, 2)), k, z)

        self.assertTrue(is_unsat(Equals(left, right), solver_name=name,
                                 logic=QF_ABV))


if __name__ == '__main__':
    main()
