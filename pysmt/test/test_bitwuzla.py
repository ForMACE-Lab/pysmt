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
from pysmt.shortcuts import BV, Equals, Not, Select, Solver, Symbol
from pysmt.logics import QF_ABV, QF_BV
from pysmt.test import TestCase, skipIfSolverNotAvailable, main
from pysmt.typing import ArrayType, BVType


class TestBitwuzla(TestCase):

    @skipIfSolverNotAvailable('bitwuzla')
    def test_incremental_solving_and_model(self):
        x = Symbol("bitwuzla_x", BVType(8))
        value = BV(10, 8)

        with Solver(name="bitwuzla", logic=QF_BV) as solver:
            solver.add_assertion(Equals(x, value))
            self.assertTrue(solver.solve())
            self.assertEqual(solver.get_value(x), value)

            solver.push()
            solver.add_assertion(Not(Equals(x, value)))
            self.assertFalse(solver.solve())
            solver.pop()

            self.assertTrue(solver.solve())

    @skipIfSolverNotAvailable('bitwuzla')
    def test_values_over_arrays(self):
        array = Symbol("bitwuzla_a", ArrayType(BVType(4), BVType(8)))
        index = Symbol("bitwuzla_i", BVType(4))

        with Solver(name="bitwuzla", logic=QF_ABV) as solver:
            solver.add_assertion(Equals(index, BV(1, 4)))
            solver.add_assertion(Equals(Select(array, index), BV(3, 8)))
            self.assertTrue(solver.solve())
            self.assertEqual(solver.get_value(index), BV(1, 4))
            self.assertEqual(solver.get_value(Select(array, index)), BV(3, 8))


if __name__ == '__main__':
    main()
