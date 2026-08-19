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
from pysmt.exceptions import InternalSolverError
from pysmt.logics import QF_ALIA, QF_LIA, QF_LRA, QF_UFLIA
from pysmt.shortcuts import (And, Equals, Function, Int, LE, Not, Real, Select,
                             Solver, Symbol, binary_interpolant, is_unsat,
                             sequence_interpolant)
from pysmt.test import TestCase, skipIfSolverNotAvailable, main
from pysmt.typing import ArrayType, FunctionType, INT, REAL


class TestSMTInterpol(TestCase):

    def assert_interpolant(self, a, b, interpolant, logic):
        self.assertIsNotNone(interpolant)
        self.assertTrue(is_unsat(And(a, Not(interpolant)),
                                 solver_name="smtinterpol", logic=logic))
        self.assertTrue(is_unsat(And(interpolant, b),
                                 solver_name="smtinterpol", logic=logic))

    @skipIfSolverNotAvailable('smtinterpol')
    def test_incremental_solver(self):
        x = Symbol("smtinterpol_solver_x", INT)
        zero = Equals(x, Int(0))

        with Solver(name="smtinterpol", logic=QF_LIA) as solver:
            solver.add_assertion(zero)
            self.assertTrue(solver.solve())
            self.assertEqual(solver.get_value(x), Int(0))
            solver.push()
            solver.add_assertion(Equals(x, Int(1)))
            self.assertFalse(solver.solve())
            solver.pop()
            self.assertTrue(solver.solve())

    @skipIfSolverNotAvailable('smtinterpol')
    def test_binary_and_sequence_lia(self):
        x = Symbol("smtinterpol_lia_x", INT)
        a = Equals(x, Int(0))
        middle = LE(x, Int(0))
        b = Equals(x, Int(1))

        interpolant = binary_interpolant(a, b, solver_name="smtinterpol",
                                         logic=QF_LIA)
        self.assert_interpolant(a, b, interpolant, QF_LIA)

        sequence = sequence_interpolant([a, middle, b],
                                        solver_name="smtinterpol",
                                        logic=QF_LIA)
        self.assertIsNotNone(sequence)
        self.assertEqual(len(sequence), 2)
        self.assert_interpolant(a, And(middle, b), sequence[0], QF_LIA)
        self.assert_interpolant(And(sequence[0], middle), b, sequence[1],
                                QF_LIA)

    @skipIfSolverNotAvailable('smtinterpol')
    def test_binary_lra(self):
        x = Symbol("smtinterpol_lra_x", REAL)
        a = LE(x, Real(0))
        b = LE(Real(1), x)

        interpolant = binary_interpolant(a, b, solver_name="smtinterpol",
                                         logic=QF_LRA)
        self.assert_interpolant(a, b, interpolant, QF_LRA)

    @skipIfSolverNotAvailable('smtinterpol')
    def test_array_interpolation(self):
        array = Symbol("smtinterpol_array", ArrayType(INT, INT))
        index = Symbol("smtinterpol_index", INT)
        selected = Select(array, index)
        a = Equals(selected, Int(0))
        b = Equals(selected, Int(1))

        interpolant = binary_interpolant(a, b, solver_name="smtinterpol",
                                         logic=QF_ALIA)
        self.assert_interpolant(a, b, interpolant, QF_ALIA)

    @skipIfSolverNotAvailable('smtinterpol')
    def test_uninterpreted_function_interpolation(self):
        x = Symbol("smtinterpol_uf_x", INT)
        function = Symbol("smtinterpol_uf", FunctionType(INT, [INT]))
        application = Function(function, [x])
        a = Equals(application, Int(0))
        b = Equals(application, Int(1))

        interpolant = binary_interpolant(a, b, solver_name="smtinterpol",
                                         logic=QF_UFLIA)
        self.assert_interpolant(a, b, interpolant, QF_UFLIA)

    @skipIfSolverNotAvailable('smtinterpol')
    def test_quoted_symbol_identity_is_preserved(self):
        state = Symbol("|smtinterpol state|", INT)
        a = Equals(state, Int(0))
        b = Equals(state, Int(1))

        interpolant = binary_interpolant(a, b, solver_name="smtinterpol",
                                         logic=QF_LIA)
        self.assert_interpolant(a, b, interpolant, QF_LIA)
        self.assertIn(state, interpolant.get_free_variables())

    @skipIfSolverNotAvailable('smtinterpol')
    def test_sat_problem_has_no_interpolant(self):
        x = Symbol("smtinterpol_sat_x", INT)
        formula = Equals(x, Int(0))
        self.assertIsNone(binary_interpolant(formula, formula,
                                             solver_name="smtinterpol",
                                             logic=QF_LIA))

    @skipIfSolverNotAvailable('smtinterpol')
    def test_rejected_script_is_not_read_as_satisfiable(self):
        """A refused command must raise, never look like "no interpolant".

        SMTInterpol reports a command it cannot accept with an ``(error ...)``
        response and carries on. The assertion is then missing from the stack,
        and an empty stack is satisfiable -- so trusting the status line turns
        a rejected script into "this query has no interpolant".
        """
        from pysmt.solvers.smtinterpol import _interpolant_terms

        self.assertEqual(_interpolant_terms("unsat\n((<= x 0))\n"),
                         ["(<= x 0)"])
        self.assertEqual(_interpolant_terms("sat\n"), [])
        # Asking for the interpolants of a satisfiable query is refused, and
        # that refusal comes after the status: it is not a rejected script.
        refused = 'sat\n(error "get-interpolants: status is sat")\n'
        self.assertEqual(_interpolant_terms(refused), [])
        with self.assertRaises(InternalSolverError):
            _interpolant_terms('(error "Undeclared sort (Int 0)")\nsat\n')


if __name__ == "__main__":
    main()
