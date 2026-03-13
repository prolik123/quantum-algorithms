import unittest
from unittest.mock import patch
from qiskit_aer import Aer

from src.algorithms.quadratic_congruence import (
    solve_quadratic_congruence,
    solve_quadratic_congruence_and_get_probabilities,
    _generate_congruence_logic_string,
    NoDiscreteRootsError
)


class TestNumberTheoryOracleLogic(unittest.TestCase):
    """
    Tests the classical logic generation for the quantum oracle.
    These tests are fast as they don't invoke the quantum simulator.
    """

    def test_generate_logic_string_success(self):
        # Problem: x^2 ≡ 1 (mod 5). Search space: 3 qubits (0 to 7)
        # Roots in [0, 7]: 1^2=1, 4^2=16≡1, 6^2=36≡1.
        logic_expr, num_solutions = _generate_congruence_logic_string(a=1, m=5, num_qubits=3)

        self.assertEqual(num_solutions, 3)
        # 1 is 001, 4 is 100, 6 is 110.
        # Since Qiskit reads right-to-left, the string should contain these configurations.
        self.assertIn('x0', logic_expr)
        self.assertIn('x1', logic_expr)
        self.assertIn('x2', logic_expr)

    def test_generate_logic_string_no_solutions(self):
        # Problem: x^2 ≡ 2 (mod 5). This has no discrete roots.
        with self.assertRaises(NoDiscreteRootsError) as context:
            _generate_congruence_logic_string(a=2, m=5, num_qubits=3)

        self.assertTrue("No solutions exist" in str(context.exception))


class TestPostProcessingAndThresholds(unittest.TestCase):
    """
    Tests the threshold filtering logic by mocking the quantum simulation results.
    """

    @patch('src.algorithms.quadratic_congruence.solve_quadratic_congruence_and_get_probabilities')
    def test_threshold_filtering(self, mock_get_probs):
        # Arrange: Mock the quantum results.
        # States 1 and 4 are high probability (roots). States 2 and 3 are noise.
        mock_get_probs.return_value = {
            1: 950,
            4: 930,
            2: 12,
            3: 8
        }

        # Act: Run the main function
        valid_roots = solve_quadratic_congruence(a=1, m=5)

        # Assert: Only 1 and 4 should pass the (950 / 2 = 475) threshold
        self.assertEqual(valid_roots, [1, 4])

    @patch('src.algorithms.quadratic_congruence.solve_quadratic_congruence_and_get_probabilities')
    def test_threshold_empty_results(self, mock_get_probs):
        # Arrange: Simulator returns nothing (edge case)
        mock_get_probs.return_value = {}

        # Act
        valid_roots = solve_quadratic_congruence(a=1, m=5)

        # Assert
        self.assertEqual(valid_roots, [])


class TestQuantumIntegration(unittest.TestCase):
    """
    End-to-End tests that actually run the Qiskit simulator.
    We use small modulo values to keep the test suite execution time low.
    """

    def setUp(self):
        # Reuse a single simulator instance for speed
        self.simulator = Aer.get_backend('aer_simulator')
        self.test_shots = 512  # Lower shots for faster testing

    def test_end_to_end_optimized_grover(self):
        # Test x^2 ≡ 1 (mod 5). Expected roots in 3 qubits (0-7): [1, 4, 6]
        roots = solve_quadratic_congruence(
            a=1, m=5,
            shots=self.test_shots,
            use_optimized=True,
            simulator=self.simulator
        )
        self.assertListEqual(roots, [1, 4, 6])

    def test_end_to_end_manual_grover(self):
        # Test x^2 ≡ 1 (mod 5) using our manual diffuser
        roots = solve_quadratic_congruence(
            a=1, m=5,
            shots=self.test_shots,
            use_optimized=False,
            simulator=self.simulator
        )
        self.assertListEqual(roots, [1, 4, 6])

    def test_custom_qubit_override(self):
        # Test x^2 ≡ 1 (mod 5), but force 4 qubits (search space 0-15)
        # Roots in [0, 15] for mod 5 are: 1, 4, 6, 9, 11, 14
        expected_roots = [1, 4, 6, 9, 11, 14]

        roots = solve_quadratic_congruence(
            a=1, m=5,
            num_qubits=4,
            shots=2048,  # Increase shots because search space is larger
            use_optimized=True,
            simulator=self.simulator
        )
        self.assertListEqual(roots, expected_roots)

    def test_invalid_qubit_count_raises_error(self):
        # Modulo 15 requires at least 4 qubits (to represent 14)
        # Passing 3 should raise the ValueError
        with self.assertRaises(ValueError):
            solve_quadratic_congruence_and_get_probabilities(a=1, m=15, num_qubits=3)


class TestExhaustiveQuadraticCongruence(unittest.TestCase):
    """
    Exhaustive test suite for values up to 50.
    """

    def setUp(self):
        self.simulator = Aer.get_backend('aer_simulator')

    def _get_classical_roots(self, a: int, m: int, num_qubits: int) -> list:
        """
        A purely classical helper to find the absolute truth.
        Iterates through the search space and returns the exact mathematical roots.
        """
        search_space = 2 ** num_qubits
        return [x for x in range(search_space) if (x ** 2) % m == a % m]

    def test_exhaustive_quantum_execution_m_and_a_under_30(self):
        """
        HEAVY TEST: Actually runs the quantum simulator for all 'm' up to 30 and a's.
        """

        # Test all modulo values from 3 up to 50
        for m in range(3, 31):
            for a_val in range(1, m):
                with self.subTest(a=a_val, m=m):
                    num_qubits = (m - 1).bit_length()
                    expected_roots = self._get_classical_roots(a_val, m, num_qubits)

                    try:
                        quantum_roots = solve_quadratic_congruence(
                            a=a_val,
                            m=m,
                            use_optimized=True,
                            simulator=self.simulator
                        )

                        self.assertListEqual(quantum_roots, expected_roots)

                    except ValueError as e:
                        self.fail(f"Quantum simulation failed for a={a_val}, m={m}: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)