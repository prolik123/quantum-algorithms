from qiskit.circuit.library import PhaseOracleGate
from qiskit_aer import Aer

from src.algorithms.grover import GroverAlgorithmManual, GroverAlgorithmOptimized


class NoDiscreteRootsError(Exception):
    """
    Custom exception raised when a quadratic congruence has no mathematical solutions
    within the given modulo search space.
    """
    pass


def solve_quadratic_congruence(a: int, m: int, num_qubits: int = None,
                               shots: int = None, use_optimized: bool = False, simulator=None):
    """
    Solves the discrete root problem x^2 ≡ a (mod m) using Grover's algorithm and
    retrieves the results. The results are chosen probability is at least half of
    the maximal probability in the set.

    :param a: The target remainder.
    :param m: The modulo.
    :param num_qubits: Optional. Number of qubits. If None, computed automatically based on 'm'.
    :param shots: Optional. Number of simulation shots. If None, computed dynamically.
    :param use_optimized: If True, uses Qiskit's built-in Grover operator.
    :param simulator: Aer backend simulator.
    :return: A list of integer solutions that are the discrete roots.
    """

    try:
        results = solve_quadratic_congruence_and_get_probabilities(
            a, m, num_qubits, shots, use_optimized, simulator
        )
    except NoDiscreteRootsError:
        return []

    max_count = max(results.values()) if results else 0
    threshold = max_count / 2

    valid_roots = [
        state_dec for state_dec, count in results.items()
        if count >= threshold and (state_dec * state_dec) % m == a
    ]

    valid_roots.sort()
    return valid_roots


def solve_quadratic_congruence_and_get_probabilities(a: int, m: int, num_qubits: int = None,
        shots: int = None, use_optimized: bool = False, simulator = None) -> dict:
    """
    Solves the discrete root problem x^2 ≡ a (mod m) using Grover's algorithm.

    :param a: The target remainder.
    :param m: The modulo.
    :param num_qubits: Optional. Number of qubits. If None, computed automatically based on 'm'.
    :param shots: Optional. Number of simulation shots. If None, computed dynamically.
    :param use_optimized: If True, uses Qiskit's built-in Grover operator.
    :param simulator: Aer backend simulator.
    :return: A dictionary mapping the decimal integer solutions to their measurement counts.
    """

    min_num_of_qubits = (m - 1).bit_length()

    if num_qubits is None:
        num_qubits = min_num_of_qubits
    elif num_qubits < min_num_of_qubits:
        raise ValueError(f"Number of qubits must be greater than or equal to search space (2^m)-1")

    search_space = 2 ** num_qubits

    if shots is None:
        shots = min(8192, max(1024, search_space * 4))

    logic_expr, num_solutions = _generate_congruence_logic_string(a, m, num_qubits)
    oracle = PhaseOracleGate(logic_expr)

    grover = GroverAlgorithmOptimized(oracle) if use_optimized else GroverAlgorithmManual(oracle)

    optimal_iters = grover.calculate_optimal_iterations(num_solutions)

    if simulator is None:
        simulator = Aer.get_backend('qasm_simulator')

    raw_results = grover.run_simulation(simulator, iterations=optimal_iters, shots=shots)

    parsed_results = {}
    for binary_state, count in raw_results.items():
        decimal_val = int(binary_state, 2)
        parsed_results[decimal_val] = count

    return parsed_results

def _generate_congruence_logic_string(a: int, m: int, num_qubits: int) -> tuple:
    """
    Generates a boolean logical expression for x^2 ≡ a (mod m).
    """
    solutions = []
    search_space = 2 ** num_qubits

    for x in range(search_space):
        if (x ** 2) % m == a % m:
            bin_x = format(x, f'0{num_qubits}b')
            clause = []
            for i, bit in enumerate(reversed(bin_x)):
                if bit == '1':
                    clause.append(f'x{i}')
                else:
                    clause.append(f'~x{i}')
            solutions.append('(' + ' & '.join(clause) + ')')

    if not solutions:
        raise NoDiscreteRootsError(f"No solutions exist for x^2 ≡ {a} (mod {m}) in the search space.")

    return ' | '.join(solutions), len(solutions)