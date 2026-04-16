import numpy as np
import math
import random
from fractions import Fraction
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import UnitaryGate

from src.utils.prime import is_prime_deterministic_to_int_64

# for now it is only a not fully refactored version

class ShorsAlgorithm:
    def __init__(self, N, qubit_num=None):
        """
        Initializes the generalized Shor's Algorithm.
        Dynamically calculates the perfect number of qubits required.
        """
        self.simulator = AerSimulator()

        # Cannot be smaller without arithmetic overflow
        self.n_target = math.ceil(math.log2(N))

        # Rule of thumb: Target register needs L qubits. Counting needs 2L for a good results but can be less.
        self.n_count = 2 * self.n_target if qubit_num is None else qubit_num

    def _build_modular_unitary(self, a, power, N):
        """
        Classically generates the unitary matrix for |y> -> |y * a^power mod N>.
        """
        size = 2 ** self.n_target
        U = np.zeros((size, size))

        a_power_mod_N = pow(a, power, N)

        for y in range(size):
            if y < N:
                # Calculate the modular multiplication safely
                target = (y * a_power_mod_N) % N
                U[target, y] = 1
            else:
                # Identity mapping for out-of-bounds states to maintain Unitarity
                U[y, y] = 1

        return UnitaryGate(U, label=f"({a}^{power})mod{N}")

    def _append_iqft(self, qc):
        """Applies the Inverse Quantum Fourier Transform."""
        for i in range(self.n_count // 2):
            qc.swap(i, self.n_count - i - 1)

        for j in range(self.n_count):
            for m in range(j):
                angle = -np.pi / (2 ** (j - m))
                qc.cp(angle, m, j)
            qc.h(j)

    def find_period(self, a, N):
        """
        Constructs and runs the quantum circuit to find the period 'r'.
        """
        # Initialize quantum circuit
        qc = QuantumCircuit(self.n_count + self.n_target, self.n_count)

        # 1. Initialize counting register in superposition
        qc.h(range(self.n_count))

        # 2. Initialize target register to state |1>
        qc.x(self.n_count)

        # 3. Apply the controlled modular exponentiations
        for i in range(self.n_count):
            power = 2 ** i
            unitary = self._build_modular_unitary(a, power, N)
            c_unitary = unitary.control(1)

            target_qubits = list(range(self.n_count, self.n_count + self.n_target))
            qc.append(c_unitary, [i] + target_qubits)

        # 4. Apply Inverse QFT
        self._append_iqft(qc)

        # 5. Measure counting register
        qc.measure(range(self.n_count), range(self.n_count))

        # 6. Transpile with optimization to reduce gate depth
        compiled_qc = transpile(qc, self.simulator, optimization_level=1)

        # 7. Execute and get results
        job = self.simulator.run(compiled_qc, shots=1024)
        counts = job.result().get_counts()

        # 8. Classical continued fractions to guess the period
        guesses = []
        for output in counts.keys():
            decimal = int(output, 2)
            if decimal == 0:
                continue

            phase = decimal / (2 ** self.n_count)
            frac = Fraction(phase).limit_denominator(N)
            guesses.append(frac.denominator)

        if not guesses:
            return None

        return max(set(guesses), key=guesses.count)


def get_factors(N, num_of_qubits=None, max_attempts=3, is_debug=True, turn_off_optimization=False):
    """
    Main execution wrapper representing the complete Shor's Algorithm pipeline.
    Set is_debug=False to silence all print statements.
    """
    def debug_print(*args, **kwargs):
        if is_debug:
            print(*args, **kwargs)

    debug_print(f"--- Attempting to factorize N = {N} ---")

    if N % 2 == 0:
        debug_print("Classical catch: Number is even.")
        return (2, N // 2)

    if is_prime_deterministic_to_int_64(N):
        debug_print("Classical catch: Number is already prime.")
        return (1, N)

    shor = ShorsAlgorithm(N, num_of_qubits)
    debug_print(f"Quantum setup: Using {shor.n_count} counting qubits and {shor.n_target} target qubits.")

    attempts = 0

    while attempts < max_attempts:
        attempts += 1

        a = random.randint(2, N - 1)
        debug_print(f"\n[Attempt {attempts}] Picked random guess a = {a}")

        gcd = math.gcd(a, N)
        if gcd > 1:
            if turn_off_optimization:
                debug_print(f"Had lucky guess! Retrying and increasing number of max attempts.")
                max_attempts += 1
                continue

            debug_print(f"Lucky classical guess!")
            return (gcd, N // gcd)

        debug_print("Running Quantum Phase Estimation (Synthesizing unitaries...)")
        r = shor.find_period(a, N)

        if r is None:
            debug_print("Quantum failure: No valid period found.")
            continue

        debug_print(f"Quantum computer measured period r = {r}")

        if r % 2 != 0:
            debug_print("Failure: Period is odd. Retrying...")
            continue

        guess1 = pow(a, r // 2, N) - 1
        guess2 = pow(a, r // 2, N) + 1

        factor1 = math.gcd(guess1, N)
        factor2 = math.gcd(guess2, N)

        if factor1 not in [1, N] and factor2 not in [1, N]:
            debug_print("Success! Non-trivial factors found.")
            return (factor1, factor2)

        debug_print("Failure: Found trivial factors. Retrying...")

    debug_print("Max attempts reached.")
    return None


if __name__ == "__main__":
    # 5 * 7 = 35
    factors_debug = get_factors(35, 9, is_debug=True, turn_off_optimization=True)
    if factors_debug:
        print(f"DEBUG RESULT: {factors_debug[0]} x {factors_debug[1]}\n")
