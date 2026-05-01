import math
from abc import ABC, abstractmethod
from qiskit import QuantumCircuit, transpile


class GroverAlgorithmBase(ABC):
    def __init__(self, oracle):
        self.oracle = oracle
        self.num_qubits = oracle.num_qubits

    def _calculate_optimal_iterations(self, num_solutions: int = None) -> int:
        if num_solutions is None:
            num_solutions = 1

        n_states = 2 ** self.num_qubits
        return math.floor(math.pi / 4 * math.sqrt(n_states / num_solutions))

    @abstractmethod
    def build_circuit(self, number_of_solutions: int = None) -> QuantumCircuit:
        """
        Abstract method to build the complete circuit for Grover's algorithm.
        Must be implemented by derived classes.
        """
        pass

    def run_simulation(self, simulator, shots: int = 1024, number_of_solutions: int = None) -> dict:
        qc = self.build_circuit(number_of_solutions)

        compiled_circuit = transpile(qc, simulator)
        job = simulator.run(compiled_circuit, shots=shots)
        result = job.result()

        return result.get_counts(compiled_circuit)