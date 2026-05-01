from qiskit import QuantumCircuit
from qiskit.circuit.library import grover_operator

from src.algorithms.grover.grover_base import GroverAlgorithmBase


class GroverAlgorithmOptimized(GroverAlgorithmBase):
    def __init__(self, oracle):
        super().__init__(oracle)
        self.grover_op = grover_operator(oracle)

    def build_circuit(self, number_of_solutions: int = None) -> QuantumCircuit:
        iterations = self._calculate_optimal_iterations(number_of_solutions)
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        qc.h(range(self.num_qubits))

        for _ in range(iterations):
            qc.compose(self.grover_op, inplace=True)

        qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc