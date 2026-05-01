from qiskit import QuantumCircuit

from src.algorithms.grover.grover_base import GroverAlgorithmBase


class GroverAlgorithmManual(GroverAlgorithmBase):
    def __init__(self, oracle):
        super().__init__(oracle)
        self.diffuser = self._build_diffuser()

    def _build_diffuser(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name="Diffuser")

        qc.h(range(self.num_qubits))
        qc.x(range(self.num_qubits))

        target_qubit = self.num_qubits - 1
        control_qubits = list(range(self.num_qubits - 1))

        qc.h(target_qubit)
        if control_qubits:
            qc.mcx(control_qubits, target_qubit)
        qc.h(target_qubit)

        qc.x(range(self.num_qubits))
        qc.h(range(self.num_qubits))
        return qc

    def build_circuit(self, number_of_solutions: int = None) -> QuantumCircuit:
        iterations = self._calculate_optimal_iterations(number_of_solutions)
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        qc.h(range(self.num_qubits))

        for _ in range(iterations):
            qc.compose(self.oracle, inplace=True)
            qc.compose(self.diffuser, inplace=True)

        qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc