import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCPhaseGate

from src.algorithms.grover.grover_base import GroverAlgorithmBase


class GroverAlgorithmFixedPoint(GroverAlgorithmBase):
    def __init__(self, oracle):
        super().__init__(oracle)
        phase = np.pi / 3
        self.diffuser_pi_3 = self._build_parameterized_diffuser(phase)
        self.oracle_pi_3 = self.oracle.power(1 / 3)

    def _build_parameterized_diffuser(self, angle: float) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=f"FixedPoint_Diffuser_({angle:.2f})")
        qc.h(range(self.num_qubits))
        qc.x(range(self.num_qubits))

        if self.num_qubits > 1:
            mcphase = MCPhaseGate(angle, self.num_qubits - 1)
            qc.append(mcphase, range(self.num_qubits))
        else:
            qc.p(angle, 0)

        qc.x(range(self.num_qubits))
        qc.h(range(self.num_qubits))
        return qc

    def build_circuit(self, number_of_solutions: int = None) -> QuantumCircuit:
        iterations = 2*(self._calculate_optimal_iterations(number_of_solutions) + 17)
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)

        qc.h(range(self.num_qubits))

        for _ in range(iterations):
            qc.compose(self.oracle_pi_3, inplace=True)
            qc.compose(self.diffuser_pi_3, inplace=True)

        qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc