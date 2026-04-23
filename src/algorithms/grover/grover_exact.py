import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCPhaseGate

from src.algorithms.grover.grover_base import GroverAlgorithmBase


class GroverAlgorithmExact(GroverAlgorithmBase):
    def __init__(self, oracle):
        super().__init__(oracle)
        self.standard_diffuser = self._build_parameterized_diffuser(np.pi)

    def _build_parameterized_diffuser(self, angle: float) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, name=f"Param_Diffuser_({angle:.2f})")
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
        if number_of_solutions is None or number_of_solutions < 1:
            raise ValueError("Number of solutions must be provided and be a positive integer.")

        n_states = 2 ** self.num_qubits
        theta = 2 * np.arcsin(np.sqrt(number_of_solutions / n_states))

        exact_iters_float = (np.pi / (2 * theta)) - 0.5
        iterations = max(0, math.floor(exact_iters_float))

        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        qc.h(range(self.num_qubits))

        for _ in range(iterations):
            qc.compose(self.oracle, inplace=True)
            qc.compose(self.standard_diffuser, inplace=True)

        remaining_angle = np.pi / 2 - (iterations + 0.5) * theta
        if remaining_angle > 1e-5:
            final_phase = 2 * remaining_angle
            final_diffuser = self._build_parameterized_diffuser(final_phase)
            qc.compose(self.oracle, inplace=True)
            qc.compose(final_diffuser, inplace=True)

        qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc