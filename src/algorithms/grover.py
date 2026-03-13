import math
from abc import ABC, abstractmethod
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import grover_operator


class GroverAlgorithmBase(ABC):
    def __init__(self, oracle):
        """
        Initializes a generic Grover's algorithm base for a given oracle.

        :param oracle: QuantumCircuit or Gate representing the target state (flips its phase).
        """
        self.oracle = oracle
        self.num_qubits = oracle.num_qubits

    def calculate_optimal_iterations(self, num_solutions: int = 1) -> int:
        """
        Calculates the optimal number of Grover iterations for a given number of solutions.
        """
        N = 2 ** self.num_qubits
        return math.floor(math.pi / 4 * math.sqrt(N / num_solutions))

    @abstractmethod
    def build_circuit(self, iterations: int = None) -> QuantumCircuit:
        """
        Abstract method to build the complete circuit for Grover's algorithm.
        Must be implemented by derived classes.
        """
        pass

    def run_simulation(self, simulator, iterations: int = None, shots: int = 1024) -> dict:
        """
        Runs the circuit on a provided simulator and returns the probability distribution.
        """
        qc = self.build_circuit(iterations)

        # Transpile the circuit to instructions understood by the simulator
        compiled_circuit = transpile(qc, simulator)

        # Run the job and fetch the results
        job = simulator.run(compiled_circuit, shots=shots)
        result = job.result()

        return result.get_counts(compiled_circuit)


class GroverAlgorithmOptimized(GroverAlgorithmBase):
    def __init__(self, oracle):
        """
        Initializes the optimized Grover's algorithm using Qiskit's built-in grover_operator.
        """
        super().__init__(oracle)

        # grover_operator automatically creates a diffuser matched to the given oracle
        self.grover_op = grover_operator(oracle)

    def build_circuit(self, iterations: int = None) -> QuantumCircuit:
        """
        Builds the complete circuit using the pre-built Grover operator.
        """
        if iterations is None:
            iterations = self.calculate_optimal_iterations()

        qc = QuantumCircuit(self.num_qubits, self.num_qubits)

        # Step 1: Put all qubits into a uniform superposition (Hadamard gates)
        qc.h(range(self.num_qubits))

        # Step 2: Apply the Grover operator (Oracle + Diffuser) the specified number of times
        for _ in range(iterations):
            qc.compose(self.grover_op, inplace=True)

        # Step 3: Measure all qubits
        qc.measure(range(self.num_qubits), range(self.num_qubits))

        return qc


class GroverAlgorithmManual(GroverAlgorithmBase):
    def __init__(self, oracle):
        """
        Initializes the manual Grover's algorithm with a custom-built diffuser from scratch.
        """
        super().__init__(oracle)
        self.diffuser = self._build_diffuser()

    def _build_diffuser(self) -> QuantumCircuit:
        """
        Builds the Grover diffusion operator entirely from basic logic gates:
        H, X, and Multi-Controlled X (MCX).
        """
        qc = QuantumCircuit(self.num_qubits, name="Diffuser")

        # Step 1: Apply H gates to all qubits
        qc.h(range(self.num_qubits))

        # Step 2: Apply X gates to all qubits
        qc.x(range(self.num_qubits))

        # Step 3: Apply a Multi-Controlled Z gate (simulated with MCX and H gates)
        target_qubit = self.num_qubits - 1
        control_qubits = list(range(self.num_qubits - 1))

        qc.h(target_qubit)
        if control_qubits:
            qc.mcx(control_qubits, target_qubit)
        qc.h(target_qubit)

        # Step 4: Apply X gates to all qubits
        qc.x(range(self.num_qubits))

        # Step 5: Apply H gates to all qubits
        qc.h(range(self.num_qubits))

        return qc

    def build_circuit(self, iterations: int = None) -> QuantumCircuit:
        """
        Builds the complete circuit by manually composing the oracle and the custom diffuser.
        """
        if iterations is None:
            iterations = self.calculate_optimal_iterations()

        qc = QuantumCircuit(self.num_qubits, self.num_qubits)

        # Step 1: Put all qubits into a uniform superposition
        qc.h(range(self.num_qubits))

        # Step 2: Apply Oracle and Diffuser for N iterations
        for _ in range(iterations):
            qc.compose(self.oracle, inplace=True)
            qc.compose(self.diffuser, inplace=True)

        # Step 3: Measure all qubits
        qc.measure(range(self.num_qubits), range(self.num_qubits))

        return qc
