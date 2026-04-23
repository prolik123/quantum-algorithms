from qiskit_aer import Aer

from qiskit.circuit.library import PhaseOracleGate

from src.algorithms.grover.grover_manual import GroverAlgorithmManual
from src.algorithms.grover.grover_optimized import GroverAlgorithmOptimized

if __name__ == "__main__":

    # Define the oracle to search for '101'
    logical_expression = '(x0 & ~x1 & x2)'
    oracle = PhaseOracleGate(logical_expression)

    # Initialize the local simulator (passed via Dependency Injection now)
    simulator = Aer.get_backend('qasm_simulator')

    print("--- Testing Optimized Version ---")
    optimized_grover = GroverAlgorithmOptimized(oracle)
    results_opt = optimized_grover.run_simulation(simulator)
    print(f"Top result: {max(results_opt, key=results_opt.get)} (Count: {max(results_opt.values())})\n")

    print("--- Testing Manual Version ---")
    manual_grover = GroverAlgorithmManual(oracle)
    results_man = manual_grover.run_simulation(simulator)
    print(f"Top result: {max(results_man, key=results_man.get)} (Count: {max(results_man.values())})")