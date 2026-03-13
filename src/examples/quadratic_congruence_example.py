from src.algorithms.quadratic_congruence import solve_quadratic_congruence

if __name__ == "__main__":
    # Problem: Find x such that x^2 ≡ 1 (mod 15) (search space 0-15)
    A_VAL = 1
    MODULO = 15

    print(f"Solving: x^2 ≡ {A_VAL} (mod {MODULO})")

    print("Executing Quantum Search (Manual Diffuser)...")
    results = solve_quadratic_congruence(A_VAL, MODULO, use_optimized=False)
    print("Results:", results)