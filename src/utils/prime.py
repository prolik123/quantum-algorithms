
def is_prime_deterministic_to_int_64(n):
    """
    A deterministic primality test for all numbers n < 2^64.
    Uses the deterministic Miller-Rabin approach with a fixed set of bases.
    """
    if n in (2, 3, 5, 7):
        return True
    if n == 1 or n % 2 == 0:
        return False

    # n - 1 = d * 2^r
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Guarantee base for deterministic accuracy for 64-bit integers
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]

    for a in bases:
        if n <= a:
            break

        x = pow(a, d, n)

        if x == 1 or x == n - 1:
            continue

        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False

    return True