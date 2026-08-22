import random
import math

def lehmann_test(p, iterations=20):
    if p < 4:
        return p == 2 or p == 3
    if p % 2 == 0:
        return False

    for _ in range(iterations):
        a = random.randint(2, p - 2)
        if math.gcd(a, p) != 1:
            return False
        result = pow(a, (p - 1) // 2, p)
        if result != 1 and result != p - 1:
            return False       # definitely composite
    return True                # probably prime

# Example
if __name__ == "__main__":
    P = int(input("Enter number P: ") or 104729)
    print(f"{P} is {'PRIME' if lehmann_test(P) else 'COMPOSITE'} (Lehmann test)")