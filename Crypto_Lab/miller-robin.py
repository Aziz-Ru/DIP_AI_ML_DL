import random


def miller_rabin(n, k=5):

    # Step 1: Handle small numbers
    if n < 2:
        return False

    if n == 2 or n == 3:
        return True

    if n % 2 == 0:
        return False

    # Step 2:
    # n - 1 = d * 2^s
    d = n - 1
    s = 0

    while d % 2 == 0:
        d //= 2
        s += 1

    # Step 3: Repeat k times
    for _ in range(k):

        # Random number: 2 <= a <= n-2
        a = random.randrange(2, n - 1)

        # x = a^d mod n
        x = pow(a, d, n)

        # If x = 1 or n-1, this round passes
        if x == 1 or x == n - 1:
            continue

        # Square x repeatedly
        for _ in range(s - 1):

            x = pow(x, 2, n)

            if x == n - 1:
                break

        else:
            return False

    return True