import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def Koch(p1, p2, n):
    if n == 0:
        return [p1, p2]

    A = p1 + (p2 - p1)/3
    B = p1 + 2*(p2 - p1)/3

    V = B - A
    angle = np.radians(60)

    x = V[0] * np.cos(angle) - V[1] * np.sin(angle)
    y = V[0] * np.sin(angle) + V[1] * np.cos(angle)

    V_rotated = np.array([x, y])

    C = A + V_rotated

    part1 = Koch(p1, A, n - 1)
    part2 = Koch(A, C, n - 1)
    part3 = Koch(C, B, n - 1)
    part4 = Koch(B, p2, n - 1)

    return part1[:-1] + part2[:-1] + part3[:-1] + part4