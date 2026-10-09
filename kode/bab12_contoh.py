"""Memeriksa setiap bilangan Contoh Soal dan prosa Bab 12."""
import math
from fractions import Fraction as Fr

import numpy as np

from bab01_data import data_mini

X, y = data_mini()
Xc = X - X.mean(0)
r2 = math.sqrt(2)

# norma Frobenius
assert (Xc ** 2).sum() == 20 == 4 ** 2 + 2 ** 2

# Contoh Soal 12.1: hampiran rank 1
A1 = np.array([[-2, -2], [0, 0], [0, 0], [2, 2]])
E = Xc - A1
assert E.tolist() == [[0, 0], [-1, 1], [1, -1], [0, 0]]
assert (E ** 2).sum() == 4 and np.isclose(np.linalg.norm(E, 2), 2)
assert Fr(16, 20) == Fr(4, 5)

# Contoh Soal 12.2: PCA
v1, v2 = np.array([1, 1]) / r2, np.array([1, -1]) / r2
Z = Xc @ np.column_stack([v1, v2])
assert np.allclose(Z * r2, [[-4, 0], [0, -2], [0, 2], [4, 0]])
assert np.isclose((Z[:, 0] ** 2).mean(), 4) and np.isclose((Z[:, 1] ** 2).mean(), 1)
assert np.isclose((16 + 16) / 2 / 4, 4) and np.isclose((4 + 4) / 2 / 4, 1)
assert np.isclose(Z[1, 0], 0) and np.allclose(X.mean(0) + Z[1, 0] * v1, [3, 3])

# Contoh Soal 12.3: penyimpanan citra
assert 20 * (600 + 512 + 1) == 22260 == 20 * 1113
assert 600 * 512 == 307200 and round(100 * 22260 / 307200, 1) == 7.2
assert 276 * 1113 == 307188 < 307200 < 277 * 1113
assert round(307200 / 1113, 2) == 276.01
for k, n in ((5, 5565), (50, 55650)):
    assert k * 1113 == n
assert round(100 * 5565 / 307200, 1) == 1.8 and round(100 * 55650 / 307200, 1) == 18.1
assert round(1 - 0.33 ** 2, 2) == 0.89

# Contoh Soal 12.4: faktorisasi matriks
A = np.array([[2, 0], [1, 1], [0, 2], [2, 1], [1, 2], [1, 0]])
B = np.array([[2, 1], [1, 0], [0, 2], [1, 2], [2, 2]])
R = A @ B.T
assert R[0, 3] == 2 == A[0] @ B[3] and R[3, 2] == 2 == A[3] @ B[2]
assert R[0, 2] == 0 and R[4].max() == 6 == R[4, 4]
assert np.linalg.matrix_rank(R) == 2
assert B.T.tolist() == [[2, 1, 0, 1, 2], [1, 0, 2, 2, 2]]
print("Contoh Soal Bab 12: semua bilangan cocok")
