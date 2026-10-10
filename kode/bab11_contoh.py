"""Memeriksa setiap bilangan Contoh Soal dan prosa Bab 11."""
import math
from fractions import Fraction as Fr

import numpy as np

from bab01_data import data_mini, rancangan

X, y = data_mini()
Xc = X - X.mean(0)
yc = y - y.mean()
r2 = math.sqrt(2)
v1, v2 = np.array([1, 1]) / r2, np.array([1, -1]) / r2

# Contoh Soal 11.1: SVD data terpusat
assert np.allclose(Xc @ v1, np.array([-4, 0, 0, 4]) / r2)
assert np.allclose(Xc @ v2, np.array([0, -2, 2, 0]) / r2)
u1, u2 = Xc @ v1 / 4, Xc @ v2 / 2
assert np.allclose(u1, np.array([-1, 0, 0, 1]) / r2)
assert np.allclose(u2, np.array([0, -1, 1, 0]) / r2)
assert np.isclose(u1 @ u2, 0) and np.isclose(u1 @ u1, 1)
assert np.isclose(np.array([-2, -2]) @ v1, -4 / r2)

# Contoh Soal 11.2: jumlah dua matriks rank 1
T1, T2 = 4 * np.outer(u1, v1), 2 * np.outer(u2, v2)
assert np.allclose(T1, [[-2, -2], [0, 0], [0, 0], [2, 2]])
assert np.allclose(T2, [[0, 0], [-1, 1], [1, -1], [0, 0]])
assert np.allclose(T1 + T2, Xc)
assert np.isclose((T1 ** 2).sum(), 16) and np.isclose((T2 ** 2).sum(), 4)
assert Fr(16, 20) == Fr(4, 5)

# matriks M
M = np.array([[3, 0], [4, 5]])
assert (M.T @ M).tolist() == [[25, 20], [20, 25]]
assert np.allclose(np.linalg.svd(M, compute_uv=False),
                   [3 * math.sqrt(5), math.sqrt(5)])
assert np.allclose(M @ v1, np.array([3, 9]) / r2)
assert np.allclose(M @ v1 / (3 * math.sqrt(5)), np.array([1, 3]) / math.sqrt(10))

# Contoh Soal 11.3: kuadrat terkecil lewat SVD
assert np.isclose(u1 @ yc, 60 / r2) and np.isclose(u2 @ yc, 10 / r2)
w = (60 / r2 / 4) * v1 + (10 / r2 / 2) * v2
assert np.allclose(w, [10, 5])
assert np.allclose((15 / r2) * v1, [7.5, 7.5])
assert np.allclose((5 / r2) * v2, [2.5, -2.5])

# bilangan kondisi
s = np.linalg.svd(rancangan(X), compute_uv=False)
assert np.allclose(s, [math.sqrt(46 + 6 * math.sqrt(57)), 2,
                       math.sqrt(46 - 6 * math.sqrt(57))])
assert round(s[0], 3) == 9.555 and round(s[2], 3) == 0.837
assert round(s[0] / s[2], 2) == 11.41 and round((s[0] / s[2]) ** 2, 1) == 130.2

# Contoh Soal 11.4: ridge lewat SVD
assert Fr(16, 18) == Fr(8, 9) and Fr(4, 6) == Fr(2, 3)
w2 = Fr(8, 9) * Fr(15, 2) * np.array([1, 1]) + Fr(2, 3) * Fr(5, 2) * np.array([1, -1])
assert list(w2) == [Fr(25, 3), 5]
assert Fr(20, 3) - Fr(5, 3) == 5
print("Contoh Soal Bab 11: semua bilangan cocok")
