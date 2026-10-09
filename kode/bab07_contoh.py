"""Memeriksa setiap bilangan Contoh Soal dan prosa Bab 7."""
import math
from fractions import Fraction as Fr

import numpy as np

from bab01_data import data_mini, rancangan

X, y = data_mini()
Xt = rancangan(X)
yi = y.astype(int)

# Contoh Soal 7.1: proyeksi ke garis dan rata-rata
a, b = np.array([3, 1]), np.array([1, 2])
assert a @ b == 5 and a @ a == 10
P = np.outer(a, a) / 10
assert np.allclose(P @ b, [1.5, 0.5]) and np.allclose(b - P @ b, [-0.5, 1.5])
assert np.isclose(a @ (b - P @ b), 0)
assert (np.outer(a, a) @ np.outer(a, a)).tolist() == [[90, 30], [30, 10]]
assert np.allclose(P @ P, P)
assert yi.sum() == 40 and (yi - 10).tolist() == [-7, 0, 2, 5]

# Contoh Soal 7.2: matriks topi
e = np.array([-1, 1, 1, -1])
H4 = 4 * np.eye(4) - np.outer(e, e)
assert H4.astype(int).tolist() == [[3, 1, 1, -1], [1, 3, -1, 1],
                                   [1, -1, 3, 1], [-1, 1, 1, 3]]
H = Xt @ np.linalg.solve(Xt.T @ Xt, Xt.T)
assert np.allclose(H, H4 / 4) and np.isclose(np.trace(H), 3)
assert (3 * 3 + 10 + 12 - 15) == 16
assert np.allclose(H @ y, [4, 9, 11, 16])

# rotasi dan W / sqrt(2)
W = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
assert np.allclose(W.T @ W, np.eye(2)) and np.isclose(np.linalg.det(W), -1)

# Contoh Soal 7.3: Gram-Schmidt
one, x1, x2 = Xt[:, 0], Xt[:, 1], Xt[:, 2]
q1 = one / 2
assert np.isclose(q1 @ x1, 6) and np.isclose(q1 @ x2, 6)
v2 = x1 - 6 * q1
assert v2.tolist() == [-2, -1, 1, 2] and v2 @ v2 == 10
q2 = v2 / math.sqrt(10)
assert np.isclose(q2 @ x2, 6 / math.sqrt(10))
v3 = x2 - 6 * q1 - 0.6 * v2
assert np.allclose(v3, 0.8 * np.array([-1, 2, -2, 1]))
assert np.allclose(0.6 * v2, [-1.2, -0.6, 0.6, 1.2])
q3 = np.array([-1, 2, -2, 1]) / math.sqrt(10)
assert np.isclose(q2 @ q3, 0) and np.isclose(q1 @ q3, 0)
assert np.isclose(math.sqrt(1 - 0.6 ** 2), 0.8)

# R dan Q^T y
R = np.array([[2, 6, 6], [0, math.sqrt(10), 3 * math.sqrt(10) / 5],
              [0, 0, 4 * math.sqrt(10) / 5]])
Q = np.column_stack([q1, q2, q3])
assert np.allclose(Q @ R, Xt)
assert np.isclose(6 / math.sqrt(10), 3 * math.sqrt(10) / 5)
assert round(3 * math.sqrt(10) / 5, 4) == 1.8974
assert round(4 * math.sqrt(10) / 5, 4) == 2.5298

# Contoh Soal 7.4: kuadrat terkecil lewat QR
assert np.isclose(q1 @ y, 20)
assert -6 - 10 + 12 + 30 == 26 and -3 + 20 - 24 + 15 == 8
assert Fr(8 * 5, 4 * 10) == 1 and (26 - 6) / 10 == 2 and (20 - 12 - 6) / 2 == 1
assert np.allclose(np.linalg.solve(R, Q.T @ y), [1, 2, 1])
assert round(20 * math.sqrt(10), 4) == 63.2456
assert np.allclose(Q @ Q.T, H)
print("Contoh Soal Bab 7: semua bilangan cocok")
