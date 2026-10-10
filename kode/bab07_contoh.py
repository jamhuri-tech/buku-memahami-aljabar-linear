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
assert yi.sum() == 200 and (yi - 50).tolist() == [-35, 0, 10, 25]

# Contoh Soal 7.2: matriks topi
e = np.array([-1, 1, 1, -1])
H4 = 4 * np.eye(4) - np.outer(e, e)
assert H4.astype(int).tolist() == [[3, 1, 1, -1], [1, 3, -1, 1],
                                   [1, -1, 3, 1], [-1, 1, 1, 3]]
H = Xt @ np.linalg.solve(Xt.T @ Xt, Xt.T)
assert np.allclose(H, H4 / 4) and np.isclose(np.trace(H), 3)
assert (3 * 15 + 50 + 60 - 75) == 80
assert np.allclose(H @ y, [20, 45, 55, 80])
assert np.allclose(y - H @ y, 5 * e)

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
assert np.isclose(q1 @ y, 100)
assert -30 - 50 + 60 + 150 == 130 and -15 + 100 - 120 + 75 == 40
assert Fr(40 * 5, 4 * 10) == 5 and (130 - 30) / 10 == 10
assert (100 - 60 - 30) / 2 == 5
assert np.allclose(np.linalg.solve(R, Q.T @ y), [5, 10, 5])
assert round(100 * math.sqrt(10), 4) == 316.2278
assert np.allclose(Q @ Q.T, H)
print("Contoh Soal Bab 7: semua bilangan cocok")
