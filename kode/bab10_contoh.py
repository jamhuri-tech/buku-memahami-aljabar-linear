"""Memeriksa setiap bilangan Contoh Soal dan prosa Bab 10."""
import math
from fractions import Fraction as Fr

import numpy as np

from bab01_data import data_mini, rancangan

X, y = data_mini()
Xt = rancangan(X).astype(int)
S = np.array([[10, 6], [6, 10]])
G = Xt.T @ Xt

# Contoh Soal 10.1: dekomposisi spektral
P1 = np.array([[1, 1], [1, 1]]) / 2
P2 = np.array([[1, -1], [-1, 1]]) / 2
assert np.allclose(16 * P1, [[8, 8], [8, 8]]) and np.allclose(4 * P2,
                                                              [[2, -2], [-2, 2]])
assert np.allclose(16 * P1 + 4 * P2, S) and np.allclose(P1 + P2, np.eye(2))

# bentuk kuadrat pada sumbu eigen
for x in ([1, 0], [0.3, -2.0], [1.5, 0.7]):
    x1, x2 = x
    f = 10 * x1 ** 2 + 12 * x1 * x2 + 10 * x2 ** 2
    z1, z2 = (x1 + x2) / math.sqrt(2), (x1 - x2) / math.sqrt(2)
    assert np.isclose(f, 16 * z1 ** 2 + 4 * z2 ** 2)
    assert np.isclose(f, np.array(x) @ S @ x)

# Contoh Soal 10.2: uji definit positif
assert 10 - Fr(6, 10) * 6 == Fr(32, 5) and np.linalg.det(S).round() == 64
B = np.array([[1, 2], [2, 1]])
assert np.allclose(np.linalg.eigvalsh(B), [-1, 3]) and 1 - 4 == -3
x = np.array([1, -1])
assert x @ B @ x == -2 == 1 - 4 + 1

# Contoh Soal 10.3: Cholesky
l22 = math.sqrt(46 - 36)
l32 = (42 - 36) / l22
l33 = math.sqrt(46 - 36 - l32 ** 2)
assert np.isclose(l32, 6 / math.sqrt(10)) and np.isclose(l32 ** 2, 3.6)
assert np.isclose(l33, math.sqrt(6.4)) and np.isclose(l33, 4 * math.sqrt(10) / 5)
L = np.array([[2, 0, 0], [6, l22, 0], [6, l32, l33]])
assert np.allclose(L @ L.T, G) and np.allclose(L, np.linalg.cholesky(G))
assert np.allclose(np.diag(L) ** 2, [4, 10, 6.4])

# Contoh Soal 10.4: kovarians dan Mahalanobis
C = np.array([[2.5, 1.5], [1.5, 2.5]])
assert np.isclose(np.linalg.det(C), 4) and Fr(25, 4) - Fr(9, 4) == 4
assert np.allclose(np.linalg.inv(C), np.array([[2.5, -1.5], [-1.5, 2.5]]) / 4)
assert np.allclose(np.linalg.eigvalsh(C), [1, 4])
Ci = np.linalg.inv(C)
for v, d2e in (([-2, -2], 8), ([-1, 1], 2)):
    v = np.array(v)
    assert v @ v == d2e and np.isclose(v @ Ci @ v, 2)
assert np.isclose((10 - 12 + 10) / 4, 2) and np.isclose((2.5 + 3 + 2.5) / 4, 2)
assert np.isclose((-4 / math.sqrt(2)) ** 2 / 4, 2) and 2 / 1 == 2
print("Contoh Soal Bab 10: semua bilangan cocok")
