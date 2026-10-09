"""Memeriksa setiap bilangan Contoh Soal dan prosa Bab 9."""
import math
from fractions import Fraction as Fr

import numpy as np

from bab01_data import data_mini, rancangan

X, y = data_mini()
Xt = rancangan(X).astype(int)
S = np.array([[10, 6], [6, 10]])
G = Xt.T @ Xt

# prosa dan Contoh Soal 9.1
assert (S @ [1, 0]).tolist() == [10, 6]
assert (S @ [1, 1]).tolist() == [16, 16] and (S @ [1, -1]).tolist() == [4, -4]
for lam in (16, 4):
    assert (10 - lam) ** 2 - 36 == 0
assert ((S - 16 * np.eye(2)) @ [1, 1] == 0).all()
assert ((S - 4 * np.eye(2)) @ [1, -1] == 0).all()
assert 16 + 4 == np.trace(S) and 16 * 4 == 64 == 100 - 36

# Contoh Soal 9.2: diagonalisasi dan S^2
V = np.array([[1, 1], [1, -1]])
Vi = np.array([[1, 1], [1, -1]]) / 2
assert np.allclose(V @ Vi, np.eye(2))
VL2 = V @ np.diag([256, 16])
assert VL2.tolist() == [[256, 16], [256, -16]]
assert np.allclose(VL2 @ Vi, [[136, 120], [120, 136]])
assert (S @ S).tolist() == [[136, 120], [120, 136]]
x2 = np.array([136, 120]) / np.hypot(136, 120)
assert np.allclose(x2.round(4), [0.7498, 0.6616])

# Contoh Soal 9.3: rantai Markov
P = np.array([[0.9, 0.5], [0.1, 0.5]])
assert np.isclose(np.trace(P), 1.4) and np.isclose(np.linalg.det(P), 0.4)
assert np.allclose(np.sort(np.linalg.eigvals(P)), [0.4, 1.0])
pi = np.array([5, 1]) / 6
assert np.allclose(P @ pi, pi)
assert round(0.4 ** 10, 4) == 0.0001

# Contoh Soal 9.4: nilai eigen G dan laju gradient descent
assert (G @ [0, 1, -1]).tolist() == [0, 4, -4]
assert np.trace(G) == 96 and round(np.linalg.det(G)) == 256
assert 96 - 4 == 92 and Fr(256, 4) == 64
assert 46 ** 2 - 64 == 2052 == 36 * 57
lo, hi = 46 - 6 * math.sqrt(57), 46 + 6 * math.sqrt(57)
assert round(lo, 3) == 0.701 and round(hi, 3) == 91.299
f = [1 - 0.01 * l for l in (lo, 4, hi)]
assert [round(v, 3) for v in f] == [0.993, 0.96, 0.087]
assert round(4 / hi, 4) == 0.0438
# pemusatan: Gram rancangan terpusat diag(4, S)
Xtc = np.column_stack([np.ones(4), X - X.mean(0)])
assert np.allclose(np.linalg.eigvalsh(Xtc.T @ Xtc), [4, 4, 16])
# rotasi 90 derajat
R = np.array([[0, -1], [1, 0]])
assert np.allclose(np.sort_complex(np.linalg.eigvals(R)), [-1j, 1j])
print("Contoh Soal Bab 9: semua bilangan cocok")
