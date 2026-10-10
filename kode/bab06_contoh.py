"""Memeriksa setiap bilangan Contoh Soal dan prosa Bab 6."""
from fractions import Fraction as Fr

import numpy as np

from bab01_data import data_mini, rancangan

X, y = data_mini()
Xt = rancangan(X).astype(int)
G = Xt.T @ Xt
Gi = [[Fr(v, 256) for v in r] for r in [[352, -48, -48], [-48, 40, -24],
                                        [-48, -24, 40]]]

# Contoh Soal 6.1: invers 2 x 2
A = np.array([[2, 1], [1, 3]])
assert 2 * 3 - 1 * 1 == 5
assert (A @ np.array([[3, -1], [-1, 2]])).tolist() == [[5, 0], [0, 5]]
assert [3 * 4 - 7, -4 + 2 * 7] == [5, 10]
assert np.allclose(np.linalg.inv(A), [[0.6, -0.2], [-0.2, 0.4]])

# Contoh Soal 6.2: Gauss-Jordan
M = [[Fr(2), Fr(1), Fr(1), Fr(0)], [Fr(1), Fr(3), Fr(0), Fr(1)]]
M[1] = [a - Fr(1, 2) * b for a, b in zip(M[1], M[0])]
assert M[1] == [0, Fr(5, 2), Fr(-1, 2), 1]
M[1] = [Fr(2, 5) * a for a in M[1]]
assert M[1] == [0, 1, Fr(-1, 5), Fr(2, 5)]
M[0] = [a - b for a, b in zip(M[0], M[1])]
assert M[0] == [2, 0, Fr(6, 5), Fr(-2, 5)]
M[0] = [a / 2 for a in M[0]]
assert M[0] == [1, 0, Fr(3, 5), Fr(-1, 5)]

# invers persamaan normal dan w
GiG = [[sum(Gi[i][k] * int(G[k][j]) for k in range(3)) for j in range(3)]
       for i in range(3)]
assert GiG == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
assert 352 * 200 - 48 * 730 - 48 * 710 == 1280 == 5 * 256
assert 70400 - 35040 - 34080 == 1280

# determinan: W, S, A, dan G
assert int(round(np.linalg.det([[1, 1], [1, -1]]))) == -2
assert 1 * 4 - 2 * 2 == 0 and 6 - 1 == 5

# Contoh Soal 6.3: kofaktor
assert 46 * 46 - 42 * 42 == 352 and 2116 - 1764 == 352
assert 12 * 46 - 42 * 12 == 48 and 12 * 42 - 46 * 12 == -48
assert 4 * 352 - 12 * 48 + 12 * (-48) == 256 == 1408 - 576 - 576
assert 4 * 10 * Fr(32, 5) == 256
assert 4 * 46 - 12 * 12 == 40
Xt3 = np.column_stack([Xt, X.sum(1).astype(int)])
assert (Xt3.T @ Xt3 @ [0, 1, 1, -1]).tolist() == [0, 0, 0, 0]

# Contoh Soal 6.4: Sherman-Morrison
x = [1, 3, 3]
u = [sum(Gi[i][k] * x[k] for k in range(3)) for i in range(3)]
assert u == [Fr(1, 4), 0, 0]
assert 352 - 48 * 3 - 48 * 3 == 64
s = sum(a * b for a, b in zip(x, u))
assert s == Fr(1, 4)
assert Fr(1, 16) / (1 + s) == Fr(1, 20)
Gi2 = [[Gi[i][j] - u[i] * u[j] / (1 + s) for j in range(3)]
       for i in range(3)]
G2 = G + np.outer(x, x)
assert np.allclose(np.array(Gi2, float), np.linalg.inv(G2))
u2 = [sum(Gi2[i][k] * x[k] for k in range(3)) for i in range(3)]
assert u2 == [Fr(1, 5), 0, 0]
assert 5 + 10 * 3 + 5 * 3 == 50
w2 = [a + b * (55 - 50) for a, b in zip([5, 10, 5], u2)]
assert w2 == [6, 10, 5]

# prosa: determinan dan kondisi
assert np.isclose(0.1 ** 10, 1e-10)
assert np.isclose(np.linalg.det([[1, 1], [1, 1.0001]]), 1e-4)
# Latihan 6.5: leverage 3/4
H = Xt @ np.linalg.inv(G) @ Xt.T
assert np.allclose(np.diag(H), 0.75)
print("Contoh Soal Bab 6: semua bilangan cocok")
