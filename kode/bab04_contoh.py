"""Memeriksa setiap bilangan Contoh Soal dan prosa Bab 4."""
from fractions import Fraction as Fr

import numpy as np

from bab01_data import data_mini, rancangan

X, y = data_mini()
Xt = rancangan(X).astype(int)
A = [[Fr(int(v)) for v in r] for r in Xt.T @ Xt]
b = [Fr(int(v)) for v in Xt.T @ y.astype(int)]
assert [[int(v) for v in r] for r in A] == [[4, 12, 12], [12, 46, 42],
                                            [12, 42, 46]]
assert b == [200, 730, 710]

# sistem 2 x 2 dan gambar kolom
A2 = np.array([[2, 1], [1, 3]])
assert (A2 @ [1, 2]).tolist() == [4, 7]
assert Fr(5, 2) * 2 == 5 and 2 * 1 + 2 == 4      # eliminasi 2 x 2

# Contoh Soal 4.1: eliminasi Gauss
M = [r + [bi] for r, bi in zip(A, b)]
l21, l31 = M[1][0] / M[0][0], M[2][0] / M[0][0]
assert l21 == l31 == 3
M[1] = [a - l21 * c for a, c in zip(M[1], M[0])]
M[2] = [a - l31 * c for a, c in zip(M[2], M[0])]
assert M[1] == [0, 10, 6, 130] and M[2] == [0, 6, 10, 110]
l32 = M[2][1] / M[1][1]
assert l32 == Fr(3, 5)
assert [l32 * c for c in M[1]] == [0, 6, Fr(18, 5), 78]
M[2] = [a - l32 * c for a, c in zip(M[2], M[1])]
assert M[2] == [0, 0, Fr(32, 5), 32]
w2 = M[2][3] / M[2][2]
w1 = (M[1][3] - M[1][2] * w2) / M[1][1]
w0 = (M[0][3] - M[0][1] * w1 - M[0][2] * w2) / M[0][0]
assert (w0, w1, w2) == (5, 10, 5)

# Persamaan E21 A dan LU
E21 = np.array([[1, 0, 0], [-3, 1, 0], [0, 0, 1]])
assert (E21 @ (Xt.T @ Xt)).tolist() == [[4, 12, 12], [0, 10, 6],
                                        [12, 42, 46]]
L = [[1, 0, 0], [3, 1, 0], [3, Fr(3, 5), 1]]
U = [[4, 12, 12], [0, 10, 6], [0, 0, Fr(32, 5)]]
LU = [[sum(L[i][k] * U[k][j] for k in range(3)) for j in range(3)]
      for i in range(3)]
assert LU == A

# Contoh Soal 4.2: substitusi maju dan mundur
assert [3 * u + Fr(3, 5) * v + z for u, v, z in zip(*U)] == [12, 42, 46]
c1 = b[0]
c2 = b[1] - 3 * c1
c3 = b[2] - 3 * c1 - Fr(3, 5) * c2
assert (c1, c2, c3) == (200, 130, 32) and 110 - 78 == 32
assert Fr(32) / Fr(32, 5) == 5
assert (130 - 30) / 10 == 10 and (200 - 120 - 60) / 4 == 5

# Contoh Soal 4.3: pivot dan banyaknya solusi
P = np.array([[0, 1], [1, 0]])
A3 = np.array([[0, 2], [3, 1]])
assert (P @ A3).tolist() == [[3, 1], [0, 2]]
assert np.allclose(np.linalg.solve(A3, [4, 5]), [1, 2])
for c in (4, 5):
    assert c - 2 * 2 == (0 if c == 4 else 1)
B2 = np.array([[1, 1], [2, 2]])
for t in (-1.5, 0, 2.0):
    assert np.allclose(B2 @ [2 - t, t], [2, 4])
assert (B2 @ [-1, 1]).tolist() == [0, 0]

# prosa: pivot kecil, banyak ruas kanan, dan biaya
assert 1 - 1e20 == -1e20 and 2 - 1e20 == -1e20
w_2y = np.linalg.solve(Xt.T @ Xt, Xt.T @ (2 * y))
w_y1 = np.linalg.solve(Xt.T @ Xt, Xt.T @ (y + 1))
assert np.allclose(w_2y, [10, 20, 10]) and np.allclose(w_y1, [6, 10, 5])
print("Contoh Soal Bab 4: semua bilangan cocok")
