"""Memeriksa setiap bilangan Contoh Soal dan prosa Bab 3."""
import math
from fractions import Fraction as Fr

import numpy as np

from bab01_data import data_mini, rancangan

X, y = data_mini()
X = X.astype(int)
Xt = rancangan(X).astype(int)

# Contoh Soal 3.1: Gram fitur
G = Xt.T @ Xt
assert G.tolist() == [[4, 12, 12], [12, 46, 42], [12, 42, 46]]
assert (G == G.T).all()

# Contoh Soal 3.2: WD dan DW
W = np.array([[1, 1], [1, -1]])
D = np.diag([2, 1])
assert (W @ D).tolist() == [[2, 1], [2, -1]]
assert (D @ W).tolist() == [[2, 2], [1, -1]]
assert ((W @ D).T == D.T @ W.T).all() and ((W @ D).T == D @ W).all()

# Contoh Soal 3.3: kovarians sebagai jumlah hasil kali luar
Xc = X - 3
luar = [np.outer(r, r) for r in Xc]
assert luar[0].tolist() == [[4, 4], [4, 4]]
assert luar[1].tolist() == [[1, -1], [-1, 1]]
assert luar[2].tolist() == [[1, -1], [-1, 1]]
assert luar[3].tolist() == [[4, 4], [4, 4]]
assert sum(luar).tolist() == [[10, 6], [6, 10]] == (Xc.T @ Xc).tolist()
S = [[Fr(int(v), 4) for v in r] for r in Xc.T @ Xc]
assert S == [[Fr(5, 2), Fr(3, 2)], [Fr(3, 2), Fr(5, 2)]]
assert S[0][1] / S[0][0] == Fr(3, 5)

# Contoh Soal 3.4: ganti fitur
F = X @ W
assert F.tolist() == [[2, 0], [6, -2], [6, 2], [10, 0]]
v = np.linalg.solve(W, [2, 1])
assert np.allclose(v, [1.5, 0.5])
assert 1 + Fr(3, 2) * 6 + Fr(1, 2) * 2 == 11 == 1 + 2 * 4 + 2
assert np.allclose(1 + X @ [2, 1], 1 + F @ v)

# prosa: X X^T, sudut cermin 22,5 derajat, biaya
K = X @ X.T
assert K.tolist() == [[2, 6, 6, 10], [6, 20, 16, 30],
                      [6, 16, 20, 30], [10, 30, 30, 50]]
Q = W / math.sqrt(2)
assert np.allclose(Q.T @ Q, np.eye(2)) and np.isclose(np.linalg.det(Q), -1)
ev, V = np.linalg.eigh(Q)
u = V[:, np.argmax(ev)]
assert np.isclose(ev.max(), 1)
assert np.isclose(abs(math.degrees(math.atan(u[1] / u[0]))), 22.5)
N = 2000
assert (N ** 3 + N ** 2) // (2 * N ** 2) == 1000
print("Contoh Soal Bab 3: semua bilangan cocok")
