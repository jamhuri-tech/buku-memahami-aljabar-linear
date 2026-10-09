"""Memeriksa setiap bilangan Contoh Soal dan prosa Bab 8."""
import math
from fractions import Fraction as Fr

import numpy as np

from bab01_data import data_mini, rancangan

X, y = data_mini()
Xt = rancangan(X).astype(int)
yi = y.astype(int)
w = np.array([1, 2, 1])
e = yi - Xt @ w

# Contoh Soal 8.1: Pythagoras
d = Xt @ (w - np.array([0, 2, 1]))
assert d.tolist() == [1, 1, 1, 1] and d @ d == 4 and e @ e == 4
r = yi - Xt @ [0, 2, 1]
assert r.tolist() == [0, 2, 2, 0] and r @ r == 8 == 4 + 4
assert e @ d == 0

# Contoh Soal 8.2: jumlah kuadrat
yc = yi - 10
yhc = Xt @ w - 10
assert yc @ yc == 78 and yhc.tolist() == [-6, -1, 1, 6] and yhc @ yhc == 74
assert Fr(74, 78) == Fr(37, 39) and round(37 / 39, 4) == 0.9487
assert yc @ yhc == 74 == 42 + 0 + 2 + 30

# FWL
v3 = np.array([-0.8, 1.6, -1.6, 0.8])
assert np.isclose(v3 @ y, 6.4) and np.isclose(v3 @ v3, 6.4)
assert -3 + 20 - 24 + 15 == 8

# Contoh Soal 8.3: galat baku
s2 = Fr(4, 4 - 3)
assert s2 == 4
assert np.isclose(math.sqrt(4 * 352 / 256), math.sqrt(5.5))
assert round(math.sqrt(5.5), 4) == 2.3452 and round(math.sqrt(0.625), 4) == 0.7906
assert np.isclose(-24 / math.sqrt(40 * 40), -0.6)
Gi = np.linalg.inv(Xt.T @ Xt)
assert np.allclose(256 * Gi, [[352, -48, -48], [-48, 40, -24],
                              [-48, -24, 40]])

# Contoh Soal 8.4: ridge
Xc = (X - X.mean(0)).astype(int)
assert (Xc.T @ Xc).tolist() == [[10, 6], [6, 10]]
assert (Xc.T @ yc).tolist() == [26, 22]
assert 100 - 36 == 64 and 260 - 132 == 128 and -156 + 220 == 64
assert 10 - 2 * 3 - 1 * 3 == 1
assert 144 - 36 == 108 and 12 * 26 - 6 * 22 == 180 and -6 * 26 + 12 * 22 == 108
assert Fr(180, 108) == Fr(5, 3)
assert round(math.sqrt(5), 3) == 2.236 and round(math.sqrt(34) / 3, 3) == 1.944
w1 = np.linalg.solve(Xc.T @ Xc + np.eye(2), Xc.T @ yc)
assert np.isclose(w1[1], 86 / 85)
print("Contoh Soal Bab 8: semua bilangan cocok")
