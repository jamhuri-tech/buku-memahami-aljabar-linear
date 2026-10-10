"""Memeriksa setiap bilangan Contoh Soal dan prosa Bab 8."""
import math
from fractions import Fraction as Fr

import numpy as np

from bab01_data import data_mini, rancangan

X, y = data_mini()
Xt = rancangan(X).astype(int)
yi = y.astype(int)
w = np.array([5, 10, 5])
e = yi - Xt @ w

# Contoh Soal 8.1: Pythagoras
d = Xt @ (w - np.array([0, 10, 5]))
assert d.tolist() == [5, 5, 5, 5] and d @ d == 100 and e @ e == 100
r = yi - Xt @ [0, 10, 5]
assert r.tolist() == [0, 10, 10, 0] and r @ r == 200 == 100 + 100
assert e @ d == 0 and Fr(200, 4) == 50

# Contoh Soal 8.2: jumlah kuadrat
yc = yi - 50
yhc = Xt @ w - 50
assert yc @ yc == 1950 and yhc.tolist() == [-30, -5, 5, 30]
assert yhc @ yhc == 1850 and 1850 + 100 == 1950
assert Fr(1850, 1950) == Fr(37, 39) and round(37 / 39, 4) == 0.9487
assert yc @ yhc == 1850 == 1050 + 0 + 50 + 750

# FWL
v3 = np.array([-0.8, 1.6, -1.6, 0.8])
assert np.isclose(v3 @ y, 32) and np.isclose(v3 @ v3, 6.4)
assert -15 + 100 - 120 + 75 == 40 and np.isclose(32 / 6.4, 5)

# Contoh Soal 8.3: galat baku
s2 = Fr(100, 4 - 3)
assert s2 == 100
assert np.isclose(100 * 352 / 256, 137.5) and np.isclose(100 * 40 / 256, 15.625)
assert round(math.sqrt(137.5), 4) == 11.726 and round(math.sqrt(15.625), 4) == 3.9528
assert np.isclose(-24 / math.sqrt(40 * 40), -0.6)
Gi = np.linalg.inv(Xt.T @ Xt)
assert np.allclose(256 * Gi, [[352, -48, -48], [-48, 40, -24],
                              [-48, -24, 40]])

# Contoh Soal 8.4: ridge
Xc = (X - X.mean(0)).astype(int)
assert (Xc.T @ Xc).tolist() == [[10, 6], [6, 10]]
assert (Xc.T @ yc).tolist() == [130, 110]
assert 100 - 36 == 64 and 1300 - 660 == 640 and -780 + 1100 == 320
assert Fr(640, 64) == 10 and Fr(320, 64) == 5
assert 50 - 10 * 3 - 5 * 3 == 5
assert 144 - 36 == 108 and 12 * 130 - 6 * 110 == 900 and -6 * 130 + 12 * 110 == 540
assert Fr(900, 108) == Fr(25, 3) and Fr(540, 108) == 5
assert round(5 * math.sqrt(5), 3) == 11.180 and round(5 * math.sqrt(34) / 3, 3) == 9.718
w1 = np.linalg.solve(Xc.T @ Xc + np.eye(2), Xc.T @ yc)
assert np.isclose(w1[1], 86 / 17) and round(86 / 17, 3) == 5.059
print("Contoh Soal Bab 8: semua bilangan cocok")
