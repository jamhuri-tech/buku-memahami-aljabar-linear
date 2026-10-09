"""Memeriksa setiap bilangan Contoh Soal dan prosa Bab 2."""
import math
from fractions import Fraction as Fr

import numpy as np

from bab01_data import data_mini

X, y = data_mini()
X, y = X.astype(int), y.astype(int)
x1, x2, one = X[:, 0], X[:, 1], np.ones(4, int)

# Contoh Soal 2.1: kombinasi linear
assert (1 * one + 2 * x1 + 1 * x2).tolist() == [4, 9, 11, 16]
assert (x1 - x2).tolist() == [0, -2, 2, 0]

# Gambar 2.1: v = (-2, 2)
v = np.array([-2, 2])
assert np.abs(v).sum() == 4 and v @ v == 8 and np.abs(v).max() == 2

# Contoh Soal 2.2: norma, Cauchy-Schwarz, segitiga
assert np.abs(x1).sum() == 12 and x1 @ x1 == 46 and x1.max() == 5
assert round(math.sqrt(46), 4) == 6.7823
assert x1 @ x2 == 42 and 42 <= 46
s = x1 + x2
assert s.tolist() == [2, 6, 6, 10] and s @ s == 176
assert round(math.sqrt(176), 2) == 13.27
assert round(2 * math.sqrt(46), 2) == 13.56
assert 176 == 46 + 2 * 42 + 46

# Contoh Soal 2.3: kosinus dan korelasi
assert Fr(42, 46) == Fr(21, 23) and round(21 / 23, 3) == 0.913
c1, c2, yc = x1 - 3, x2 - 3, y - 10
assert yc.tolist() == [-7, 0, 2, 5] and yc @ yc == 78
assert c1 @ c2 == 6 and c1 @ c1 == 10 and c2 @ c2 == 10
assert round(math.degrees(math.acos(0.6)), 2) == 53.13
assert c1 @ yc == 26 and c2 @ yc == 22
assert round(26 / math.sqrt(780), 4) == 0.9309
assert round(22 / math.sqrt(780), 4) == 0.7877
assert np.isclose(np.corrcoef(x1, y)[0, 1], 26 / math.sqrt(780))

# Contoh Soal 2.4: jarak
D2 = {(i, k): int(((X[i] - X[k]) ** 2).sum())
      for i in range(4) for k in range(i + 1, 4)}
assert D2 == {(0, 1): 10, (0, 2): 10, (0, 3): 32, (1, 2): 8,
              (1, 3): 10, (2, 3): 10}
D1 = [int(np.abs(X[i] - X[k]).sum())
      for i in range(4) for k in range(i + 1, 4)]
assert D1 == [4, 4, 8, 4, 4, 4]
assert X[0] @ X[0] == 2 and X[3] @ X[3] == 50 and X[0] @ X[3] == 10
assert 2 + 50 - 20 == 32

# prosa: simpangan baku kedua fitur sqrt(10/4), panjang kolom baku sqrt(m)
assert np.isclose(X.std(axis=0), math.sqrt(10 / 4)).all()
Z = (X - X.mean(0)) / X.std(0)
assert np.allclose((Z ** 2).sum(0), 4)
# prosa: contoh dua dimensi u = (3, 1), v = (1, 2), w = (-1, 3)
u, v2, w = np.array([3, 1]), np.array([1, 2]), np.array([-1, 3])
assert (u + v2).tolist() == [4, 3] and (2 * v2).tolist() == [2, 4]
assert (2 * u - v2).tolist() == [5, 0]
assert u @ v2 == 5 and u @ w == 0
assert u @ u == 10 and np.abs(u).sum() == 4 and np.abs(u).max() == 3
assert round(math.sqrt(10), 2) == 3.16 and v2 @ v2 == 5
assert round(math.sqrt(50), 2) == 7.07
assert np.isclose(5 / math.sqrt(50), 1 / math.sqrt(2))
assert np.isclose(math.degrees(math.acos(5 / math.sqrt(50))), 45)
e = v2 / math.sqrt(5)
assert np.allclose(e.round(3), [0.447, 0.894])
d = X[1] - X[2]
assert d.tolist() == [-2, 2] and d @ d == 8 and np.abs(d).sum() == 4
print("Contoh Soal Bab 2: semua bilangan cocok")
