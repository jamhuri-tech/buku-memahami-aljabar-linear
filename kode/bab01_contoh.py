"""Memeriksa setiap bilangan Contoh Soal dan prosa Bab 1."""
from fractions import Fraction as Fr

import numpy as np

from bab01_data import data_mini, rancangan

X, y = data_mini()
Xt = rancangan(X)
Xi, yi = X.astype(int), y.astype(int)

# prosa: x_32 = 2, kolom x2 = kolom x1 dengan sampel 2 dan 3 bertukar
assert Xi[2, 1] == 2
assert list(Xi[[0, 2, 1, 3], 0]) == list(Xi[:, 1])

# Contoh Soal 1.1: rata-rata dan pemusatan
assert Xi[:, 0].sum() == 12 and Xi[:, 1].sum() == 12 and yi.sum() == 40
Xc = Xi - 3
assert Xc.tolist() == [[-2, -2], [-1, 1], [1, -1], [2, 2]]
assert (Xc.sum(0) == 0).all() and (Xc ** 2).sum(0).tolist() == [10, 10]

# Contoh Soal 1.2: satu sampel, w = (0, 2, 1)
w = np.array([0, 2, 1])
x2 = Xt[1].astype(int)
assert x2.tolist() == [1, 2, 4]
yh = x2 @ w
assert yh == 8 and yi[1] - yh == 2 and (yi[1] - yh) ** 2 == 4
g = -2 * 2 * x2
assert g.tolist() == [-4, -8, -16]
wb = [Fr(int(a)) - Fr(1, 100) * int(b) for a, b in zip(w, g)]
assert wb == [Fr(4, 100), Fr(208, 100), Fr(116, 100)]
assert sum(Fr(int(a)) * b for a, b in zip(x2, wb)) == Fr(884, 100)
assert (x2 @ x2) == 21 and 8 + Fr(2, 100) * 2 * 21 == Fr(884, 100)

# Contoh Soal 1.3: bentuk matriks
for w, yh_, e_, L_, Xte, grad in [
        ([0, 2, 1], [3, 8, 10, 15], [0, 2, 2, 0], 2, [4, 12, 12],
         [-2, -6, -6]),
        ([1, 2, 1], [4, 9, 11, 16], [-1, 1, 1, -1], 1, [0, 0, 0],
         [0, 0, 0])]:
    w = np.array(w)
    assert (Xt.astype(int) @ w).tolist() == yh_
    e = yi - Xt.astype(int) @ w
    assert e.tolist() == e_ and Fr(int(e @ e), 4) == L_
    assert (Xt.astype(int).T @ e).tolist() == Xte
    assert [Fr(-2, 4) * v for v in Xte] == grad

# prosa: X~^T y, langkah pertama gradient descent
assert (Xt.astype(int).T @ yi).tolist() == [40, 146, 142]
assert np.allclose(0.02 * 0.5 * np.array([40, 146, 142]), [0.4, 1.46, 1.42])
# nilai eigen X~^T X~ dan faktor penyusutan
G = Xt.T @ Xt
# polinomial karakteristik (lam - 4)(lam^2 - 92 lam + 64)
assert np.allclose(np.poly(G), np.polymul([1, -4], [1, -92, 64]))
lmin = 46 - 6 * np.sqrt(57)
assert round(lmin, 2) == 0.70 and round(1 - 0.02 * 2 / 4 * lmin, 3) == 0.993
ev, V = np.linalg.eigh(Xt.T @ Xt)
assert abs(V[0, 0]) > 0.98
print("Contoh Soal Bab 1: semua bilangan cocok")
