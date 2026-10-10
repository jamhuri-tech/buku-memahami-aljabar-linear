"""Memeriksa setiap bilangan Contoh Soal dan prosa Bab 5."""
from fractions import Fraction as Fr

import numpy as np

from bab01_data import data_mini, rancangan

X, y = data_mini()
Xt = rancangan(X).astype(int)
Xt3 = np.column_stack([Xt, X.sum(1).astype(int)])
u, v = np.array([3, 1]), np.array([1, 2])

# Contoh Soal 5.1: rentang dan subruang
assert (u + v).tolist() == [4, 3] and (2 * u - v).tolist() == [5, 0]
assert Fr(3) - Fr(4, 3) == Fr(5, 3) and Fr(0) - Fr(5, 3) == Fr(-5, 3)
assert 2 + 2 != 2

# Contoh Soal 5.2: eliminasi X~ dan kebergantungan
M = Xt.astype(float).copy()
M[1:] -= M[0]
assert M.tolist() == [[1, 1, 1], [0, 1, 3], [0, 3, 1], [0, 4, 4]]
M[2] -= 3 * M[1]
M[3] -= 4 * M[1]
assert M[2].tolist() == [0, 0, -8] and M[3].tolist() == [0, 0, -8]
M[3] -= M[2]
assert M[3].tolist() == [0, 0, 0]
assert Xt3[:, 3].tolist() == [2, 6, 6, 10]
n = np.array([0, 1, 1, -1])
assert (Xt3 @ n).tolist() == [0, 0, 0, 0]

# rank, ruang nol, rank-nulitas
assert np.linalg.matrix_rank(Xt) == 3 and np.linalg.matrix_rank(Xt3) == 3
assert 3 + 1 == 4

# Contoh Soal 5.3: bobot berbeda, ramalan sama, norma minimum
wp = np.array([5, 10, 5, 0])
for t in (-5, 5):
    assert (Xt3 @ (wp + t * n)).tolist() == [20, 45, 55, 80]
assert (wp - 5 * n).tolist() == [5, 5, 0, 5]
assert (wp + 5 * n).tolist() == [5, 15, 10, -5]
assert Xt3[1] @ [5, 5, 0, 5] == 45 and Xt3[1] @ [5, 15, 10, -5] == 45
assert wp @ n == 15 and n @ n == 3 and Fr(-15, 3) == -5
wm = wp - 5 * n
assert wm @ n == 0 and wm @ wm == 75 and wp @ wp == 150
assert round(np.sqrt(75), 4) == 8.6603 and round(np.sqrt(150), 4) == 12.2474
assert np.allclose(np.linalg.lstsq(Xt3, y, rcond=None)[0], wm)

# Contoh Soal 5.4: empat ruang fundamental
e = np.array([-5, 5, 5, -5])
assert (y - Xt @ [5, 10, 5]).tolist() == e.tolist()
assert (Xt.T @ e).tolist() == [0, 0, 0]
assert [int(r @ n) for r in Xt3] == [0, 0, 0, 0]
for A, dims in [(Xt, (3, 1, 3, 0)), (Xt3, (3, 1, 3, 1))]:
    m_, n_ = A.shape
    r = np.linalg.matrix_rank(A)
    assert (r, m_ - r, r, n_ - r) == dims
print("Contoh Soal Bab 5: semua bilangan cocok")
