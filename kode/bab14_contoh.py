"""Memeriksa setiap bilangan Contoh Soal dan prosa Bab 14."""
import math

import numpy as np
import scipy.sparse as sp

from bab01_data import data_mini, rancangan

X, y = data_mini()

# epsilon mesin dan jarak bilangan di sekitar 1e16
assert np.finfo(float).eps == 2.0 ** -52
assert 1e16 + 1 == 1e16 and np.spacing(1e16) == 2.0
# Contoh Soal 14.1: varians tergeser
x = X[:, 0] + 1e8
assert (10 ** 8 + 1) ** 2 == 10 ** 16 + 2 * 10 ** 8 + 1
assert np.isclose((1e8 + 3) ** 2, 1.00000006e16)
assert np.allclose(x - x.mean(), [-2, -1, 1, 2]) and np.var(x) == 2.5
assert 1e8 + 5 < 2 ** 53

# Contoh Soal 14.2: sistem peka
A = np.array([[1, 1], [1, 1.0001]])
assert np.isclose(np.linalg.det(A), 1e-4)
assert np.allclose(np.linalg.solve(A, [2, 2.0001]), [1, 1])
assert np.allclose(np.linalg.solve(A, [2, 2.0002]), [0, 2])
nb = math.hypot(2, 2.0001)
assert round(nb, 2) == 2.83 and round(100 * 1e-4 / nb, 4) == 0.0035
rasio = (math.sqrt(2) / math.sqrt(2)) / (1e-4 / nb)
assert round(rasio / 1e4, 1) == 2.8 and rasio < np.linalg.cond(A)

# Contoh Soal 14.3: pemusatan
Xtc = rancangan(X - X.mean(0))
assert (Xtc.T @ Xtc).astype(int).tolist() == [[4, 0, 0], [0, 10, 6],
                                              [0, 6, 10]]
assert np.allclose(np.linalg.eigvalsh(Xtc.T @ Xtc), [4, 4, 16])
assert np.allclose(np.linalg.svd(Xtc, compute_uv=False), [4, 2, 2])
assert np.isclose(np.linalg.cond(Xtc), 2)
assert [round(1 - 0.01 * l, 2) for l in (16, 4)] == [0.84, 0.96]

# CSR kecil dan Contoh Soal 14.4
M = sp.csr_matrix(np.array([[5, 0, 0], [0, 0, 3], [2, 0, 7]]))
assert M.data.tolist() == [5, 3, 2, 7] and M.indices.tolist() == [0, 2, 0, 2]
assert M.indptr.tolist() == [0, 1, 2, 4]
assert 10 ** 8 * 8 == 800_000_000
assert 50_000 * 8 + 50_000 * 4 + 10_001 * 4 == 640_004
assert 49_986 * 12 + 10_001 * 4 == 639_836
assert round(100 * 640_004 / 8e8, 2) == 0.08
print("Contoh Soal Bab 14: semua bilangan cocok")
