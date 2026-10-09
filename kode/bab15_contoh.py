"""Memeriksa setiap bilangan Contoh Soal dan prosa Bab 15."""
from fractions import Fraction as Fr

import numpy as np
from sklearn.datasets import load_digits

d = load_digits()
X = d.data
assert X.shape == (1797, 64)
gambar = d.images
assert np.array_equal(gambar.reshape(1797, 64), X)

# Contoh Soal 15.1: indeks piksel
for j, (r, c) in ((0, (0, 0)), (32, (4, 0)), (39, (4, 7))):
    assert divmod(j, 8) == (r, c) and 8 * r + c == j
    assert (gambar[:, r, c] == 0).all() and (X[:, j] == 0).all()
assert np.linalg.matrix_rank(X) == 61 and 64 - 61 == 3

# Contoh Soal 15.2: penyimpanan PCA
assert 1797 * 40 == 71880 and 40 * 64 == 2560
assert 71880 + 2560 + 64 == 74504 and 1797 * 64 == 115008
assert round(100 * 74504 / 115008, 1) == 64.8

# Contoh Soal 15.3: target satu-lawan-semua dan galat
t3 = -np.ones(10)
t3[3] = 1
assert t3.tolist() == [-1, -1, -1, 1, -1, -1, -1, -1, -1, -1]
skor = np.array([-0.9, -1.1, 0.1, 0.4, -0.8, 0.3, -1.0, -0.7, 0.2, -0.5])
assert np.argmax(skor) == 3
t5 = -np.ones(10)
t5[5] = 1
g = t5 - skor
assert np.allclose(g, [-0.1, 0.1, -1.1, -1.4, -0.2, 0.7, 0, -0.3, -1.2, -0.5])
assert np.isclose((g ** 2).sum(), 5.5)

# Contoh Soal 15.4: kondisi
s1, k = 475.453, 833.6
smin = s1 / k
assert round(smin, 4) == 0.5704
assert round(k ** 2 / 1e5, 2) == 6.95
assert round(s1 ** 2) == 226056 and round(smin ** 2, 3) == 0.325
assert round((s1 ** 2 + 100) / (smin ** 2 + 100)) == 2254
assert (k ** 2) / 2254 > 300
print("Contoh Soal Bab 15: semua bilangan cocok")
