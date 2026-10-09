"""Bab 2: norma, hasil kali titik, kosinus, korelasi, dan jarak
antarsampel, dicocokkan dengan NumPy dan SciPy."""
import numpy as np
from scipy.spatial.distance import cdist

from bab01_data import SEED, data_mini

np.set_printoptions(precision=4, suppress=True)
X, y = data_mini()
x1, x2 = X[:, 0], X[:, 1]

# (1) Norma kolom x1 dan x2
print("(1) norma      L1     L2      Linf")
for nama, v in [("x1", x1), ("x2", x2), ("x1 - x2", x1 - x2)]:
    print("    %-8s %5.1f  %6.4f  %4.1f" % (
        nama, np.linalg.norm(v, 1), np.linalg.norm(v),
        np.linalg.norm(v, np.inf)))

# (2) Kosinus vektor terpusat sama dengan korelasi Pearson
Z = np.column_stack([X, y])
Zc = Z - Z.mean(axis=0)
Zn = Zc / np.linalg.norm(Zc, axis=0)
print("(2) kosinus vektor terpusat (x1, x2, y):")
print(Zn.T @ Zn)
print("    np.corrcoef sama:", np.allclose(Zn.T @ Zn,
                                         np.corrcoef(Z.T)))

# (3) Matriks jarak antarsampel
D2 = cdist(X, X, "sqeuclidean")
sq = (X ** 2).sum(axis=1)
D2b = sq[:, None] + sq[None, :] - 2 * X @ X.T
print("(3) jarak Euclid kuadrat:")
print(D2.astype(int))
print("    lewat |a|^2 + |b|^2 - 2 a.b sama:", np.allclose(D2, D2b))
print("    jarak Manhattan:")
print(cdist(X, X, "cityblock").astype(int))

# (4) Kosinus dua vektor acak pada dimensi tinggi
rng = np.random.default_rng(SEED)
print("(4) dimensi  rata-rata |cos|  simpangan baku cos")
for d in (2, 10, 100, 1000, 10000):
    A = rng.standard_normal((2000, d))
    B = rng.standard_normal((2000, d))
    c = (A * B).sum(1) / np.linalg.norm(A, axis=1) \
        / np.linalg.norm(B, axis=1)
    print("    %6d   %.4f          %.4f" % (d, np.abs(c).mean(),
                                            c.std()))
