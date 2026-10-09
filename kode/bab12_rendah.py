"""Bab 12: hampiran rank rendah (Eckart-Young), PCA lewat SVD,
kompresi citra, dan faktorisasi matriks untuk rekomendasi."""
import matplotlib.cbook as cbook
import matplotlib.image as mimage
import numpy as np
from sklearn.decomposition import PCA

from bab01_data import BENIH, data_mini

np.set_printoptions(precision=4, suppress=True)
X, y = data_mini()
xbar = X.mean(0)
Xc = X - xbar


def bertanda(U, s, Vt):
    """Tanda baku: unsur tak nol pertama setiap baris V^T positif."""
    pertama = np.argmax(np.abs(Vt) > 1e-8, axis=1)
    t = np.sign(Vt[np.arange(len(s)), pertama])
    return U * t, s, Vt * t[:, None]


# (1) Hampiran rank 1 data mini
U, s, Vt = bertanda(*np.linalg.svd(Xc, full_matrices=False))
A1 = s[0] * np.outer(U[:, 0], Vt[0])
print("(1) hampiran rank 1 dari Xc =")
print(np.round(A1, 10) + 0.0)
print("    galat Frobenius^2 = %.4f = sigma_2^2" % ((Xc - A1) ** 2).sum())
print("    bagian energi rank 1 = %.4f" % (s[0] ** 2 / (s ** 2).sum()))

# (2) PCA lewat SVD lawan scikit-learn
skor = Xc @ Vt.T
pca = PCA(2).fit(X)
skor_sk = pca.transform(X) * np.sign(pca.components_[:, 0])
print("(2) skor PCA * sqrt(2) =")
print(np.round(np.sqrt(2) * skor, 10) + 0.0)
print("    varians (pembagi m):", s ** 2 / len(X))
print("    explained_variance_ratio_:", pca.explained_variance_ratio_)
print("    skor sama dengan sklearn:", np.allclose(skor, skor_sk))

# (3) Kompresi citra
jalur = cbook.get_sample_data("grace_hopper.jpg", asfileobj=False)
G = mimage.imread(jalur).astype(float).mean(axis=2)
m, n = G.shape
sG = np.linalg.svd(G, compute_uv=False)
print("(3) citra %d x %d, %d bilangan" % (m, n, m * n))
for k in (5, 20, 50):
    simpan = k * (m + n + 1)
    galat = np.sqrt((sG[k:] ** 2).sum() / (sG ** 2).sum())
    print("    rank %2d: %6d bilangan (%4.1f%%), galat relatif %4.1f%%"
          % (k, simpan, 100 * simpan / (m * n), 100 * galat))

# (4) Faktorisasi matriks: rekomendasi dengan skor hilang
A = np.array([[2, 0], [1, 1], [0, 2], [2, 1], [1, 2], [1, 0]], float)
B = np.array([[2, 1], [1, 0], [0, 2], [1, 2], [2, 2]], float)
R = A @ B.T
hilang = [(0, 3), (1, 1), (2, 0), (3, 2), (4, 4), (5, 2)]
ada = np.ones_like(R, bool)
for i, j in hilang:
    ada[i, j] = False
rng = np.random.default_rng(BENIH)
P, Q = rng.random((6, 2)), rng.random((5, 2))
lam = 1e-3
for _ in range(2000):           # kuadrat terkecil bergantian (ALS)
    for i in range(6):
        Qi = Q[ada[i]]
        P[i] = np.linalg.solve(Qi.T @ Qi + lam * np.eye(2),
                               Qi.T @ R[i, ada[i]])
    for j in range(5):
        Pj = P[ada[:, j]]
        Q[j] = np.linalg.solve(Pj.T @ Pj + lam * np.eye(2),
                               Pj.T @ R[ada[:, j], j])
print("(4) skor lengkap R = A B^T (rank 2) =")
print(R.astype(int))
print("    ramalan ALS untuk skor yang disembunyikan:")
for i, j in hilang:
    print("      pengguna %d, film %d: %.2f (sebenarnya %d)"
          % (i + 1, j + 1, round(P[i] @ Q[j], 2) + 0.0, R[i, j]))
