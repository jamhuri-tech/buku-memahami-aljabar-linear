"""Bab 10: teorema spektral, bentuk kuadrat, definit positif,
Cholesky, kovarians, jarak Mahalanobis, dan whitening."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import mahalanobis

from bab01_data import SEED, data_mini, rancangan

np.set_printoptions(precision=4, suppress=True)
X, y = data_mini()
Xt = rancangan(X)
Xc = X - X.mean(0)
S = Xc.T @ Xc
G = Xt.T @ Xt

# (1) Teorema spektral
lam, Q = np.linalg.eigh(S)
print("(1) Q^T Q = I:", np.allclose(Q.T @ Q, np.eye(2)),
      "  Q Lambda Q^T = S:", np.allclose(Q @ np.diag(lam) @ Q.T, S))
jumlah = sum(l * np.outer(q, q) for l, q in zip(lam, Q.T))
print("    jumlah lambda_k q_k q_k^T =", np.round(jumlah, 10).tolist())


# (2) Uji definit positif: nilai eigen dan Cholesky
def definit_positif(A):
    """Definit positif bila nilai eigen terkecil jelas positif.
    Cholesky dipakai untuk matriks yang jelas; matriks singular diuji
    dengan nilai eigen supaya tidak bergantung pada galat pembulatan."""
    lam = np.linalg.eigvalsh(A)
    return bool(lam[0] > 1e-9 * lam[-1])


B = np.array([[1.0, 2.0], [2.0, 1.0]])
X3 = np.column_stack([Xt, X.sum(1)])
for nama, A in [("S", S), ("B", B), ("G", G), ("G3", X3.T @ X3)]:
    print("(2) %-2s nilai eigen %s" % (nama,
          np.round(np.linalg.eigvalsh(A), 4) + 0.0))
    print("       definit positif: %s" % definit_positif(A))

# (3) Cholesky persamaan normal
L = np.linalg.cholesky(G)
print("(3) L =")
print(L)
print("    L = R^T dari QR (sampai tanda):",
      np.allclose(np.abs(L.T), np.abs(np.linalg.qr(Xt)[1])))
print("    w lewat cho_solve:", cho_solve(cho_factor(G), Xt.T @ y))

# (4) Kovarians, Mahalanobis, whitening
C = Xc.T @ Xc / len(X)
Ci = np.linalg.inv(C)
d2 = [mahalanobis(x, X.mean(0), Ci) ** 2 for x in X]
print("(4) kovarians =", C.tolist())
print("    4 * invers   =", np.round(4 * Ci, 10).tolist())
print("    Mahalanobis^2 ke rata-rata:", np.round(d2, 10) + 0.0)
print("    Euclid^2 ke rata-rata     :", (Xc ** 2).sum(1))
lc, Qc = np.linalg.eigh(C)
Z = Xc @ Qc / np.sqrt(lc)
print("    kovarians sesudah whitening =",
      np.round(Z.T @ Z / len(X), 10) + 0.0)

# (5) Membangkitkan sampel normal dengan Cholesky
rng = np.random.default_rng(SEED)
Lc = np.linalg.cholesky(C)
sampel = rng.standard_normal((100000, 2)) @ Lc.T + X.mean(0)
K = np.cov(sampel.T, bias=True)
print("(5) kovarians 100 000 sampel:", np.round(K, 2).tolist())
