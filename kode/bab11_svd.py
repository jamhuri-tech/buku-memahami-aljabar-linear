"""Bab 11: dekomposisi nilai singular, pseudoinvers, bilangan kondisi,
dan ridge lewat SVD, dicocokkan dengan NumPy."""
import numpy as np

from bab01_data import data_mini, rancangan

np.set_printoptions(precision=4, suppress=True)
X, y = data_mini()
Xt = rancangan(X)
Xc = X - X.mean(0)
yc = y - y.mean()


def svd_bertanda(A):
    """SVD tipis dengan tanda baku: unsur tak nol pertama setiap baris
    V^T positif, supaya cetakan tidak bergantung pada pustaka."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    pertama = np.argmax(np.abs(Vt) > 1e-8, axis=1)
    tanda = np.sign(Vt[np.arange(len(s)), pertama])
    return U * tanda, s, Vt * tanda[:, None]


# (1) SVD data terpusat
U, s, Vt = svd_bertanda(Xc)
print("(1) nilai singular:", s)
print("    sqrt(2) V^T =", np.round(np.sqrt(2) * Vt, 10) + 0.0)
print("    sqrt(2) U =")
print(np.round(np.sqrt(2) * U, 10) + 0.0)
print("    U S V^T = Xc:", np.allclose(U * s @ Vt, Xc))

# (2) Bilangan kondisi
sX = np.linalg.svd(Xt, compute_uv=False)
print("(2) nilai singular Xt:", sX)
print("    kondisi Xt      = %.4f, kuadratnya = %.4f"
      % (sX[0] / sX[-1], (sX[0] / sX[-1]) ** 2))
print("    kondisi Xt^T Xt = %.4f" % np.linalg.cond(Xt.T @ Xt))

# (3) Pseudoinvers
w_c = np.linalg.pinv(Xc) @ yc
X3 = np.column_stack([Xt, X.sum(1)])
w3 = np.linalg.pinv(X3) @ y
print("(3) pinv(Xc) yc =", w_c)
print("    pinv(Xt3) y =", np.round(w3, 10) + 0.0)
print("    nilai singular Xt3:", np.round(np.linalg.svd(
    X3, compute_uv=False), 4) + 0.0)

# (4) Ridge lewat SVD
for lam in (2.0, 10.0):
    f = s ** 2 / (s ** 2 + lam)
    w_svd = Vt.T @ (f / s * (U.T @ yc))
    w_ls = np.linalg.solve(Xc.T @ Xc + lam * np.eye(2), Xc.T @ yc)
    print("(4) lambda = %4.1f  faktor %s" % (lam, f))
    print("    w = %s  sama dengan Bab 8: %s"
          % (w_svd, np.allclose(w_svd, w_ls)))
