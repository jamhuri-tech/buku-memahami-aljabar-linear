"""Bab 6: invers, determinan, dan pembaruan Sherman-Morrison,
dicocokkan dengan NumPy dan SciPy."""
import numpy as np
from scipy.linalg import lu

from bab01_data import data_mini, rancangan

np.set_printoptions(precision=4, suppress=True)


def rapi(v):
    """Membulatkan dan membuang nol bertanda supaya cetakan stabil."""
    return np.round(v, 8) + 0.0

X, y = data_mini()
Xt = rancangan(X)
G = Xt.T @ Xt

# (1) Invers 2 x 2 dan invers persamaan normal
A = np.array([[2.0, 1.0], [1.0, 3.0]])
rumus = np.array([[3, -1], [-1, 2]]) / 5
print("(1) inv(A) =", np.linalg.inv(A).tolist())
print("    rumus 2 x 2 sama:", np.allclose(np.linalg.inv(A), rumus))
print("    256 * inv(Xt^T Xt) =")
print(np.round(256 * np.linalg.inv(G)).astype(int))
print("    inv(G) @ Xt^T y =", rapi(np.linalg.inv(G) @ Xt.T @ y))

# (2) Determinan tiga cara
P, L, U = lu(G)
print("(2) det G: np.linalg.det = %.4f" % np.linalg.det(G))
tanda, logdet = np.linalg.slogdet(G)
print("    slogdet: tanda %+.0f, log|det| = %.4f, exp = %.4f"
      % (tanda, logdet, np.exp(logdet)))
print("    det P * hasil kali pivot = %.4f"
      % (np.linalg.det(P) * np.prod(np.diag(U))))
Xt3 = np.column_stack([Xt, X.sum(1)])
print("    |det| dengan fitur rangkap < 1e-9:",
      abs(np.linalg.det(Xt3.T @ Xt3)) < 1e-9)

# (3) Sherman-Morrison: menambah sampel kelima
x5, y5 = np.array([1.0, 3.0, 3.0]), 11.0
Gi = np.linalg.inv(G)
u = Gi @ x5
Gi_baru = Gi - np.outer(u, u) / (1 + x5 @ u)
G_baru = G + np.outer(x5, x5)
w = Gi @ Xt.T @ y
w_baru = w + Gi_baru @ x5 * (y5 - x5 @ w)
w_langsung = np.linalg.solve(G_baru, Xt.T @ y + y5 * x5)
print("(3) inv(G) x5 =", rapi(u), " x5^T inv(G) x5 = %.4f" % (x5 @ u))
print("    Sherman-Morrison sama dengan inv langsung:",
      np.allclose(Gi_baru, np.linalg.inv(G_baru)))
print("    w baru (pembaruan) :", rapi(w_baru))
print("    w baru (dari awal) :", rapi(w_langsung))

# (4) Determinan bukan ukuran kedekatan ke singular
for nama, M in [("0.1 I (10 x 10)", 0.1 * np.eye(10)),
                ("[[1, 1], [1, 1.0001]]",
                 np.array([[1.0, 1.0], [1.0, 1.0001]]))]:
    print("(4) %-22s det = %.1e  cond = %.1e"
          % (nama, np.linalg.det(M), np.linalg.cond(M)))
