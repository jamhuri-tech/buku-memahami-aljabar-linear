"""Bab 5: rank, ruang nol, dan fitur rangkap x3 = x1 + x2, dicocokkan
dengan NumPy, SciPy, dan scikit-learn."""
import numpy as np
from scipy.linalg import null_space
from sklearn.linear_model import LinearRegression

from bab01_data import data_mini, rancangan

np.set_printoptions(precision=4, suppress=True)


def rapi(v):
    """Membulatkan dan membuang nol bertanda (-0.) supaya cetakan stabil."""
    return np.round(v, 10) + 0.0

X, y = data_mini()
Xt = rancangan(X)
X3 = np.column_stack([X, X.sum(axis=1)])      # x3 = x1 + x2
Xt3 = rancangan(X3)

# (1) Rank
print("(1) rank Xt  (4 x 3):", np.linalg.matrix_rank(Xt))
print("    rank Xt3 (4 x 4):", np.linalg.matrix_rank(Xt3))

# (2) Ruang nol
n3 = null_space(Xt3)[:, 0]
n3 = n3 / n3[1]
print("(2) ruang nol Xt3 dibentang", rapi(n3))
nT = null_space(Xt.T)[:, 0]
nT = nT / nT[1]
print("    ruang nol Xt^T dibentang", rapi(nT))

# (3) Bobot berbeda, ramalan sama
for w in ([5, 10, 5, 0], [5, 5, 0, 5], [5, 15, 10, -5]):
    w = np.array(w, float)
    print("(3) w =", w, " yhat", Xt3 @ w,
          "|w| %.4f" % np.linalg.norm(w))

# (4) Solusi norma minimum
w_ls = np.linalg.lstsq(Xt3, y, rcond=None)[0]
lr = LinearRegression().fit(X3, y)
print("(4) lstsq norma minimum   :", rapi(w_ls))
print("    LinearRegression w0, w:", round(lr.intercept_, 4),
      rapi(lr.coef_))
