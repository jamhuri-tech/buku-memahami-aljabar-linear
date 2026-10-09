"""Bab 8: kuadrat terkecil dengan empat cara, dekomposisi jumlah
kuadrat, galat baku bobot, Frisch-Waugh-Lovell, dan ridge,
dicocokkan dengan NumPy, scikit-learn, dan statsmodels."""
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge

from bab01_data import data_mini, rancangan

np.set_printoptions(precision=4, suppress=True)
X, y = data_mini()
Xt = rancangan(X)
m, n = X.shape

# (1) Empat cara, satu jawaban
w_normal = np.linalg.solve(Xt.T @ Xt, Xt.T @ y)
Q, R = np.linalg.qr(Xt)
w_qr = np.linalg.solve(R, Q.T @ y)
w_lstsq = np.linalg.lstsq(Xt, y, rcond=None)[0]
lr = LinearRegression().fit(X, y)
print("(1) persamaan normal:", w_normal)
print("    QR              :", w_qr)
print("    lstsq           :", w_lstsq)
print("    LinearRegression:", np.r_[lr.intercept_, lr.coef_])

# (2) Jumlah kuadrat dan R^2
yhat = Xt @ w_normal
e = y - yhat
SST = ((y - y.mean()) ** 2).sum()
SSR = ((yhat - y.mean()) ** 2).sum()
SSE = (e ** 2).sum()
print("(2) SST = %.4f  SSR = %.4f  SSE = %.4f" % (SST, SSR, SSE))
print("    R^2 = %.6f  korelasi(y, yhat)^2 = %.6f"
      % (SSR / SST, np.corrcoef(y, yhat)[0, 1] ** 2))

# (3) Galat baku bobot
s2 = SSE / (m - n - 1)
se = np.sqrt(s2 * np.diag(np.linalg.inv(Xt.T @ Xt)))
ols = sm.OLS(y, Xt).fit()
print("(3) sigma^2 = %.4f  galat baku:" % s2, se)
print("    statsmodels         :", ols.bse)

# (4) Frisch-Waugh-Lovell untuk w2
A = Xt[:, :2]
r_x2 = X[:, 1] - A @ np.linalg.lstsq(A, X[:, 1], rcond=None)[0]
print("(4) sisa x2 sesudah 1 dan x1:", r_x2)
print("    w2 = r^T y / r^T r = %.4f" % (r_x2 @ y / (r_x2 @ r_x2)))

# (5) Ridge dengan fitur terpusat
Xc = X - X.mean(0)
yc = y - y.mean()
print("(5) lambda   w1       w2      (rumus)  sklearn Ridge")
for lam in (0.0, 2.0, 10.0, 100.0):
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(2), Xc.T @ yc)
    rg = Ridge(alpha=lam).fit(X, y)
    print("    %6.1f  %7.4f  %7.4f    %s" % (lam, w[0], w[1],
                                            rg.coef_))
