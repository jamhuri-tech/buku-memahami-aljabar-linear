"""Bab 13: kalkulus matriks. Gradien bentuk linear dan kuadrat,
gradien dan Hessian regresi logistik, langkah Newton, propagasi balik
jaringan kecil, dan turunan log det, diperiksa dengan beda hingga."""
import numpy as np

from bab01_data import data_mini, rancangan

np.set_printoptions(precision=4, suppress=True)
X, y = data_mini()
Xt = rancangan(X)
m = len(y)
h = 1e-6


def r(v):
    """Membulatkan dan membuang nol bertanda supaya cetakan stabil."""
    return np.round(v, 4) + 0.0


def beda_hingga(f, w):
    """Gradien numerik dengan beda hingga terpusat."""
    w = np.asarray(w, float)
    g = np.zeros_like(w)
    for j in range(w.size):
        e = np.zeros_like(w)
        e.flat[j] = h
        g.flat[j] = (f(w + e) - f(w - e)) / (2 * h)
    return g


# (1) Bentuk kuadrat dan kuadrat terkecil
A = np.array([[2.0, 1.0], [0.0, 3.0]])
w = np.array([1.0, -2.0])
print("(1) grad w^T A w: rumus", r((A + A.T) @ w),
      " beda hingga", r(beda_hingga(lambda v: v @ A @ v, w)))
L = lambda v: np.sum((y - Xt @ v) ** 2) / m
w = np.array([0.0, 2.0, 1.0])
print("    grad L(0, 2, 1): rumus      ", r(-2 / m * Xt.T @ (y - Xt @ w)))
print("                     beda hingga", r(beda_hingga(L, w)))

# (2) Regresi logistik: lulus bila skor >= 11
yb = (y >= 11).astype(float)
sig = lambda t: 1 / (1 + np.exp(-t))


def log_loss(v):
    p = sig(Xt @ v)
    return -np.mean(yb * np.log(p) + (1 - yb) * np.log(1 - p))


w0 = np.zeros(3)
p = sig(Xt @ w0)
grad = Xt.T @ (p - yb) / m
H = Xt.T @ np.diag(p * (1 - p)) @ Xt / m
print("(2) y lulus =", yb)
print("    gradien di w = 0: rumus      ", r(grad))
print("                      beda hingga", r(beda_hingga(log_loss, w0)))
print("    16 * Hessian =")
print(r(16 * H))
w1 = w0 - np.linalg.solve(H, grad)
print("    langkah Newton: w =", r(w1), "loss %.4f -> %.4f"
      % (log_loss(w0), log_loss(w1)))

# (3) Propagasi balik jaringan 2-2-1 dengan ReLU, sampel 2
W1 = np.array([[1.0, -1.0], [1.0, 1.0]])
b1 = np.zeros(2)
w2 = np.array([1.0, 1.0])
b2 = 0.0
x, t = X[1], y[1]


def maju(W1, b1, w2, b2):
    z = W1 @ x + b1
    a = np.maximum(z, 0)
    return z, a, w2 @ a + b2


z, a, yh = maju(W1, b1, w2, b2)
d = 2 * (yh - t)                       # dL / d yhat
g_w2, g_b2 = d * a, d
g_a = d * w2
g_z = g_a * (z > 0)
g_W1, g_b1 = np.outer(g_z, x), g_z
print("(3) z =", r(z), " a =", r(a), " yhat = %.1f" % yh)
print("    dL/dW1 =", r(g_W1).tolist(), " dL/db1 =", r(g_b1))
print("    dL/dw2 =", r(g_w2))
fd = beda_hingga(lambda V: (maju(V, b1, w2, b2)[2] - t) ** 2, W1)
print("    dL/dW1 beda hingga:", r(fd).tolist())

# (4) Turunan log det
A = np.array([[2.0, 1.0], [1.0, 3.0]])
print("(4) d log det / dA: rumus", r(np.linalg.inv(A).T).tolist())
print("    beda hingga         ",
      r(beda_hingga(lambda M: np.log(np.linalg.det(M)), A)).tolist())
