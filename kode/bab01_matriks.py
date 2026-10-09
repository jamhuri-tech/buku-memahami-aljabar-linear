"""Bab 1: data sebagai matriks di NumPy, model linear untuk satu
sampel dan untuk seluruh data, gradien, dan gradient descent."""
import numpy as np

from bab01_data import data_mini, rancangan

np.set_printoptions(precision=4, suppress=True)
X, y = data_mini()
Xt = rancangan(X)
m, n = X.shape

# (1) Bentuk (shape) setiap objek
print("(1) shape: X", X.shape, " y", y.shape, " Xt", Xt.shape)
print("    baris ke-2 X[1]   =", X[1], " shape", X[1].shape)
print("    kolom x2 X[:, 1]  =", X[:, 1], " shape", X[:, 1].shape)
print("    rata-rata kolom   =", X.mean(axis=0))
print("    X - X.mean(0) =")
print((X - X.mean(axis=0)).astype(int))

# (2) Jebakan shape: (4,) lawan (4, 1)
yk = y.reshape(-1, 1)
print("(2) y - y.mean()          shape", (y - y.mean()).shape)
print("    yk - y               shape", (yk - y).shape)


# (3) Loss dan gradien untuk beberapa bobot
def loss(w):
    e = y - Xt @ w
    return e @ e / m


def gradien(w):
    return -2 / m * Xt.T @ (y - Xt @ w)


print("(3) w            loss      gradien")
for w in ([0, 2, 1], [1, 2, 1]):
    w = np.array(w, float)
    print("    %-12s %-9.4f %s" % (w, loss(w), gradien(w)))

# (4) Gradien rumus lawan beda hingga terpusat
w = np.array([0.0, 2.0, 1.0])
h = 1e-6
fd = np.array([(loss(w + h * e) - loss(w - h * e)) / (2 * h)
               for e in np.eye(3)])
print("(4) rumus      :", gradien(w))
print("    beda hingga:", fd)

# (5) Gradient descent dari nol
w = np.zeros(3)
eta = 0.02
riwayat = [loss(w)]
for t in range(1, 3001):
    w = w - eta * gradien(w)
    riwayat.append(loss(w))
    if t in (1, 10, 100, 1000, 3000):
        print("(5) t = %5d  w = %s  loss = %.6f" % (t, w, loss(w)))
