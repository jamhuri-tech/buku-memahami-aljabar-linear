"""Memeriksa setiap bilangan Contoh Soal dan prosa Bab 13."""
import math
from fractions import Fraction as Fr

import numpy as np

from bab01_data import data_mini, rancangan

X, y = data_mini()
Xt = rancangan(X).astype(int)

# Contoh Soal 13.1: bentuk kuadrat
A = np.array([[2, 1], [0, 3]])
assert (A + A.T).tolist() == [[4, 1], [1, 6]]
w = np.array([1, -2])
assert ((A + A.T) @ w).tolist() == [2, -11]
assert 4 * 1 + (-2) == 2 and 1 + 6 * (-2) == -11
wl = np.array([0, 10, 5])
g = -2 / 4 * Xt.T @ (y - Xt @ wl)
assert np.allclose(g, [-10, -30, -30])

# Contoh Soal 13.2: regresi logistik
yb = (y >= 55).astype(int)
assert yb.tolist() == [0, 0, 1, 1]
pmy = np.array([Fr(1, 2)] * 4) - yb
assert [sum(c * d for c, d in zip(col, pmy)) for col in Xt.T] == [0, -3, -1]
grad = [Fr(v, 4) for v in (0, -3, -1)]
assert grad == [0, Fr(-3, 4), Fr(-1, 4)]
Gi = np.array([[352, -48, -48], [-48, 40, -24], [-48, -24, 40]])
t = Gi @ np.array([0, -0.75, -0.25])
assert np.allclose(t, [48, -24, 8]) and np.allclose([36 + 12, -30 + 6, 18 - 10],
                                                    [48, -24, 8])
assert np.allclose(16 / 256 * t, [3, -1.5, 0.5])
H = Xt.T @ Xt / 16
assert np.allclose(np.linalg.solve(H, [0, -0.75, -0.25]), [3, -1.5, 0.5])
sig = lambda s: 1 / (1 + np.exp(-s))
w1 = np.array([-3, 1.5, -0.5])
p = sig(Xt @ w1)
loss = -np.mean(yb * np.log(p) + (1 - yb) * np.log(1 - p))
assert round(loss, 4) == 0.1269 and round(math.log(2), 4) == 0.6931

# Contoh Soal 13.3: propagasi balik
W1 = np.array([[1, -1], [1, 1]])
x = np.array([2, 4])
z = W1 @ x
assert z.tolist() == [-2, 6]
a = np.maximum(z, 0)
w2 = np.array([5, 5])
yh = w2 @ a
assert y[1] == 50 and x.tolist() == X[1].tolist()
assert a.tolist() == [0, 6] and yh == 30 and (yh - 50) ** 2 == 400
d = 2 * (yh - 50)
assert d == -40 and (d * a).tolist() == [0, -240]
assert (d * w2).tolist() == [-200, -200]
gz = d * w2 * (z > 0)
assert gz.tolist() == [0, -200]
assert np.outer(gz, x).tolist() == [[0, 0], [-400, -800]]

# Contoh Soal 13.4: turunan log det
A2 = np.array([[2, 1], [1, 3]])
assert round(np.linalg.det(A2)) == 5
eps = 1e-6
assert np.isclose((math.log(5 + 3 * eps) - math.log(5)) / eps, 0.6, atol=1e-5)
assert np.allclose(np.linalg.inv(A2).T, np.array([[3, -1], [-1, 2]]) / 5)
print("Contoh Soal Bab 13: semua bilangan cocok")
