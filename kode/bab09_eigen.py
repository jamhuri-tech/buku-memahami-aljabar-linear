"""Bab 9: nilai eigen dan vektor eigen, diagonalisasi, iterasi pangkat,
rantai Markov, PageRank, dan laju gradient descent."""
import numpy as np

from bab01_data import data_mini, rancangan

np.set_printoptions(precision=4, suppress=True)
X, y = data_mini()
Xt = rancangan(X)
Xc = X - X.mean(0)
S = Xc.T @ Xc
G = Xt.T @ Xt

# (1) Nilai eigen dengan NumPy
lam, V = np.linalg.eigh(S)
print("(1) S = Xc^T Xc, nilai eigen:", lam)
V = V * np.sign(V[0])          # tanda: unsur pertama positif
print("    vektor eigen (kolom) * sqrt(2):")
print(V * np.sqrt(2))
lamG = np.linalg.eigvalsh(G)
print("    nilai eigen Xt^T Xt:", lamG)
print("    46 -+ 6 sqrt(57)   :",
      np.round([46 - 6 * np.sqrt(57), 46 + 6 * np.sqrt(57)], 4))

# (2) Iterasi pangkat
x = np.array([1.0, 0.0])
print("(2) k   x_k (dinormalkan)    hasil bagi Rayleigh")
for k in range(1, 7):
    x = S @ x
    x = x / np.linalg.norm(x)
    print("    %d   %s   %.4f" % (k, x, x @ S @ x))

# (3) Rantai Markov dan PageRank
P = np.array([[0.9, 0.5], [0.1, 0.5]])
lp, Vp = np.linalg.eig(P)
pi = Vp[:, np.argmax(lp)]
pi = pi / pi.sum()
print("(3) nilai eigen P:", np.sort(lp)[::-1], " stasioner:", pi)
print("    P^20 =")
print(np.linalg.matrix_power(P, 20))
# Empat halaman: 1 -> 2, 3; 2 -> 3; 3 -> 1; 4 -> 3
L = np.array([[0, 0, 1, 0], [0.5, 0, 0, 0], [0.5, 1, 0, 1],
              [0, 0, 0, 0]])
d = 0.85
M = d * L + (1 - d) / 4 * np.ones((4, 4))
r = np.ones(4) / 4
for _ in range(100):
    r = M @ r
lm, Vm = np.linalg.eig(M)
v = np.real(Vm[:, np.argmax(np.real(lm))])
print("    PageRank (iterasi):", r)
print("    PageRank (eig)    :", v / v.sum())

# (4) Laju gradient descent = 1 - (2 eta / m) lambda
m = len(y)
eta = 0.02
w_hat = np.linalg.solve(G, Xt.T @ y)
w = np.zeros(3)
lG, VG = np.linalg.eigh(G)
galat = []
for t in range(11):
    galat.append(VG.T @ (w - w_hat))
    w = w - eta * (-2 / m) * Xt.T @ (y - Xt @ w)
galat = np.array(galat)
print("(4) faktor teori 1 - 2 eta lambda / m:", 1 - 2 * eta * lG / m)
print("    rasio galat terukur (t = 10 / 9)  :",
      galat[10] / galat[9])
