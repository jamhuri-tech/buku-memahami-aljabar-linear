"""Bab 3: empat cara memandang perkalian matriks, matriks Gram, ganti
fitur sebagai perkalian matriks, lapisan linear, dan biaya hitung."""
import numpy as np

from bab01_data import data_mini, rancangan

np.set_printoptions(precision=4, suppress=True)
X, y = data_mini()
Xt = rancangan(X)
m, n = X.shape

# (1) Empat cara menghitung G = Xt^T Xt
A, B = Xt.T, Xt
G1 = np.array([[A[i] @ B[:, j] for j in range(3)] for i in range(3)])
G2 = np.column_stack([A @ B[:, j] for j in range(3)])
G3 = np.vstack([A[i] @ B for i in range(3)])
G4 = sum(np.outer(A[:, k], B[k]) for k in range(m))
print("(1) Xt^T Xt =")
print(G1.astype(int))
print("    unsur, kolom, baris, hasil kali luar sama:",
      all(np.array_equal(G1, G) for G in (G2, G3, G4, A @ B)))

# (2) Matriks Gram baris dan jarak
K = X @ X.T
q = np.diag(K)
print("(2) X X^T =")
print(K.astype(int))
print("    jarak kuadrat = q_i + q_k - 2 K_ik:")
print((q[:, None] + q[None, :] - 2 * K).astype(int))

# (3) Ganti fitur: t = x1 + x2, d = x1 - x2
W = np.array([[1, 1], [1, -1]])
F = X @ W
v = np.linalg.solve(W, [2, 1])
print("(3) X W =", F.astype(int).tolist())
print("    bobot baru untuk (t, d):", v)
print("    ramalan sama:",
      np.allclose(1 + X @ [2, 1], 1 + F @ v))

# (4) Lapisan linear pada satu batch
rng = np.random.default_rng(20261010)
Wl = rng.standard_normal((5, 2))
bl = rng.standard_normal(5)
H = X @ Wl.T + bl
print("(4) batch X", X.shape, "@ W^T", Wl.T.shape, "+ b", bl.shape,
      "->", H.shape)
print("    baris ke-2 sama dengan W x_2 + b:",
      np.allclose(H[1], Wl @ X[1] + bl))


# (5) Banyaknya perkalian skalar: urutan kurung
def biaya(p, q, r):
    return p * q * r


N = 2000
kiri = biaya(N, N, N) + biaya(N, N, 1)
kanan = biaya(N, N, 1) + biaya(N, N, 1)
print("(5) A, B: %d x %d, v: %d x 1" % (N, N, N))
print("    (A B) v : %15d perkalian" % kiri)
print("    A (B v) : %15d perkalian" % kanan)
print("    perbandingan        : %d kali" % (kiri // kanan))
