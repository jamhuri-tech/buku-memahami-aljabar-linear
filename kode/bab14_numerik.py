"""Bab 14: bilangan floating point, pembatalan, bilangan kondisi,
pemusatan, laju gradient descent, dan sparse matrix."""
import numpy as np
import scipy.sparse as sp

from bab01_data import SEED, data_mini, rancangan

np.set_printoptions(precision=4, suppress=True)
X, y = data_mini()
Xt = rancangan(X)

# (1) Floating point dan pembatalan
print("(1) epsilon mesin float64 = %.4e = 2^-52" % np.finfo(float).eps)
print("    0.1 + 0.2 == 0.3 :", 0.1 + 0.2 == 0.3,
      "  selisih = %.1e" % (0.1 + 0.2 - 0.3))
# x1 digeser 10^8. Penjumlahan ditulis berurutan dengan float Python
# supaya hasilnya sama di setiap komputer; np.sum memakai urutan
# penjumlahan yang bergantung pada prosesor.
x = [float(v) + 1e8 for v in X[:, 0]]
jumlah = jumlah_kuadrat = 0.0
for v in x:
    jumlah += v
    jumlah_kuadrat += v * v
rata = jumlah / len(x)
v_satu = jumlah_kuadrat / len(x) - rata * rata   # bukan rata ** 2: pow libm
v_dua = 0.0
for v in x:
    v_dua += (v - rata) ** 2
v_dua /= len(x)
print("    varians x1 + 1e8: E[x^2] - E[x]^2 = %.1f" % v_satu)
print("                      E[(x - xbar)^2] = %.1f" % v_dua)

# (2) Sistem yang peka
A = np.array([[1.0, 1.0], [1.0, 1.0001]])
for b in ([2.0, 2.0001], [2.0, 2.0002]):
    print("(2) b = %s -> x = %s" % (b, np.round(np.linalg.solve(A, b),
                                                   6) + 0.0))
print("    kondisi A = %.0f" % np.linalg.cond(A))

# (3) Kondisi matriks rancangan dan laju gradient descent
Xtc = rancangan(X - X.mean(0))
Xm = rancangan(np.column_stack([60 * X[:, 0], X[:, 1]]))
for nama, M in [("asli", Xt), ("terpusat", Xtc), ("x1 dalam menit", Xm)]:
    print("(3) %-15s kondisi = %8.2f" % (nama, np.linalg.cond(M)))


def iterasi_gd(M, eta, tol=1e-6):
    w_hat = np.linalg.lstsq(M, y, rcond=None)[0]
    L_min = np.mean((y - M @ w_hat) ** 2)
    w = np.zeros(M.shape[1])
    for t in range(1, 100001):
        w -= eta * (-2 / len(y)) * M.T @ (y - M @ w)
        if np.mean((y - M @ w) ** 2) - L_min < tol:
            return t
    return None


print("    iterasi GD (eta = 0.02) sampai loss < minimum + 1e-6:")
print("      asli %d, terpusat %d" % (iterasi_gd(Xt, 0.02),
                                       iterasi_gd(Xtc, 0.02)))

# (4) Sparse matrix
n = 10000
rng = np.random.default_rng(SEED)
baris = np.repeat(np.arange(n), 5)
kolom = rng.integers(0, n, size=5 * n)
S = sp.csr_matrix((np.ones(5 * n), (baris, kolom)), shape=(n, n))
padat = n * n * 8
jarang = S.data.nbytes + S.indices.nbytes + S.indptr.nbytes
print("(4) %d x %d, %d unsur tak nol" % (n, n, S.nnz))
print("    padat : %11d byte" % padat)
print("    CSR   : %11d byte (%.3f%%)" % (jarang, 100 * jarang / padat))
