"""Bab 7: proyeksi, matriks topi, Gram-Schmidt klasik dan
termodifikasi, dan QR, dicocokkan dengan NumPy."""
import numpy as np

from bab01_data import data_mini, rancangan

np.set_printoptions(precision=4, suppress=True)
X, y = data_mini()
Xt = rancangan(X)


def gs_klasik(A):
    """Gram-Schmidt klasik: setiap kolom dikurangi proyeksinya pada
    semua q sebelumnya, dengan koefisien dari kolom asli."""
    m, n = A.shape
    Q, R = np.zeros((m, n)), np.zeros((n, n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R


def gs_modifikasi(A):
    """Gram-Schmidt termodifikasi: koefisien dihitung dari vektor
    yang sudah dikurangi, satu q demi satu q."""
    m, n = A.shape
    Q, R = np.zeros((m, n)), np.zeros((n, n))
    V = A.astype(float).copy()
    for j in range(n):
        R[j, j] = np.linalg.norm(V[:, j])
        Q[:, j] = V[:, j] / R[j, j]
        for k in range(j + 1, n):
            R[j, k] = Q[:, j] @ V[:, k]
            V[:, k] -= R[j, k] * Q[:, j]
    return Q, R


# (1) Matriks topi
G = Xt.T @ Xt
H = Xt @ np.linalg.solve(G, Xt.T)
print("(1) 4 H =")
print(np.round(4 * H).astype(int))
print("    H^2 = H:", np.allclose(H @ H, H), "  H^T = H:",
      np.allclose(H, H.T), "  trace H = %.4f" % np.trace(H))
print("    H y =", H @ y)

# (2) QR data mini
Q, R = gs_klasik(Xt)
print("(2) sqrt(10) * Q =")
print(np.round(np.sqrt(10) * Q, 4))
print("    R =")
print(R)
print("    Q^T Q = I:", np.allclose(Q.T @ Q, np.eye(3)))
Qn, Rn = np.linalg.qr(Xt)
print("    np.linalg.qr sama sampai tanda:",
      np.allclose(np.abs(Qn), np.abs(Q)), np.allclose(np.abs(Rn),
                                                      np.abs(R)))

# (3) Kuadrat terkecil lewat QR
c = Q.T @ y
print("(3) Q^T y * sqrt(10) =", c * np.sqrt(10))
print("    w dari R w = Q^T y:", np.linalg.solve(R, c))
