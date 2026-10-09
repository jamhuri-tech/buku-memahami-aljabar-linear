"""Bab 4: eliminasi Gauss, dekomposisi LU, substitusi maju-mundur,
dan pivot parsial, dicocokkan dengan NumPy dan SciPy."""
import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve

from bab01_data import data_mini, rancangan

np.set_printoptions(precision=4, suppress=True)
X, y = data_mini()
Xt = rancangan(X)
A = Xt.T @ Xt
b = Xt.T @ y


def eliminasi(A, b, pivot=True):
    """Eliminasi Gauss lalu substitusi mundur. Mengembalikan x."""
    M = np.column_stack([A, b]).astype(float)
    n = len(b)
    for k in range(n - 1):
        if pivot:
            p = k + np.argmax(np.abs(M[k:, k]))
            M[[k, p]] = M[[p, k]]
        for i in range(k + 1, n):
            M[i] -= M[i, k] / M[k, k] * M[k]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, n] - M[i, i + 1:n] @ x[i + 1:]) / M[i, i]
    return x


def lu_tanpa_pivot(A):
    """A = L U tanpa menukar baris (Doolittle)."""
    n = len(A)
    L, U = np.eye(n), A.astype(float).copy()
    for k in range(n - 1):
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i] -= L[i, k] * U[k]
    return L, U


# (1) Persamaan normal data mini
print("(1) eliminasi buatan sendiri:", eliminasi(A, b, pivot=False))
print("    np.linalg.solve          :", np.linalg.solve(A, b))

# (2) LU tanpa pivot dan LU SciPy (dengan pivot)
L, U = lu_tanpa_pivot(A)
print("(2) L =")
print(L)
print("    U =")
print(U)

# (3) Satu faktorisasi, banyak ruas kanan
faktor = lu_factor(A)
B = np.column_stack([b, Xt.T @ (2 * y), Xt.T @ (y + 1)])
print("(3) tiga ruas kanan, satu LU:")
print(lu_solve(faktor, B).T)

# (4) Pivot kecil
eps = 1e-20
A4 = np.array([[eps, 1.0], [1.0, 1.0]])
b4 = np.array([1.0, 2.0])
print("(4) tanpa pivot :", eliminasi(A4, b4, pivot=False))
print("    dengan pivot:", eliminasi(A4, b4, pivot=True))
print("    np.linalg   :", np.linalg.solve(A4, b4))

# (5) LU SciPy memakai pivot parsial
P, L2, U2 = lu(A)
print("(5) SciPy menukar baris, P =")
print(P.astype(int))
print("    L U = A:", np.allclose(L @ U, A),
      "  P L2 U2 = A:", np.allclose(P @ L2 @ U2, A))
print("    solusi:", lu_solve(lu_factor(A), b))
