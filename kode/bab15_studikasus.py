"""Bab 15: studi kasus digit tulisan tangan (load_digits). Matriks
data, kemiripan kosinus, SVD dan PCA, klasifikasi kuadrat terkecil
satu-lawan-semua, ridge, dan bilangan kondisi."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from bab01_data import BENIH

np.set_printoptions(precision=3, suppress=True)
d = load_digits()
X, y = d.data, d.target
m, n = X.shape

# (1) Data sebagai matriks
nol = np.where(X.std(axis=0) == 0)[0]
print("(1) X: %d x %d, kelas %s" % (m, n, np.unique(y)))
print("    citra 8 x 8 -> baris 64: piksel (r, c) di kolom 8 r + c")
print("    kolom selalu nol:", nol.tolist())
print("    rank X = %d, rank [1 | X] = %d"
      % (np.linalg.matrix_rank(X),
         np.linalg.matrix_rank(np.column_stack([np.ones(m), X]))))

Xl, Xu, yl, yu = train_test_split(X, y, test_size=0.3, stratify=y,
                                  random_state=BENIH)
print("    latih %d, uji %d" % (len(yl), len(yu)))

# (2) Kemiripan kosinus antar rata-rata kelas
M = np.array([Xl[yl == k].mean(0) for k in range(10)])
Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
C = Mn @ Mn.T
np.fill_diagonal(C, 0)
i, j = np.unravel_index(np.argmax(C), C.shape)
print("(2) rata-rata kelas paling mirip: %d dan %d, cos = %.3f"
      % (i, j, C[i, j]))
pred_cos = np.argmax((Xu / np.linalg.norm(Xu, axis=1, keepdims=True))
                     @ Mn.T, axis=1)
print("    klasifikasi rata-rata terdekat (kosinus): akurasi %.4f"
      % np.mean(pred_cos == yu))

# (3) SVD dan PCA
mu = Xl.mean(0)
U, s, Vt = np.linalg.svd(Xl - mu, full_matrices=False)
proporsi = np.cumsum(s ** 2) / np.sum(s ** 2)
print("(3) tiga nilai singular terbesar:", s[:3])
for k in (2, 10, 20, 40):
    print("    %2d komponen: proporsi varians %.3f" % (k, proporsi[k - 1]))
print("    komponen untuk 90%%: %d" % (np.argmax(proporsi >= 0.9) + 1))


# (4) Klasifikasi kuadrat terkecil satu-lawan-semua
def satu_panas(t):
    Y = -np.ones((len(t), 10))
    Y[np.arange(len(t)), t] = 1
    return Y


def latih(A, Y, lam=0.0):
    """Ridge pada kolom terpusat; lam = 0 memberi norma minimum."""
    a, b = A.mean(0), Y.mean(0)
    Ac = A - a
    if lam == 0:
        W = np.linalg.lstsq(Ac, Y - b, rcond=None)[0]
    else:
        W = np.linalg.solve(Ac.T @ Ac + lam * np.eye(A.shape[1]),
                            Ac.T @ (Y - b))
    return a, b, W


def akurasi(model, A, t):
    a, b, W = model
    return np.mean(np.argmax((A - a) @ W + b, axis=1) == t)


Yl = satu_panas(yl)
print("(4) metode                         akurasi uji")
for lam in (0.0, 10.0, 100.0, 1000.0):
    mdl = latih(Xl, Yl, lam)
    print("    kuadrat terkecil, lambda %6.0f  %.4f"
          % (lam, akurasi(mdl, Xu, yu)))
for k in (10, 20, 40):
    Zl, Zu = (Xl - mu) @ Vt[:k].T, (Xu - mu) @ Vt[:k].T
    print("    PCA %2d komponen + kuadrat ter.  %.4f"
          % (k, akurasi(latih(Zl, Yl), Zu, yu)))
lr = LogisticRegression(max_iter=5000).fit(Xl, yl)
print("    regresi logistik (sklearn)      %.4f" % lr.score(Xu, yu))

# (5) Bilangan kondisi
Xc = Xl - mu
hidup = Xl.std(0) > 0
print("(5) kolom yang selalu nol di data latih:", np.where(~hidup)[0].tolist())
print("    rank Xc = %d dari %d kolom -> Xc^T Xc singular"
      % (np.linalg.matrix_rank(Xc), n))
s2 = np.linalg.svd(Xc[:, hidup], compute_uv=False)
print("    sesudah kolom nol dibuang: kondisi Xc = %.1f" % (s2[0] / s2[-1]))
print("                               kondisi Xc^T Xc = %.3g"
      % ((s2[0] / s2[-1]) ** 2))
print("    kondisi Xc^T Xc + 100 I = %.1f"
      % ((s2[0] ** 2 + 100) / (s2[-1] ** 2 + 100)))
