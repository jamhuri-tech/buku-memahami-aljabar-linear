"""Data mini yang dipakai di seluruh buku.

Empat mahasiswa, dua fitur dan satu target: x1 = jam belajar mandiri
per hari, x2 = jam belajar kelompok per hari, y = skor kuis (0-20).
Kolom x2 adalah kolom x1 dengan sampel 2 dan 3 bertukar, sehingga kedua
fitur mempunyai rata-rata dan ragam yang sama.
"""
import numpy as np

BENIH = 20261010

DATA_MINI = np.array([[1, 1, 3],
                      [2, 4, 10],
                      [4, 2, 12],
                      [5, 5, 15]], dtype=float)


def data_mini():
    """Mengembalikan (X, y): X berukuran 4 x 2, y berukuran 4."""
    return DATA_MINI[:, :2].copy(), DATA_MINI[:, 2].copy()


def rancangan(X):
    """Matriks rancangan [1 | X]: kolom satu di depan, untuk w0."""
    return np.column_stack([np.ones(len(X)), X])
