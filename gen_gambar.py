# -*- coding: utf-8 -*-
"""Membangkitkan seluruh gambar Matplotlib ke gbr/ dalam dua bentuk:
PDF vektor untuk cetak dan PNG 300 dpi untuk EPUB.

Satu fungsi per gambar, dinamai babNN_nama(), yang memanggil
simpan(fig, "babNN-nama"). Fungsi bernama babNN_* dijalankan otomatis.
Seed selalu tetap, supaya gambar tidak berubah setiap build.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 20261010  # sama dengan seluruh kode/bab*.py
GBR = Path("gbr")

# Sebagian gambar memakai kelas yang sudah ditulis di kode/, supaya
# logikanya tidak terduplikasi di dua tempat.
sys.path.insert(0, str(Path(__file__).parent / "kode"))

# Warna mengikuti preamble.tex.
BIRU = "#1B3B6F"
HIJAU = "#1E6F5C"
JINGGA = "#B85C00"
MERAH = "#9B1B30"
ABU = "#5A6472"
ABU_GARIS = "#C9CED6"
BIRU_MUDA = "#E8EEF7"
HIJAU_MUDA = "#E6F2EF"
JINGGA_MUDA = "#FDF0E3"
MERAH_MUDA = "#FBE9EC"

plt.rcParams.update({
    "font.size": 7.5,
    "axes.edgecolor": ABU_GARIS,
    "axes.labelcolor": ABU,
    "axes.titlesize": 8,
    "axes.titlecolor": BIRU,
    "xtick.color": ABU,
    "ytick.color": ABU,
    "text.color": ABU,
    "grid.color": ABU_GARIS,
    "legend.frameon": False,
    "figure.dpi": 300,
})


def angka(v, n=3):
    """Angka dengan koma desimal, sesuai kaidah bahasa Indonesia."""
    return f"{v:.{n}f}".replace(".", ",")


def angka_mat(v, n=3):
    """Seperti angka(), untuk mode matematika: koma tanpa spasi."""
    return angka(v, n).replace(",", "{,}")


def _koma(fig):
    """Mengubah pemisah desimal pada label sumbu menjadi koma.

    Sumbu berskala logaritmik dilewati, karena labelnya berupa pangkat
    sepuluh dan tidak memuat pemisah desimal.
    """
    from matplotlib.ticker import FuncFormatter
    rapi = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    for ax in fig.axes:
        kunci = getattr(ax, "_label_terkunci", set())
        if "x" not in kunci and ax.get_xscale() == "linear":
            ax.xaxis.set_major_formatter(rapi)
        if "y" not in kunci and ax.get_yscale() == "linear":
            ax.yaxis.set_major_formatter(rapi)


def kunci_label(ax, *sumbu):
    """Menandai sumbu yang labelnya kita tetapkan sendiri."""
    ax._label_terkunci = getattr(ax, "_label_terkunci",
                                 set()) | set(sumbu)


def simpan(fig, nama):
    _koma(fig)
    GBR.mkdir(exist_ok=True)
    fig.savefig(GBR / f"{nama}.pdf", bbox_inches="tight")
    fig.savefig(GBR / f"{nama}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("gbr/" + nama)


def _rapikan(ax):
    """Gaya sumbu seri: tanpa bingkai atas dan kanan."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------
#  Gambar per bab ditambahkan di bawah ini sebagai fungsi babNN_nama().





# ============================ Bab 1 ==================================

def _mini():
    from bab01_data import data_mini, rancangan
    X, y = data_mini()
    return X, y, rancangan(X)


def bab01_data():
    X, y, _ = _mini()
    fig, ax = plt.subplots(1, 2, figsize=(4.6, 2.2))
    a = ax[0]
    a.scatter(X[:, 0], X[:, 1], s=28, color=BIRU, zorder=3)
    for i, (u, v) in enumerate(X):
        a.annotate(f"$i={i + 1}$\n$y={y[i]:g}$", (u, v),
                   textcoords="offset points", xytext=(6, -4),
                   fontsize=6.5, color=ABU)
    a.set_xlim(0, 7.2)
    a.set_ylim(0, 6.5)
    a.set_xlabel("$x_1$ (jam mandiri)")
    a.set_ylabel("$x_2$ (jam kelompok)")
    a.set_title("baris: sampel sebagai titik")
    _rapikan(a)
    a = ax[1]
    lebar = 0.25
    idx = np.arange(4)
    for j, (kol, w) in enumerate([(X[:, 0], BIRU), (X[:, 1], HIJAU),
                                  (y / 3, JINGGA)]):
        a.bar(idx + (j - 1) * lebar, kol, lebar, color=w,
              label=["$x_1$", "$x_2$", "$y/3$"][j])
    a.set_xticks(idx, [f"{i + 1}" for i in idx])
    kunci_label(a, "x")
    a.set_xlabel("sampel $i$")
    a.set_title("kolom: fitur sebagai vektor")
    a.legend(fontsize=6, loc="upper left")
    _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab01-data")


def bab01_turun():
    X, y, Xt = _mini()
    m = len(y)
    w = np.zeros(3)
    W, L = [w.copy()], [np.mean((y - Xt @ w) ** 2)]
    for _ in range(3000):
        w = w + 0.02 * 2 / m * Xt.T @ (y - Xt @ w)
        W.append(w.copy())
        L.append(np.mean((y - Xt @ w) ** 2))
    W, L = np.array(W), np.array(L)
    t = np.arange(len(L))
    fig, ax = plt.subplots(1, 2, figsize=(4.6, 2.0))
    a = ax[0]
    a.semilogy(t[1:1600], L[1:1600] - 1, color=BIRU)
    a.set_xlabel("iterasi $t$")
    a.set_ylabel("$L(\\mathbf{w}^{(t)}) - 1$")
    a.set_title("loss menuju minimum 1")
    _rapikan(a)
    a = ax[1]
    for j, (c, nama) in enumerate([(ABU, "$w_0$"), (BIRU, "$w_1$"),
                                   (HIJAU, "$w_2$")]):
        a.plot(t, W[:, j], color=c, label=nama)
    a.set_xscale("symlog", linthresh=1)
    a.set_xlabel("iterasi $t$")
    a.set_title("bobot menuju $(1, 2, 1)$")
    a.legend(fontsize=6)
    _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab01-turun")


# ============================ Bab 2 ==================================

def bab02_operasi():
    u, v, w = np.array([3, 1]), np.array([1, 2]), np.array([-1, 3])
    fig, ax = plt.subplots(1, 2, figsize=(4.6, 2.3))

    def panah(a, p, q, c, teks, dx=0.1, dy=0.1, lw=1.1):
        a.annotate("", xy=q, xytext=p,
                   arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                                   shrinkA=0, shrinkB=0,
                                   mutation_scale=7))
        a.text(q[0] + dx, q[1] + dy, teks, color=c, fontsize=7)

    a = ax[0]
    panah(a, (0, 0), u, BIRU, "$\\mathbf{u} = (3\\;\\;1)^\\top$", -0.2, -0.45)
    panah(a, (0, 0), v, HIJAU, "$\\mathbf{v} = (1\\;\\;2)^\\top$", -1.0, 0.15)
    panah(a, u, u + v, HIJAU, "", lw=0.6)
    panah(a, v, u + v, BIRU, "", lw=0.6)
    panah(a, (0, 0), u + v, JINGGA,
          "$\\mathbf{u} + \\mathbf{v} = (4\\;\\;3)^\\top$", -1.2, 0.2)
    panah(a, (0, 0), 2 * v, ABU, "$2\\mathbf{v}$", 0.1, 0.0, lw=0.6)
    a.set_title("penjumlahan dan perkalian skalar")
    a.set_xlim(-0.3, 5)
    a.set_ylim(-0.3, 4.5)
    a = ax[1]
    panah(a, (0, 0), u, BIRU, "$\\mathbf{u}$", 0.1, -0.35)
    panah(a, (0, 0), v, HIJAU, "$\\mathbf{v}$", 0.05, 0.1)
    panah(a, (0, 0), w, MERAH, "$\\mathbf{w} = (-1\\;\\;3)^\\top$", -0.9, 0.15)
    t = np.linspace(np.arctan2(1, 3), np.arctan2(2, 1), 30)
    a.plot(0.8 * np.cos(t), 0.8 * np.sin(t), color=ABU, lw=0.7)
    a.text(0.9, 0.75, "$45^\\circ$", fontsize=6.5, color=ABU)
    q = 0.35 * u / np.linalg.norm(u)
    r = 0.35 * w / np.linalg.norm(w)
    a.plot([q[0], q[0] + r[0], r[0]], [q[1], q[1] + r[1], r[1]],
           color=ABU, lw=0.6)
    a.set_title("sudut dan tegak lurus")
    a.set_xlim(-1.6, 3.6)
    a.set_ylim(-0.3, 3.6)
    for a in ax:
        a.axhline(0, color=ABU_GARIS, lw=0.5)
        a.axvline(0, color=ABU_GARIS, lw=0.5)
        a.set_aspect("equal")
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab02-operasi")


def bab02_bola():
    t = np.linspace(0, 2 * np.pi, 400)
    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    ax.plot(np.cos(t), np.sin(t), color=BIRU, label="$L_2$")
    ax.plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], color=HIJAU,
            label="$L_1$")
    ax.plot([1, -1, -1, 1, 1], [1, 1, -1, -1, 1], color=JINGGA,
            label="$L_\\infty$")
    v = np.array([-2, 2])
    for nrm, c in [(4, HIJAU), (np.sqrt(8), BIRU), (2, JINGGA)]:
        ax.plot(*(v / nrm), "o", color=c, ms=4)
    ax.annotate("", xy=v / 2, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=ABU, lw=0.8))
    ax.text(-1.05, 1.08, "$\\mathbf{v}/\\|\\mathbf{v}\\|_\\infty$",
            fontsize=6.5, color=JINGGA)
    ax.axhline(0, color=ABU_GARIS, lw=0.5)
    ax.axvline(0, color=ABU_GARIS, lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.3, 1.4)
    ax.legend(fontsize=6, loc="lower right")
    _rapikan(ax)
    simpan(fig, "bab02-bola")


def bab02_jarak():
    from bab01_data import data_mini, SEED
    X, _ = data_mini()
    fig, ax = plt.subplots(1, 2, figsize=(4.6, 2.1))
    a = ax[0]
    for i in range(4):
        for j in range(i + 1, 4):
            d2 = int(((X[i] - X[j]) ** 2).sum())
            a.plot(*X[[i, j]].T, color=ABU_GARIS, lw=0.8, zorder=1)
            f = {(0, 3): 0.25, (1, 2): 0.3}.get((i, j), 0.5)
            u, v = (1 - f) * X[i] + f * X[j]
            a.text(u, v, str(d2), fontsize=6, color=JINGGA,
                   ha="center", va="center",
                   bbox=dict(fc="white", ec="none", pad=0.4))
    a.scatter(*X.T, color=BIRU, s=22, zorder=3)
    for i, (u, v) in enumerate(X):
        a.text(u - 0.15, v + 0.25, f"{i + 1}", fontsize=6.5, color=BIRU)
    a.set_xlim(0, 6)
    a.set_ylim(0, 6)
    a.set_aspect("equal")
    a.set_xlabel("$x_1$")
    a.set_ylabel("$x_2$")
    a.set_title("jarak Euclid kuadrat")
    _rapikan(a)
    a = ax[1]
    rng = np.random.default_rng(SEED)
    for d, c in [(2, ABU), (100, HIJAU), (10000, BIRU)]:
        A = rng.standard_normal((2000, d))
        B = rng.standard_normal((2000, d))
        cs = (A * B).sum(1) / np.linalg.norm(A, axis=1) \
            / np.linalg.norm(B, axis=1)
        a.hist(cs, bins=np.linspace(-1, 1, 81), histtype="step",
               color=c, density=True, label=f"$d = {d}$")
    a.set_xlabel("kosinus dua vektor acak")
    a.set_yticks([])
    a.legend(fontsize=6)
    a.set_title("kosinus memusat di 0")
    _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab02-jarak")


# ============================ Bab 3 ==================================

def _matriks_kecil(ax, M, judul, vmax):
    from matplotlib.colors import LinearSegmentedColormap
    peta = LinearSegmentedColormap.from_list(
        "bj", [JINGGA, "white", BIRU])
    ax.imshow(M, cmap=peta, vmin=-vmax, vmax=vmax)
    for (r, c), v in np.ndenumerate(M):
        ax.text(c, r, f"{v:g}", ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 0.6 * vmax else ABU)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(ABU_GARIS)
    ax.set_title(judul, fontsize=6.5)


def bab03_luar():
    X, _, _ = _mini()
    Xc = X - X.mean(0)
    fig, ax = plt.subplots(1, 5, figsize=(4.6, 1.25))
    for i in range(4):
        _matriks_kecil(ax[i], np.outer(Xc[i], Xc[i]),
                       f"$i = {i + 1}$", 10)
    _matriks_kecil(ax[4], Xc.T @ Xc,
                   "$\\mathbf{X}_c^{\\top}\\mathbf{X}_c$", 10)
    fig.subplots_adjust(wspace=0.45)
    for k, t in enumerate(["+", "+", "+", "="]):
        b0, b1 = ax[k].get_position(), ax[k + 1].get_position()
        fig.text((b0.x1 + b1.x0) / 2, (b0.y0 + b0.y1) / 2, t,
                 fontsize=10, color=ABU, ha="center", va="center")
    simpan(fig, "bab03-luar")


def bab03_ubah():
    X, y, _ = _mini()
    W = np.array([[1, 1], [1, -1]])
    F = X @ W
    fig, ax = plt.subplots(1, 2, figsize=(4.6, 2.1))
    for a, P, (lx, ly), judul in [
            (ax[0], X, ("$x_1$", "$x_2$"), "fitur asli"),
            (ax[1], F, ("$t = x_1 + x_2$", "$d = x_1 - x_2$"),
             "fitur baru $\\mathbf{X}\\mathbf{W}$")]:
        a.scatter(*P.T, color=BIRU, s=22, zorder=3)
        for i, (u, v) in enumerate(P):
            a.text(u + 0.25, v + 0.2, f"{i + 1}", fontsize=6.5,
                   color=BIRU)
        a.axhline(0, color=ABU_GARIS, lw=0.5)
        a.axvline(0, color=ABU_GARIS, lw=0.5)
        a.set_xlabel(lx)
        a.set_ylabel(ly)
        a.set_title(judul)
        a.set_aspect("equal")
        _rapikan(a)
    ax[0].set_xlim(-0.5, 6)
    ax[0].set_ylim(-0.5, 6)
    ax[1].set_xlim(-0.5, 11)
    ax[1].set_ylim(-3, 3)
    fig.tight_layout()
    simpan(fig, "bab03-ubah")


# ============================ Bab 4 ==================================

def _panah(a, p, q, c, teks="", dx=0.1, dy=0.1, lw=1.1):
    a.annotate("", xy=q, xytext=p,
               arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                               shrinkA=0, shrinkB=0, mutation_scale=7))
    if teks:
        a.text(q[0] + dx, q[1] + dy, teks, color=c, fontsize=7)


def bab04_gambar():
    fig, ax = plt.subplots(1, 2, figsize=(4.6, 2.3))
    a = ax[0]
    t = np.linspace(-0.5, 4, 50)
    a.plot(t, 4 - 2 * t, color=BIRU, label="$2x_1 + x_2 = 4$")
    a.plot(t, (7 - t) / 3, color=HIJAU, label="$x_1 + 3x_2 = 7$")
    a.plot(1, 2, "o", color=JINGGA, ms=4, zorder=3)
    a.text(1.15, 2.15, "$(1, 2)$", color=JINGGA, fontsize=7)
    a.set_xlim(-0.5, 4)
    a.set_ylim(-0.5, 4)
    a.set_xlabel("$x_1$")
    a.set_ylabel("$x_2$")
    a.set_title("gambar baris: dua garis")
    a.legend(fontsize=6, loc="upper right")
    a = ax[1]
    a1, a2 = np.array([2, 1]), np.array([1, 3])
    _panah(a, (0, 0), a1, BIRU, "kolom 1", 0.05, -0.5)
    _panah(a, a1, a1 + 2 * a2, HIJAU, "", lw=0.8)
    _panah(a, (0, 0), a2, HIJAU, "kolom 2", -1.3, 0.1)
    _panah(a, (0, 0), a1 + 2 * a2, JINGGA, "$\\mathbf{b}$", 0.1, 0.0)
    a.text(2.6, 4.0, "$2 \\times$ kolom 2", color=HIJAU, fontsize=6.5)
    a.set_xlim(-1.5, 5)
    a.set_ylim(-0.5, 7.5)
    a.set_title("gambar kolom: kombinasi kolom")
    for a in ax:
        a.axhline(0, color=ABU_GARIS, lw=0.5)
        a.axvline(0, color=ABU_GARIS, lw=0.5)
        a.set_aspect("equal")
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab04-gambar")


def bab04_jenis():
    fig, ax = plt.subplots(1, 3, figsize=(4.6, 1.7))
    t = np.linspace(-0.5, 3, 20)
    kasus = [("tepat satu", [(2, 1, 4), (1, 3, 7)]),
             ("tidak ada", [(1, 1, 2), (1, 1, 3)]),
             ("tak hingga", [(1, 1, 2), (2, 2, 4)])]
    for a, (judul, garis) in zip(ax, kasus):
        for (p, q, r), c, gaya in zip(garis, [BIRU, HIJAU], ["-", "--"]):
            a.plot(t, (r - p * t) / q, color=c, ls=gaya, lw=1.2)
        if judul == "tepat satu":
            a.plot(1, 2, "o", color=JINGGA, ms=3.5)
        a.set_xlim(-0.5, 3)
        a.set_ylim(-0.5, 3.5)
        a.set_xticks([0, 1, 2, 3])
        a.set_yticks([0, 1, 2, 3])
        a.set_title(judul + " solusi")
        a.set_aspect("equal")
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab04-jenis")


# ============================ Bab 5 ==================================

def bab05_rentang():
    u, v = np.array([3, 1]), np.array([1, 2])
    fig, ax = plt.subplots(1, 2, figsize=(4.6, 2.3))
    a = ax[0]
    t = np.linspace(-1.6, 1.6, 2)
    a.plot(t * u[0], t * u[1], color=BIRU, lw=1.0)
    for c in np.arange(-1.5, 1.51, 0.5):
        a.plot(c * u[0], c * u[1], "o", color=BIRU, ms=2.5)
    _panah(a, (0, 0), u, JINGGA, "$\\mathbf{u}$", 0.1, -0.5)
    a.set_title("rentang satu vektor: garis")
    a = ax[1]
    for c1 in np.arange(-1.5, 1.51, 0.5):
        for c2 in np.arange(-1.5, 1.51, 0.5):
            q = c1 * u + c2 * v
            a.plot(*q, "o", color=ABU_GARIS, ms=2.0)
    for c in np.arange(-1.5, 1.51, 0.5):
        a.plot(*np.array([c * u - 1.5 * v, c * u + 1.5 * v]).T,
               color=ABU_GARIS, lw=0.4)
        a.plot(*np.array([c * v - 1.5 * u, c * v + 1.5 * u]).T,
               color=ABU_GARIS, lw=0.4)
    _panah(a, (0, 0), u, BIRU, "$\\mathbf{u}$", 0.1, -0.5)
    _panah(a, (0, 0), v, HIJAU, "$\\mathbf{v}$", -0.6, 0.1)
    a.set_title("rentang dua vektor: bidang")
    for a in ax:
        a.axhline(0, color=ABU_GARIS, lw=0.5)
        a.axvline(0, color=ABU_GARIS, lw=0.5)
        a.set_xlim(-5, 5)
        a.set_ylim(-4, 4)
        a.set_aspect("equal")
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab05-rentang")


def bab05_lembah():
    X, y, _ = _mini()
    X3 = np.column_stack([np.ones(4), X, X.sum(1)])
    w = np.array([1.0, 2, 1, 0])
    n = np.array([0.0, 1, 1, -1])
    d = np.array([0.0, 1, 0, 0])
    t = np.linspace(-2, 2, 200)
    L = lambda v: np.mean((y - X3 @ v) ** 2)
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    ax.plot(t, [L(w + s * n) for s in t], color=BIRU,
            label="arah ruang nol $(0\\;\\;1\\;\\;1\\;\\;-1)^\\top$")
    ax.plot(t, [L(w + s * d) for s in t], color=JINGGA,
            label="arah $w_1$ saja $(0\\;\\;1\\;\\;0\\;\\;0)^\\top$")
    for s, c in [(0, ABU), (-1, HIJAU)]:
        ax.plot(s, 1, "o", color=c, ms=3.5, zorder=3)
    ax.text(-1.05, 1.9, "norma\nminimum", color=HIJAU, fontsize=6,
            ha="center")
    ax.set_xlabel("langkah $t$ dari $\\mathbf{w} = (1\\;\\;2\\;\\;1\\;\\;0)^\\top$")
    ax.set_ylabel("loss $L$")
    ax.set_ylim(0, 12)
    ax.legend(fontsize=6, loc="center right")
    _rapikan(ax)
    simpan(fig, "bab05-lembah")


# ============================ Bab 6 ==================================

def bab06_luas():
    from matplotlib.patches import Polygon
    persegi = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    kasus = [("persegi satuan, luas 1", np.eye(2), BIRU),
             ("$\\mathbf{A}$: luas $|\\det\\mathbf{A}| = 5$",
              np.array([[2, 1], [1, 3]]), HIJAU),
             ("$\\mathbf{S}$: luas $|\\det\\mathbf{S}| = 0$",
              np.array([[1, 2], [2, 4]]), MERAH)]
    fig, ax = plt.subplots(1, 3, figsize=(4.6, 1.9))
    for a, (judul, M, c) in zip(ax, kasus):
        Q = persegi @ M.T
        a.add_patch(Polygon(Q, closed=True, fc=c, alpha=0.18, ec=c,
                            lw=1.0))
        _panah(a, (0, 0), M[:, 0], c, "", lw=1.0)
        _panah(a, (0, 0), M[:, 1], c, "", lw=1.0)
        a.set_xlim(-0.5, 4.5)
        a.set_ylim(-0.5, 6.5)
        a.set_aspect("equal")
        a.set_title(judul, fontsize=6.5)
        a.axhline(0, color=ABU_GARIS, lw=0.5)
        a.axvline(0, color=ABU_GARIS, lw=0.5)
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab06-luas")


# ============================ Bab 7 ==================================

def bab07_proyeksi():
    a, b = np.array([3.0, 1.0]), np.array([1.0, 2.0])
    pr = (a @ b) / (a @ a) * a
    fig, ax = plt.subplots(figsize=(2.8, 2.0))
    t = np.linspace(-0.3, 1.25, 2)
    ax.plot(t * a[0], t * a[1], color=ABU_GARIS, lw=0.8)
    _panah(ax, (0, 0), a, BIRU, "$\\mathbf{a}$", 0.05, -0.3)
    _panah(ax, (0, 0), b, HIJAU, "$\\mathbf{b}$", -0.3, 0.05)
    _panah(ax, (0, 0), pr, JINGGA, "$\\mathbf{p}$", -0.1, -0.35, lw=1.4)
    ax.plot(*np.array([pr, b]).T, color=MERAH, lw=0.9, ls="--")
    ax.text(1.3, 1.25, "$\\mathbf{b} - \\mathbf{p}$", color=MERAH,
            fontsize=7)
    u = a / np.linalg.norm(a) * 0.18
    w = (b - pr) / np.linalg.norm(b - pr) * 0.18
    q = pr
    ax.plot([q[0] - u[0], q[0] - u[0] + w[0], q[0] + w[0]],
            [q[1] - u[1], q[1] - u[1] + w[1], q[1] + w[1]],
            color=ABU, lw=0.6)
    ax.set_xlim(-0.3, 3.6)
    ax.set_ylim(-0.3, 2.4)
    ax.set_aspect("equal")
    ax.axhline(0, color=ABU_GARIS, lw=0.5)
    ax.axvline(0, color=ABU_GARIS, lw=0.5)
    _rapikan(ax)
    simpan(fig, "bab07-proyeksi")


def bab07_ortogonal():
    sys.path.insert(0, str(Path(__file__).parent / "kode"))
    from bab07_proyeksi import gs_klasik, gs_modifikasi
    t = np.linspace(0, 1, 50)
    kond, hasil = [], {"klasik": [], "modifikasi": [], "Householder": []}
    for d in range(2, 15):
        A = np.vander(t, d, increasing=True)
        kond.append(np.linalg.cond(A))
        for nama, f in [("klasik", gs_klasik),
                        ("modifikasi", gs_modifikasi),
                        ("Householder", np.linalg.qr)]:
            Q = f(A)[0]
            hasil[nama].append(np.linalg.norm(Q.T @ Q - np.eye(d)))
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    for (nama, v), c in zip(hasil.items(), [MERAH, JINGGA, BIRU]):
        ax.loglog(kond, np.maximum(v, 1e-17), "o-", color=c, ms=2.5,
                  lw=0.9, label=("Gram\u2013Schmidt " + nama
                                 if nama != "Householder"
                                 else "Householder (np.linalg.qr)"))
    ax.set_xlabel("bilangan kondisi $\\mathbf{A}$")
    ax.set_ylabel("$\\|\\mathbf{Q}^{\\top}\\mathbf{Q} - \\mathbf{I}\\|$")
    ax.legend(fontsize=6)
    _rapikan(ax)
    simpan(fig, "bab07-ortogonal")


# ============================ Bab 8 ==================================

def bab08_cocok():
    X, y, Xt = _mini()
    w = np.linalg.solve(Xt.T @ Xt, Xt.T @ y)
    yh = Xt @ w
    fig, ax = plt.subplots(1, 2, figsize=(4.6, 2.1))
    a = ax[0]
    a.plot([2, 17], [2, 17], color=ABU_GARIS, lw=0.8)
    for u, v in zip(yh, y):
        a.plot([u, u], [u, v], color=MERAH, lw=0.9)
    a.scatter(yh, y, color=BIRU, s=20, zorder=3)
    a.axhline(10, color=ABU, lw=0.5, ls=":")
    a.text(2.3, 10.4, "$\\bar{y} = 10$", fontsize=6.5, color=ABU)
    a.set_xlabel("ramalan $\\hat{y}$")
    a.set_ylabel("target $y$")
    a.set_title("galat: garis merah")
    a.set_aspect("equal")
    _rapikan(a)
    a = ax[1]
    bag = [78, 74, 4]
    a.bar([0, 1, 2], bag, color=[ABU, BIRU, MERAH], width=0.6)
    for i, v in enumerate(bag):
        a.text(i, v + 1.5, str(v), ha="center", fontsize=7, color=ABU)
    a.set_xticks([0, 1, 2], ["SST", "SSR", "SSE"])
    kunci_label(a, "x")
    a.set_ylim(0, 90)
    a.set_title("SST = SSR + SSE")
    _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab08-cocok")


def bab08_ridge():
    X, y, _ = _mini()
    Xc, yc = X - X.mean(0), y - y.mean()
    lam = np.logspace(-2, 3, 200)
    W = np.array([np.linalg.solve(Xc.T @ Xc + l * np.eye(2), Xc.T @ yc)
                  for l in lam])
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    ax.semilogx(lam, W[:, 0], color=BIRU, label="$w_1$")
    ax.semilogx(lam, W[:, 1], color=HIJAU, label="$w_2$")
    ax.axvline(2, color=ABU_GARIS, lw=0.6, ls="--")
    ax.text(2.3, 2.05, "$\\lambda = 2$", fontsize=6.5, color=ABU)
    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("bobot ridge")
    ax.legend(fontsize=6.5)
    _rapikan(ax)
    simpan(fig, "bab08-ridge")


# ============================ Bab 9 ==================================

def bab09_elips():
    S = np.array([[10.0, 6.0], [6.0, 10.0]]) / 4
    t = np.linspace(0, 2 * np.pi, 300)
    C = np.vstack([np.cos(t), np.sin(t)])
    E = S @ C
    fig, ax = plt.subplots(figsize=(2.8, 2.6))
    ax.plot(*C, color=ABU_GARIS, lw=0.8)
    ax.plot(*E, color=BIRU, lw=1.0)
    v1 = np.array([1, 1]) / np.sqrt(2)
    v2 = np.array([1, -1]) / np.sqrt(2)
    _panah(ax, (0, 0), v1, HIJAU, "", lw=1.0)
    _panah(ax, (0, 0), 4 * v1, HIJAU, "$4\\,\\mathbf{v}_1$", 0.1, 0.0,
           lw=0.6)
    _panah(ax, (0, 0), v2, JINGGA, "", lw=1.0)
    _panah(ax, (0, 0), 1 * v2, JINGGA, "$\\mathbf{v}_2$", 0.05, -0.35,
           lw=0.6)
    x = np.array([1.0, 0.0])
    _panah(ax, (0, 0), x, MERAH, "", lw=0.9)
    _panah(ax, (0, 0), S @ x, MERAH, "$\\frac{1}{4}\\mathbf{S}\\mathbf{u}_1$",
           0.05, -0.4, lw=0.6)
    ax.set_xlim(-3.3, 3.6)
    ax.set_ylim(-3.3, 3.6)
    ax.set_aspect("equal")
    ax.axhline(0, color=ABU_GARIS, lw=0.5)
    ax.axvline(0, color=ABU_GARIS, lw=0.5)
    _rapikan(ax)
    simpan(fig, "bab09-elips")


def bab09_konvergen():
    S = np.array([[10.0, 6.0], [6.0, 10.0]])
    P = np.array([[0.9, 0.5], [0.1, 0.5]])
    v1 = np.array([1, 1]) / np.sqrt(2)
    x = np.array([1.0, 0.0])
    k = np.arange(1, 16)
    sudut, markov = [], []
    for i in k:
        x = S @ x
        x /= np.linalg.norm(x)
        sudut.append(abs(x[0] - x[1]) / np.sqrt(2))   # sin sudut
        markov.append(np.abs(np.linalg.matrix_power(P, i)
                             - np.array([[5, 5], [1, 1]]) / 6).max())
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    ax.semilogy(k, sudut, "o-", color=BIRU, ms=2.5,
                label="iterasi pangkat: sudut ke $\\mathbf{v}_1$")
    ax.semilogy(k, 0.6 * 0.25 ** k, ":", color=BIRU, lw=0.8,
                label="$\\propto (4/16)^k$")
    ax.semilogy(k, markov, "s-", color=JINGGA, ms=2.5,
                label="Markov: galat $\\mathbf{P}^k$")
    ax.semilogy(k, 0.5 * 0.4 ** k, ":", color=JINGGA, lw=0.8,
                label="$\\propto 0{,}4^k$")
    ax.set_xlabel("langkah $k$")
    ax.legend(fontsize=5.5)
    _rapikan(ax)
    simpan(fig, "bab09-konvergen")


# ============================ Bab 10 =================================

def bab10_kuadrat():
    t = np.linspace(-2, 2, 201)
    U, V = np.meshgrid(t, t)
    kasus = [("definit positif", np.array([[10, 6], [6, 10]]) / 4),
             ("tak tentu (pelana)", np.array([[1, 2], [2, 1]])),
             ("semidefinit (lembah)", np.array([[1, 1], [1, 1]]))]
    fig, ax = plt.subplots(1, 3, figsize=(4.6, 1.75))
    for a, (judul, A) in zip(ax, kasus):
        F = A[0, 0] * U ** 2 + 2 * A[0, 1] * U * V + A[1, 1] * V ** 2
        a.contour(U, V, F, levels=[-6, -3, -1, 0, 1, 3, 6, 10],
                  colors=[JINGGA] * 3 + [ABU] + [BIRU] * 4,
                  linewidths=0.7)
        a.set_title(judul, fontsize=6.5)
        a.set_aspect("equal")
        a.set_xticks([-2, 0, 2])
        a.set_yticks([-2, 0, 2])
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab10-kuadrat")


def bab10_mahalanobis():
    X, _, _ = _mini()
    Xc = X - X.mean(0)
    C = Xc.T @ Xc / 4
    lam, Q = np.linalg.eigh(C)
    t = np.linspace(0, 2 * np.pi, 200)
    lingkar = np.vstack([np.cos(t), np.sin(t)])
    fig, ax = plt.subplots(1, 2, figsize=(4.6, 2.2))
    a = ax[0]
    for r, c in [(1, ABU_GARIS), (np.sqrt(2), HIJAU)]:
        E = Q @ np.diag(np.sqrt(lam)) @ (r * lingkar)
        a.plot(*(E + X.mean(0)[:, None]), color=c, lw=0.9)
    a.scatter(*X.T, color=BIRU, s=20, zorder=3)
    a.plot(*X.mean(0), "*", color=JINGGA, ms=6)
    a.set_title("elips Mahalanobis $d = 1$, $\\sqrt{2}$")
    a.set_xlim(-0.5, 6.5)
    a.set_ylim(-0.5, 6.5)
    a = ax[1]
    Z = Xc @ Q / np.sqrt(lam)
    a.plot(*(np.sqrt(2) * lingkar), color=HIJAU, lw=0.9)
    a.plot(*lingkar, color=ABU_GARIS, lw=0.9)
    a.scatter(*Z.T, color=BIRU, s=20, zorder=3)
    a.set_title("sesudah whitening")
    a.set_xlim(-2.2, 2.2)
    a.set_ylim(-2.2, 2.2)
    for a in ax:
        a.set_aspect("equal")
        a.axhline(0 if a is ax[1] else 3, color=ABU_GARIS, lw=0.4)
        a.axvline(0 if a is ax[1] else 3, color=ABU_GARIS, lw=0.4)
        _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab10-mahalanobis")


# ============================ Bab 11 =================================

def bab11_geometri():
    M = np.array([[3.0, 0.0], [4.0, 5.0]])
    U, sv, Vt = np.linalg.svd(M)
    if U[0, 0] < 0:
        U[:, 0] *= -1
        Vt[0] *= -1
    t = np.linspace(0, 2 * np.pi, 300)
    C = np.vstack([np.cos(t), np.sin(t)])
    tahap = [("$\\mathbf{x}$", np.eye(2)),
             ("$\\mathbf{V}^{\\top}\\mathbf{x}$", Vt),
             ("$\\Sigma\\mathbf{V}^{\\top}\\mathbf{x}$", np.diag(sv) @ Vt),
             ("$\\mathbf{U}\\Sigma\\mathbf{V}^{\\top}\\mathbf{x} = "
              "\\mathbf{M}\\mathbf{x}$", M)]
    fig, ax = plt.subplots(1, 4, figsize=(4.6, 1.5))
    for a, (judul, T) in zip(ax, tahap):
        E = T @ C
        a.plot(*E, color=BIRU, lw=0.9)
        for k, c in [(0, HIJAU), (1, JINGGA)]:
            p = T @ Vt[k]
            _panah(a, (0, 0), p, c, "", lw=0.9)
        a.set_xlim(-7.5, 7.5)
        a.set_ylim(-7.5, 7.5)
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(judul, fontsize=6)
        for sp in a.spines.values():
            sp.set_color(ABU_GARIS)
    fig.tight_layout()
    simpan(fig, "bab11-geometri")


def bab11_susut():
    lam = np.logspace(-1, 3, 200)
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    for sv, c in [(4, BIRU), (2, HIJAU)]:
        ax.semilogx(lam, sv ** 2 / (sv ** 2 + lam), color=c,
                    label=f"$\\sigma = {sv}$")
    ax.axvline(2, color=ABU_GARIS, lw=0.6, ls="--")
    ax.plot([2, 2], [16 / 18, 4 / 6], "o", color=ABU, ms=3)
    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("$\\sigma^2/(\\sigma^2 + \\lambda)$")
    ax.legend(fontsize=6.5)
    _rapikan(ax)
    simpan(fig, "bab11-susut")


# ============================ Bab 12 =================================

def _citra():
    import matplotlib.cbook as cbook
    import matplotlib.image as mimage
    jalur = cbook.get_sample_data("grace_hopper.jpg", asfileobj=False)
    return mimage.imread(jalur).astype(float).mean(axis=2)


def bab12_citra():
    G = _citra()
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    fig, ax = plt.subplots(1, 4, figsize=(4.6, 1.55))
    for a, k in zip(ax, [None, 5, 20, 50]):
        A = G if k is None else (U[:, :k] * s[:k]) @ Vt[:k]
        a.imshow(A, cmap="gray", vmin=0, vmax=255)
        a.set_xticks([])
        a.set_yticks([])
        a.set_title("asli" if k is None else f"rank {k}", fontsize=6.5)
        for sp in a.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    simpan(fig, "bab12-citra")


def bab12_singular():
    s = np.linalg.svd(_citra(), compute_uv=False)
    fig, ax = plt.subplots(1, 2, figsize=(4.6, 1.9))
    a = ax[0]
    a.semilogy(np.arange(1, len(s) + 1), s, color=BIRU, lw=0.9)
    a.set_xlabel("indeks $k$")
    a.set_ylabel("$\\sigma_k$")
    a.set_title("nilai singular citra")
    _rapikan(a)
    a = ax[1]
    e = np.cumsum(s ** 2) / (s ** 2).sum()
    a.plot(np.arange(1, 101), e[:100], color=HIJAU, lw=1.0)
    for k in (5, 20, 50):
        a.plot(k, e[k - 1], "o", color=JINGGA, ms=3)
    a.set_xlabel("rank $k$")
    a.set_ylabel("bagian energi")
    a.set_ylim(0.85, 1.0)
    a.set_title("energi yang tersimpan")
    _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab12-singular")


# ============================ Bab 13 =================================

def bab13_graf():
    fig, ax = plt.subplots(figsize=(4.6, 1.9))
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    simpul = [(0.8, "$\\mathbf{x}$", "$(2\\;\\;4)^\\top$"),
              (2.9, "$\\mathbf{z} = \\mathbf{W}_1\\mathbf{x}$",
               "$(-2\\;\\;6)^\\top$"),
              (5.0, "$\\mathbf{a} = \\mathrm{ReLU}(\\mathbf{z})$",
               "$(0\\;\\;6)^\\top$"),
              (7.1, "$\\hat{y} = \\mathbf{w}_2^\\top\\mathbf{a}$", "$6$"),
              (9.2, "$L = (\\hat{y} - y)^2$", "$16$")]
    mundur = ["", "$(0\\;\\;-8)^\\top$", "$(-8\\;\\;-8)^\\top$", "$-8$",
              "$1$"]
    for (x, nama, nilai), g in zip(simpul, mundur):
        ax.add_patch(plt.Rectangle((x - 0.85, 1.35), 1.7, 0.9,
                                   fc=BIRU_MUDA, ec=BIRU, lw=0.7))
        ax.text(x, 1.8, nama, ha="center", va="center", fontsize=6,
                color=BIRU)
        ax.text(x, 2.65, nilai, ha="center", va="center", fontsize=6,
                color=HIJAU)
        if g:
            ax.text(x, 0.9, g, ha="center", va="center", fontsize=6,
                    color=MERAH)
    for (x0, *_), (x1, *_) in zip(simpul[:-1], simpul[1:]):
        ax.annotate("", xy=(x1 - 0.87, 1.95), xytext=(x0 + 0.87, 1.95),
                    arrowprops=dict(arrowstyle="-|>", color=HIJAU,
                                    lw=0.7, mutation_scale=6))
        ax.annotate("", xy=(x0 + 0.87, 1.6), xytext=(x1 - 0.87, 1.6),
                    arrowprops=dict(arrowstyle="-|>", color=MERAH,
                                    lw=0.7, mutation_scale=6))
    ax.text(0.1, 3.3, "forward pass: nilai", color=HIJAU, fontsize=6.5)
    ax.text(0.1, 0.3, "backward pass: turunan $L$ terhadap setiap simpul",
            color=MERAH, fontsize=6.5)
    simpan(fig, "bab13-graf")


# ============================ Bab 14 =================================

def bab14_presisi():
    t = np.linspace(0, 1, 50)
    kond, galat = [], {k: [] for k in ("persamaan normal + inv",
                                         "persamaan normal + solve",
                                         "QR", "lstsq (SVD)")}
    for d in range(2, 16):
        V = np.vander(t, d, increasing=True)
        w = np.ones(d)
        yv = V @ w
        kond.append(np.linalg.cond(V))
        G, c = V.T @ V, V.T @ yv
        Q, R = np.linalg.qr(V)
        hasil = [np.linalg.inv(G) @ c, np.linalg.solve(G, c),
                 np.linalg.solve(R, Q.T @ yv),
                 np.linalg.lstsq(V, yv, rcond=None)[0]]
        for k, h in zip(galat, hasil):
            galat[k].append(max(np.linalg.norm(h - w) / np.linalg.norm(w),
                                1e-17))
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    for (k, v), c, mk in zip(galat.items(), [MERAH, JINGGA, HIJAU, BIRU],
                             ["o", "s", "^", "v"]):
        ax.loglog(kond, v, mk + "-", color=c, ms=2.5, lw=0.8, label=k)
    kk = np.array(kond)
    ax.loglog(kk, 1.1e-16 * kk, ":", color=ABU, lw=0.7)
    ax.loglog(kk, 1.1e-16 * kk ** 2, "--", color=ABU, lw=0.7)
    ax.text(kk[-4], 1.1e-16 * kk[-4] * 3, "$\\epsilon\\kappa$", fontsize=6.5,
            color=ABU)
    ax.text(kk[5], 1.1e-16 * kk[5] ** 2 * 4, "$\\epsilon\\kappa^2$",
            fontsize=6.5, color=ABU)
    ax.set_xlabel("bilangan kondisi $\\kappa(\\mathbf{V})$")
    ax.set_ylabel("galat relatif bobot")
    ax.set_ylim(1e-17, 1e2)
    ax.legend(fontsize=5.5, loc="upper left")
    _rapikan(ax)
    simpan(fig, "bab14-presisi")


# ============================ Bab 15 =================================

def _digits():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from bab01_data import SEED
    d = load_digits()
    return train_test_split(d.data, d.target, test_size=0.3,
                            stratify=d.target, random_state=SEED)


def bab15_contoh():
    Xl, Xu, yl, yu = _digits()
    fig, ax = plt.subplots(2, 10, figsize=(4.6, 1.15))
    for k in range(10):
        contoh = Xl[yl == k][0]
        rata = Xl[yl == k].mean(0)
        for a, v in [(ax[0, k], contoh), (ax[1, k], rata)]:
            a.imshow(v.reshape(8, 8), cmap="gray_r", vmin=0, vmax=16)
            a.set_xticks([])
            a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
        ax[0, k].set_title(str(k), fontsize=6.5)
    ax[0, 0].set_ylabel("contoh", fontsize=6)
    ax[1, 0].set_ylabel("rata-rata", fontsize=6)
    fig.tight_layout(pad=0.2)
    simpan(fig, "bab15-contoh")


def bab15_pca():
    Xl, Xu, yl, yu = _digits()
    mu = Xl.mean(0)
    U, s, Vt = np.linalg.svd(Xl - mu, full_matrices=False)
    Z = (Xl - mu) @ Vt[:2].T
    fig, ax = plt.subplots(1, 2, figsize=(4.6, 2.2))
    a = ax[0]
    peta = plt.get_cmap("tab10")
    for k in range(10):
        a.scatter(*Z[yl == k].T, s=2, color=peta(k), alpha=0.6)
        a.text(*np.median(Z[yl == k], 0), str(k), fontsize=7,
               fontweight="bold", color="black", ha="center",
               va="center")
    a.set_xlabel("komponen 1")
    a.set_ylabel("komponen 2")
    a.set_title("skor dua komponen utama")
    _rapikan(a)
    a = ax[1]
    e = np.cumsum(s ** 2) / np.sum(s ** 2)
    a.plot(np.arange(1, len(e) + 1), e, color=BIRU)
    a.axhline(0.9, color=ABU_GARIS, lw=0.6, ls="--")
    a.plot(21, e[20], "o", color=JINGGA, ms=3)
    a.text(24, 0.84, "21 komponen: 90%", fontsize=6.5, color=JINGGA)
    a.set_xlabel("banyak komponen $k$")
    a.set_ylabel("proporsi varians")
    a.set_title("proporsi varians kumulatif")
    _rapikan(a)
    fig.tight_layout()
    simpan(fig, "bab15-pca")


def bab15_rekonstruksi():
    Xl, Xu, yl, yu = _digits()
    mu = Xl.mean(0)
    U, s, Vt = np.linalg.svd(Xl - mu, full_matrices=False)
    x = Xu[yu == 3][0]
    ks = [0, 2, 5, 10, 20, 40, 64]
    fig, ax = plt.subplots(1, len(ks), figsize=(4.6, 0.95))
    for a, k in zip(ax, ks):
        z = (x - mu) @ Vt[:k].T
        r = x if k == 64 else mu + z @ Vt[:k]
        a.imshow(r.reshape(8, 8), cmap="gray_r", vmin=0, vmax=16)
        a.set_xticks([])
        a.set_yticks([])
        a.set_title("asli" if k == 64 else f"$k = {k}$", fontsize=6)
        for sp in a.spines.values():
            sp.set_visible(False)
    fig.tight_layout(pad=0.2)
    simpan(fig, "bab15-rekonstruksi")


# Fungsi gambar baru disisipkan DI ATAS penanda ini.



if __name__ == "__main__":
    pola = re.compile(r"^bab\d\d_")
    pilihan = sys.argv[1:]
    fungsi = [(k, v) for k, v in sorted(globals().items())
              if pola.match(k) and callable(v)]
    if pilihan:
        fungsi = [(k, v) for k, v in fungsi
                  if any(p in k for p in pilihan)]
    if not fungsi:
        print("tidak ada gambar yang cocok")
    for _, f in fungsi:
        f()
