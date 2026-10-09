# -*- coding: utf-8 -*-
"""Membangkitkan seluruh gambar Matplotlib ke gbr/ dalam dua bentuk:
PDF vektor untuk cetak dan PNG 300 dpi untuk EPUB.

Satu fungsi per gambar, dinamai babNN_nama(), yang memanggil
simpan(fig, "babNN-nama"). Fungsi bernama babNN_* dijalankan otomatis.
Benih acak selalu tetap, supaya gambar tidak berubah setiap build.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENIH = 20261010  # sama dengan seluruh kode/bab*.py
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
    panah(a, (0, 0), u, BIRU, "$\\mathbf{u} = (3, 1)$", -0.2, -0.45)
    panah(a, (0, 0), v, HIJAU, "$\\mathbf{v} = (1, 2)$", -1.0, 0.15)
    panah(a, u, u + v, HIJAU, "", lw=0.6)
    panah(a, v, u + v, BIRU, "", lw=0.6)
    panah(a, (0, 0), u + v, JINGGA,
          "$\\mathbf{u} + \\mathbf{v} = (4, 3)$", -1.2, 0.2)
    panah(a, (0, 0), 2 * v, ABU, "$2\\mathbf{v}$", 0.1, 0.0, lw=0.6)
    a.set_title("penjumlahan dan perkalian skalar")
    a.set_xlim(-0.3, 5)
    a.set_ylim(-0.3, 4.5)
    a = ax[1]
    panah(a, (0, 0), u, BIRU, "$\\mathbf{u}$", 0.1, -0.35)
    panah(a, (0, 0), v, HIJAU, "$\\mathbf{v}$", 0.05, 0.1)
    panah(a, (0, 0), w, MERAH, "$\\mathbf{w} = (-1, 3)$", -0.9, 0.15)
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
    from bab01_data import data_mini, BENIH
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
    rng = np.random.default_rng(BENIH)
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
