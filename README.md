# Kode — Memahami Aljabar Linear untuk Machine Learning

Repositori pendamping buku **_Memahami Aljabar Linear untuk Machine
Learning: Dari Vektor sampai Dekomposisi, Kolom demi Kolom_** (Edisi
Pertama, 2026) oleh Mohammad Jamhuri, seri *Memahami*. Tag `edisi-1`
akan menandai kode yang tepat dipakai untuk mencetak edisi pertama.

Buku ini memakai satu **data mini** dari awal sampai akhir: empat
mahasiswa dengan dua fitur, x1 = jam belajar mandiri (1, 2, 4, 5) dan
x2 = jam belajar kelompok (1, 4, 2, 5), dan target y = skor kuis
(3, 10, 12, 15) di kolom terakhir. Kuadrat terkecil dengan konstanta
memberi bobot tepat w = (1, 2, 1) dengan galat (−1, 1, 1, −1), dan data
terpusat mempunyai nilai singular tepat 4 dan 2, sehingga LU, QR,
nilai eigen, dan SVD dapat dihitung tangan. File `kode/babNN_contoh.py`
memeriksa setiap bilangan di kotak Contoh Soal Bab NN.

**Setiap angka keluaran yang tercetak di buku dihasilkan oleh kode di
sini**, dan `periksa.py` membuktikannya: skrip itu menjalankan ulang
kode setiap bab dan mencocokkan hasilnya dengan blok keluaran yang
tercetak di buku.

Seluruh kode boleh dipakai, disalin, diubah, dan disebarluaskan secara
bebas untuk keperluan apa pun, termasuk komersial, tanpa kewajiban
mencantumkan sumber (lisensi [0BSD](LICENSE)).

## Menjalankan

```bash
git clone https://github.com/jamhuri-tech/buku-memahami-aljabar-linear.git
cd buku-memahami-aljabar-linear
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python kode/bab01_matriks.py
```

Pembangkit bilangan acak selalu memakai benih tetap (20261010),
sehingga keluarannya sama setiap kali dijalankan.

## Memeriksa angka di buku

```bash
.venv/bin/python periksa.py        # semua bab
.venv/bin/python periksa.py 02     # Bab 2 saja
```

Keluaran `SEMUA COCOK` berarti setiap blok keluaran di buku dihasilkan
ulang oleh kode ini. Pemeriksaan yang sama berjalan otomatis di GitHub
Actions setiap kali kode berubah.

## Struktur

```
kode/
  bab01_data.py      data mini dan matriks rancangan [1 | X]
  babNN_*.py         kode Bab NN
  babNN_contoh.py    pemeriksa hitungan tangan Contoh Soal Bab NN
keluaran/babNN.txt   blok keluaran yang tercetak di Bab NN
gen_gambar.py        membangkitkan semua gambar buku ke gbr/
periksa.py           mencocokkan kode dengan keluaran/
requirements.txt     versi pustaka yang dipakai buku
```

`python gen_gambar.py bab03` membangkitkan gambar Bab 3 saja.

## Salah ketik, galat, dan saran

Silakan buka [issue](../../issues) di repositori ini: sebutkan bab,
halaman, dan apa yang keliru. Saran juga dapat dikirim ke
m.jamhuri@live.com.
