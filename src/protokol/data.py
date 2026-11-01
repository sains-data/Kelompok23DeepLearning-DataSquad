
"""Pemuat data kasus tim: Deteksi anomali konsumsi daya.

Data:
- Sumber: rekaman arus dari ESP32.
- Kanal: total, kipas, pompa.
- Frekuensi: 200 Hz per kanal.
- Window: 2 detik (400 sampel).
- Label: 0 = normal, 1 = anomali.
- Split: src.common.split, seed 2026.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common import manifest, split


TUGAS = "klasifikasi"

# Pengaturan dataset
FOLDER = manifest.AKAR / "data"
FREKUENSI = 200
DURASI_WINDOW = 2
SAMPEL_PER_WINDOW = FREKUENSI * DURASI_WINDOW

# Buang 1 detik pertama setiap kondisi
BUANG_DETIK = 1

# Urutan kanal input model
KANAL = ["total", "kipas", "pompa"]

# Pemetaan nilai kanal jika ESP32 mengirim angka.
# Sesuaikan jika firmware menggunakan kode berbeda.
PEMETAAN_KANAL = {
    "0": "total",
    "1": "kipas",
    "2": "pompa",
    "total": "total",
    "kipas": "kipas",
    "pompa": "pompa",
}


def _baca_csv() -> pd.DataFrame:
    """Membaca seluruh CSV hasil rekaman dari folder data."""

    berkas = sorted(FOLDER.glob("rekaman_*.csv"))

    if not berkas:
        raise FileNotFoundError(
            "Tidak ditemukan rekaman_*.csv di folder data/. "
            "Masukkan file hasil rekaman ESP32 terlebih dahulu."
        )

    daftar = []

    for file in berkas:
        df = pd.read_csv(file)
        df["sesi"] = file.stem
        daftar.append(df)

    data = pd.concat(daftar, ignore_index=True)

    kolom_wajib = {
        "kondisi",
        "ulangan",
        "kipas",
        "lampu",
        "pompa",
        "label",
        "ms_esp32",
        "kanal",
        "arus_mA",
    }

    kurang = kolom_wajib - set(data.columns)

    if kurang:
        raise ValueError(
            f"Kolom CSV tidak lengkap: {sorted(kurang)}"
        )

    return data


def _buat_window(data: pd.DataFrame):
    """Mengubah data arus mentah menjadi window berukuran (400, 3)."""

    X = []
    y = []
    groups = []

    # Satu rekaman = satu kombinasi kondisi dan ulangan
    kolom_grup = ["sesi", "kondisi", "ulangan"]

    for identitas, bagian in data.groupby(kolom_grup, sort=True):

        bagian = bagian.copy()

        # Konversi kode kanal menjadi nama kanal
        bagian["kanal"] = (
            bagian["kanal"].astype(str).str.strip().str.lower()
            .map(PEMETAAN_KANAL)
        )

        if bagian["kanal"].isna().any():
            raise ValueError(
                "Ada kode kanal yang belum dikenali. "
                "Periksa PEMETAAN_KANAL."
            )

        # Susun data menjadi matriks waktu x kanal
        lebar = bagian.pivot_table(
            index="ms_esp32",
            columns="kanal",
            values="arus_mA",
            aggfunc="mean",
        )

        # Pastikan ketiga kanal tersedia
        if not set(KANAL).issubset(lebar.columns):
            continue

        lebar = lebar[KANAL].sort_index()

        # Buang data satu detik pertama
        jumlah_buang = int(BUANG_DETIK * FREKUENSI)
        lebar = lebar.iloc[jumlah_buang:]

        nilai = lebar.to_numpy(dtype=np.float32)

        # Label kondisi
        label = int(bagian["label"].iloc[0])

        if label not in (0, 1):
            raise ValueError(f"Label tidak valid: {label}")

        # Bentuk window 400 sampel
        jumlah_window = len(nilai) // SAMPEL_PER_WINDOW

        for i in range(jumlah_window):
            awal = i * SAMPEL_PER_WINDOW
            akhir = awal + SAMPEL_PER_WINDOW

            window = nilai[awal:akhir]

            # Lewati window yang mengandung nilai kosong
            if not np.isfinite(window).all():
                continue

            X.append(window)
            y.append(label)
            groups.append(identitas)

    if not X:
        raise ValueError(
            "Tidak ada window yang berhasil dibuat. "
            "Periksa jumlah sampel, frekuensi, dan nama kanal."
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(groups, dtype=object),
    )


def muat():
    """Mengembalikan data training, validasi, dan testing."""

    # 1. Baca data mentah
    data = _baca_csv()

    # 2. Bentuk window
    X, y, groups = _buat_window(data)

    # 3. Split dengan fungsi resmi starter
    tr, va = split.indeks(len(X), porsi_val=0.1)

    X_train = X[tr]
    y_train = y[tr]

    X_val = X[va]
    y_val = y[va]

    # 4. Tidak ada file test terpisah pada dataset ini
    X_test = None
    y_test = None

    print("Berhasil memuat dataset!")
    print("Bentuk X:", X.shape)
    print("Jumlah training:", len(X_train))
    print("Jumlah validasi:", len(X_val))
    print("Distribusi label:", np.unique(y, return_counts=True))

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )