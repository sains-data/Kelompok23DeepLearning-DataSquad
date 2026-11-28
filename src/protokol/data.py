"""Pemuat data kasus Tim 23 (Deteksi anomali konsumsi daya) dari sesi meja asli.

Sumber   : sesi meja ``sesi-20261113-124856`` (160 kondisi, 13 Nov 2026), berkas di
           ``data/sesi-20261113-124856/`` dan tercantum di ``data/MANIFEST.json``.
Label    : dari perintah skrip di label.csv -> pompa 0 = normal (0), pompa 255 = anomali (1).
Masukan  : INA219 kipas (satu-satunya sensor arus yang terpasang), 2 kanal:
           tegangan rel 5 V (``bus_mV``) dan arus kipas (``arus_mA``).
           Rencana meminta arus total/kipas/pompa, tetapi INA219 total (0x44) dan pompa (0x41)
           belum terpasang per 12 Nov (diakui di kartu Kasus tim). Pompa dideteksi tidak
           langsung lewat penurunan tegangan rel karena semua alat berbagi catu 5 V.
Jendela  : 2 s (400 sampel @ 200 Hz menurut jam ESP32), bergeser 1 s, detik pertama tiap
           kondisi dibuang. Jendela dengan celah sampel > 50 ms ditolak.
Kondisi  : kondisi yang perintah kipas/pompa di aktuator.csv tidak sesuai label dikeluarkan.
Lampu    : perintah lampu tercatat terbalik (info dosen 13 Nov; lux lebih terang saat perintah 0).
           label.csv TIDAK diubah; keadaan fisik lampu dikoreksi saat dibaca (``lampu_fisik``).
Pemisahan: per ULANGAN (1-20) memakai ``src.common.split.indeks`` seed 2026:
           7 ulangan uji, 3 ulangan validasi, 10 ulangan latih. Satu ulangan tidak pernah
           muncul di dua bagian sekaligus.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import manifest, split

TUGAS = "klasifikasi"  # ke penilai starter: 2 kelas (0 normal, 1 anomali)

# ---- pengaturan dataset -------------------------------------------------------
SESI = "sesi-20261113-124856"
FOLDER = manifest.AKAR / "data" / SESI
FREKUENSI = 200          # Hz
PANJANG = 2 * FREKUENSI  # 400 sampel = jendela 2 s
GESER = 1.0              # detik
BUANG_DETIK = 1.0        # detik pertama tiap kondisi dibuang (panduan bagian 9.1)
CELAH_MAKS = 0.05        # detik; jendela dengan celah lebih panjang ditolak
KANAL = ["bus_mV", "arus_mA"]
VERSI_POTONG = "v2"     # naikkan bila aturan pemotongan berubah, agar cache dibuat ulang
CACHE = manifest.AKAR / "data" / f"_cache_{SESI}.npz"


# ---- langkah 1: baca dan selaraskan waktu ------------------------------------------
def _baca_daya() -> tuple[np.ndarray, np.ndarray]:
    """Kembalikan (t, sinyal[N, 2]) terurut waktu.

    Waktu memakai jam ESP32 (lebih rapat daripada jam laptop yang menerima per paket),
    lalu digeser ke skala jam laptop agar cocok dengan t_mulai/t_selesai di label.csv.
    """
    d = pd.read_csv(FOLDER / "daya.csv")
    d = d[d["kanal"] == "kipas"]
    t_esp = d["t_esp_ms"].to_numpy(np.float64) / 1000.0
    geser = np.median(d["t_host"].to_numpy(np.float64) - t_esp)
    t = t_esp + geser
    urut = np.argsort(t, kind="stable")          # 4 titik waktu mundur di data asli
    return t[urut], d[KANAL].to_numpy(np.float32)[urut]


# ---- langkah 2: kondisi yang perintah alatnya tidak sesuai label -------------------------
def kondisi_tidak_sesuai(lab: pd.DataFrame) -> list[int]:
    """Nomor kondisi yang perintah kipas/pompa (aktuator.csv) berbeda dari label.

    Perintah yang berlaku pada detik ke-1 harus sama dengan label, dan tidak boleh berganti
    ke nilai lain selama kondisi berjalan.
    """
    ak = pd.read_csv(FOLDER / "aktuator.csv").sort_values("t_host")
    keluar = []
    for r in lab.itertuples():
        for alat in ("kipas", "pompa"):
            a = ak[ak["nama"] == alat]
            harap = getattr(r, alat) > 0
            sebelum = a[a["t_host"] <= r.t_mulai_host + BUANG_DETIK]
            # perintah "matikan" di akhir kondisi (±0,01 s sebelum t_selesai) tidak dihitung
            dalam = a[(a["t_host"] > r.t_mulai_host + BUANG_DETIK) & (a["t_host"] < r.t_selesai_host - 0.1)]
            ok = len(sebelum) > 0 and (sebelum["nilai"].iloc[-1] > 0) == harap
            ok = ok and bool(((dalam["nilai"] > 0) == harap).all())
            if not ok:
                keluar.append(int(r.no))
                break
    return keluar


# ---- langkah 3: potong jendela -------------------------------------------------------
def _potong() -> dict[str, np.ndarray]:
    lab = pd.read_csv(FOLDER / "label.csv")
    t, s = _baca_daya()
    keluar = set(kondisi_tidak_sesuai(lab))
    X, meta = [], []
    for r in lab.itertuples():
        if int(r.no) in keluar:
            continue
        awal = r.t_mulai_host + BUANG_DETIK
        while awal + 2.0 <= r.t_selesai_host + 1e-9:
            i0, i1 = np.searchsorted(t, [awal, awal + 2.0])
            tt = t[i0:i1]
            utuh = len(tt) >= 2 and tt[0] - awal <= CELAH_MAKS and awal + 2.0 - tt[-1] <= CELAH_MAKS
            if utuh and np.diff(tt).max() <= CELAH_MAKS:
                # samakan ke 400 titik berjarak rata (interpolasi linear antar-sampel)
                grid = awal + (np.arange(PANJANG) + 0.5) / FREKUENSI
                X.append(np.stack([np.interp(grid, tt, s[i0:i1, k]) for k in range(len(KANAL))]))
                meta.append((r.no, r.ulangan, r.kipas, r.lampu, r.pompa))
            awal += GESER
    meta = np.array(meta, dtype=np.int64)
    return {
        "X": np.asarray(X, dtype=np.float32),            # [N, 2, 400]
        "y": (meta[:, 4] > 0).astype(np.int64),           # 1 = anomali (pompa menyala)
        "no": meta[:, 0], "ulangan": meta[:, 1], "kipas": meta[:, 2],
        "lampu_perintah": meta[:, 3],
        "lampu_fisik": np.where(meta[:, 3] > 0, 0, 255),  # koreksi lampu terbalik
        "keluar": np.array(sorted(keluar), dtype=np.int64),
    }


def jendela() -> dict[str, np.ndarray]:
    """Semua jendela beserta metadatanya; disimpan sementara di data/ (tidak di-commit)."""
    kunci = manifest.data_hash() + VERSI_POTONG
    if CACHE.exists():
        c = np.load(CACHE, allow_pickle=False)
        if str(c["kunci"]) == kunci:
            return {k: c[k] for k in c.files if k != "kunci"}
    j = _potong()
    np.savez_compressed(CACHE, kunci=kunci, **j)
    return j


# ---- langkah 4: pemisahan per ulangan ------------------------------------------------
def ulangan_split() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ulangan latih, validasi, uji) dari 20 ulangan, beku dengan seed 2026."""
    semua = np.arange(1, 21)
    sisa, uji = split.indeks(len(semua), porsi_val=0.35)      # 7 ulangan uji
    latih, val = split.indeks(len(sisa), porsi_val=0.23)      # 3 ulangan validasi
    return np.sort(semua[sisa][latih]), np.sort(semua[sisa][val]), np.sort(semua[uji])


def bagian(nama: str) -> np.ndarray:
    """Indeks jendela untuk 'latih' | 'val' | 'uji'."""
    j = jendela()
    u = dict(zip(("latih", "val", "uji"), ulangan_split()))[nama]
    return np.flatnonzero(np.isin(j["ulangan"], u))


def muat():
    """Kontrak starter: (X_train, y_train, X_val, y_val, X_test, y_test), X = [N, 2, 400]."""
    j = jendela()
    il, iv, iu = bagian("latih"), bagian("val"), bagian("uji")
    return j["X"][il], j["y"][il], j["X"][iv], j["y"][iv], j["X"][iu], j["y"][iu]
