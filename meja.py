import argparse
import csv
import itertools
import math
import random
from datetime import datetime
from pathlib import Path


# Konfigurasi simulasi kasus Tim 23
DURASI_DEFAULT = 30
ULANGAN_DEFAULT = 20
FREKUENSI_DEFAULT = 200  # Hz per kanal
KANAL = ["total", "kipas", "pompa"]

# 8 kombinasi: 2 pilihan kipas x 2 lampu x 2 pompa
KONDISI = list(
    itertools.product(
        [0, 160],   # kipas
        [0, 255],   # lampu
        [0, 255],   # pompa
    )
)


def buat_arus_sintetis(waktu, indeks_kanal, kipas, lampu, pompa, rng):
    """
    Menghasilkan arus buatan untuk mengetes alur program.
    Angka ini bukan hasil pengukuran sensor dan bukan model fisik alat.
    """
    arus_dasar = [18.0, 20.0, 16.0][indeks_kanal]
    kontribusi_kipas = (kipas / 160.0) * [95.0, 82.0, 105.0][indeks_kanal]
    kontribusi_lampu = (lampu / 255.0) * [42.0, 48.0, 38.0][indeks_kanal]
    kontribusi_pompa = (pompa / 255.0) * [230.0, 205.0, 250.0][indeks_kanal]

    riak = 2.0 * math.sin(2.0 * math.pi * 1.2 * waktu)
    noise = rng.gauss(0.0, 1.5)

    riak_pompa = 0.0
    if pompa > 0:
        riak_pompa = 8.0 * math.sin(
            2.0 * math.pi * 7.0 * waktu + indeks_kanal
        )

    arus = (
        arus_dasar
        + kontribusi_kipas
        + kontribusi_lampu
        + kontribusi_pompa
        + riak
        + riak_pompa
        + noise
    )
    return max(0.0, arus)


def jalankan_simulasi(durasi, ulangan, frekuensi, seed, folder_output):
    if durasi <= 1:
        raise ValueError("Durasi harus lebih dari 1 detik karena 1 detik pertama dibuang.")
    if ulangan < 1:
        raise ValueError("Jumlah ulangan minimal 1.")
    if frekuensi < 1:
        raise ValueError("Frekuensi minimal 1 Hz.")

    folder = Path(folder_output)
    folder.mkdir(parents=True, exist_ok=True)

    nama_file = folder / (
        "rekaman_simulasi_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    )

    rng = random.Random(seed)
    total_sesi = len(KONDISI) * ulangan
    jumlah_baris = 0

    print("=" * 65)
    print("MODE SIMULASI - TANPA ESP32 DAN TANPA AKTUATOR FISIK")
    print(f"Jumlah kombinasi       : {len(KONDISI)}")
    print(f"Ulangan per kombinasi  : {ulangan}")
    print(f"Durasi per kondisi     : {durasi} detik")
    print(f"Frekuensi per kanal    : {frekuensi} Hz")
    print("Detik pertama dibuang  : Ya")
    print(f"File keluaran          : {nama_file}")
    print("=" * 65)

    with nama_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "kondisi",
            "ulangan",
            "kipas",
            "lampu",
            "pompa",
            "label",
            "waktu_eksperimen",
            "ms_esp32",
            "kanal",
            "bus_mV",
            "arus_mA",
        ])

        nomor_sesi = 0

        for nomor_kondisi, (kipas, lampu, pompa) in enumerate(KONDISI, start=1):
            label = 0 if pompa == 0 else 1

            for nomor_ulangan in range(1, ulangan + 1):
                nomor_sesi += 1
                print(
                    f"[{nomor_sesi}/{total_sesi}] "
                    f"Kondisi {nomor_kondisi}/8, "
                    f"Ulangan {nomor_ulangan}/{ulangan}, "
                    f"Kipas={kipas}, Lampu={lampu}, Pompa={pompa}, "
                    f"Label={label}"
                )

                # Sampel dimulai pada detik ke-1: detik pertama dibuang.
                # Satu waktu sampel menghasilkan satu baris untuk tiap kanal.
                for nomor_sampel in range(frekuensi, durasi * frekuensi):
                    waktu = nomor_sampel / frekuensi
                    waktu_ms = int(waktu * 1000)

                    for indeks_kanal, nama_kanal in enumerate(KANAL):
                        arus = buat_arus_sintetis(
                            waktu,
                            indeks_kanal,
                            kipas,
                            lampu,
                            pompa,
                            rng,
                        )
                        tegangan_bus = 5000.0 + rng.gauss(0.0, 3.0)

                        writer.writerow([
                            nomor_kondisi,
                            nomor_ulangan,
                            kipas,
                            lampu,
                            pompa,
                            label,
                            f"{waktu:.4f}",
                            waktu_ms,
                            nama_kanal,
                            f"{tegangan_bus:.2f}",
                            f"{arus:.4f}",
                        ])
                        jumlah_baris += 1

                file.flush()

    print("\nSIMULASI SELESAI")
    print(f"Jumlah baris data: {jumlah_baris:,}")
    print(f"File tersimpan   : {nama_file}")
    print(
        "PERINGATAN: data ini sintetis untuk mengetes program. "
        "Bukan data primer ESP32 dan bukan untuk evaluasi model final."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Simulasi pencatatan arus 3 kanal untuk Tim 23."
    )

    parser.add_argument(
        "--simulasi",
        action="store_true",
        help="Jalankan simulasi lokal tanpa perangkat fisik.",
    )
    parser.add_argument(
        "--durasi",
        type=int,
        default=DURASI_DEFAULT,
        help=f"Durasi tiap kondisi dalam detik (default {DURASI_DEFAULT}).",
    )
    parser.add_argument(
        "--ulangan",
        type=int,
        default=ULANGAN_DEFAULT,
        help=f"Ulangan tiap kombinasi (default {ULANGAN_DEFAULT}).",
    )
    parser.add_argument(
        "--frekuensi",
        type=int,
        default=FREKUENSI_DEFAULT,
        help=f"Frekuensi sampel per kanal dalam Hz (default {FREKUENSI_DEFAULT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Seed simulasi (default 2026).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Folder penyimpanan CSV (default data).",
    )

    args = parser.parse_args()

    if not args.simulasi:
        parser.error("Gunakan opsi --simulasi. Script ini tidak mengontrol meja uji.")

    try:
        jalankan_simulasi(
            durasi=args.durasi,
            ulangan=args.ulangan,
            frekuensi=args.frekuensi,
            seed=args.seed,
            folder_output=args.output,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
