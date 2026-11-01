# Catatan model 123450039 — Feby Angelina, jalur Autoencoder (ae)

Jalur utama tim untuk kasus deteksi anomali konsumsi daya (arus 3 kanal, jendela 2 detik).

## Rencana (Sprint 0)
- Autoencoder dilatih hanya pada data normal (pompa = 0); skor anomali = galat rekonstruksi.
- Ambang ditentukan dari data validasi, lalu dilaporkan AUROC dan F1 pada ambang (sesuai kartu Kasus tim).
- Menunggu protokol data tim (`src/protokol/data.py`) dari Penjaga Data setelah perekaman 5–6 Nov.

## Alasan optimizer

## Regularisasi

## Pencarian hyperparameter (tabel)

## Ablation (>= 2)

## Hasil 3 seed konfigurasi akhir
