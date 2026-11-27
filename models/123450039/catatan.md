# Catatan model 123450039 — Feby Angelina, jalur Autoencoder (ae, jalur utama)

Hasil di bawah adalah data **validasi** (ulangan 3, 6, 17; 518 jendela). Data uji (ulangan 1, 2, 7, 10, 14, 15, 19)
belum disentuh dan baru dievaluasi sekali di sprint terakhir. Angka mentah: `results/123450039/*/seed*/`,
ringkasan: `results/123450039/ringkasan.md` (`python models/123450039/ringkas.py`).

## Masukan dan arsitektur
- Masukan: jendela 2 s `[2, 400]` = tegangan rel `bus_mV` + arus kipas `arus_mA` dari INA219 kipas
  (kanal total/pompa tidak terpasang; pompa terlihat tidak langsung dari turunnya tegangan rel).
- Konfigurasi utama menambah 2 kanal **konteks** bernilai tetap: keadaan kipas dan keadaan lampu fisik
  (diketahui pengendali dari perintahnya sendiri). Keadaan pompa **tidak** pernah dimasukkan.
- AE konvolusi 1-D: encoder 400 → 200 → 100 → 50 langkah, laten 8 kanal × 50; decoder ConvTranspose kembali ke 400.
  13.370 parameter, ONNX 54,7 KB, latensi ±0,17 ms per jendela di CPU laptop.
- Dilatih **hanya** pada jendela normal latih (pompa = 0; 899 jendela). Skor anomali = MSE rekonstruksi kanal sinyal.
- Ambang = persentil 95 skor jendela normal latih (dipilih tanpa melihat contoh anomali, validasi, atau uji).

## Alasan optimizer
Adam (lr 1e-3): laju belajar adaptif per parameter membuat AE kecil konvergen stabil dalam 40 epoch tanpa jadwal
laju belajar. SGD perlu penyetelan lr dan momentum tambahan; belum diperlukan untuk Sprint 1.

## Regularisasi
- Weight decay 1e-4 (L2) pada Adam.
- Dropout 0,1 pada dua lapis pertama encoder.
- Bottleneck sempit (8 kanal) agar AE tidak sekadar menyalin masukan.
- Epoch tetap 40 (tidak memilih epoch dengan label validasi).

## Pencarian hyperparameter
Sprint 1 belum melakukan pencarian hyperparameter (lr, ukuran laten, epoch, dropout memakai nilai awal di atas).
Rencana Sprint 2: grid kecil lr {1e-3, 3e-3} × laten {4, 8, 16} × dropout {0, 0,1}, dipilih dengan AUROC kondisi validasi.

## Ablation (3 seed, validasi)

| Konfigurasi | AUROC kondisi | AUROC jendela | F1 anomali | val_acc | alarm palsu (dari 267 jendela normal) |
|---|---|---|---|---|---|
| **utama: 2 kanal + konteks** | **0,742 ± 0,038** | 0,749 ± 0,038 | 0,512 ± 0,110 | 0,656 ± 0,055 | 22,3 ± 1,5 |
| 2 kanal tanpa konteks | 0,625 ± 0,008 | 0,644 ± 0,010 | 0,467 ± 0,038 | 0,622 ± 0,004 | 31,0 ± 10,4 |
| hanya `bus_mV`, tanpa konteks | 0,331 ± 0,021 | 0,368 ± 0,019 | 0,000 | 0,430 ± 0,004 | 44,3 ± 2,1 |
| hanya `arus_mA`, tanpa konteks | 0,431 ± 0,027 | 0,390 ± 0,014 | 0,000 | 0,453 ± 0,002 | 32,3 ± 1,2 |

AUROC kondisi menurut keadaan (konfigurasi utama vs tanpa konteks):

| Keadaan | utama | tanpa konteks |
|---|---|---|
| kipas mati, lampu mati | 1,000 ± 0,000 | 0,000 ± 0,000 |
| kipas mati, lampu menyala | 1,000 ± 0,000 | 1,000 ± 0,000 |
| kipas menyala, lampu mati | 0,222 ± 0,096 | 0,111 ± 0,096 |
| kipas menyala, lampu menyala | 0,778 ± 0,192 | 0,722 ± 0,096 |

Pembacaan:
- Tanpa konteks, AE **terbalik** (AUROC 0) saat kipas & lampu mati: pompa menurunkan tegangan rel sebesar lampu
  menyala (±140–180 mV), sehingga "pompa + lampu mati" mirip "normal + lampu menyala", pola normal yang sudah dikenal.
  Memberi tahu keadaan lampu/kipas memperbaiki keadaan itu menjadi 1,0. Inilah alasan konteks dipilih sebagai konfigurasi utama.
- Satu kanal saja (bus atau arus) di bawah acak: kedua kanal saling melengkapi.
- Masih gagal saat kipas menyala + lampu mati (0,22). Dugaan (belum diuji): MSE didominasi riak arus kipas; saat pompa
  menyala arus kipas sedikit turun (±156 → ±142 mA) sehingga galatnya justru mengecil. Uji di Sprint 2: normalisasi per
  keadaan, atau skor dari kanal tegangan saja dengan konteks.

## Hasil 3 seed konfigurasi akhir (Sprint 1)
Seed 1 / 2 / 3: AUROC kondisi 0,708 / 0,783 / 0,733; F1 anomali 0,446 / 0,638 / 0,450.
Ambang berbeda cukup jauh antar-seed (0,58 / 0,13 / 0,58), tanda skor belum stabil; dibahas di Sprint 2.
Perintah reproduksi: `for s in 1 2 3; do python models/123450039/run.py --seed $s; done`.
