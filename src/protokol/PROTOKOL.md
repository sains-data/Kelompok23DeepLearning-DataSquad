# Protokol Evaluasi Tim — DataSquad (Tim 23)

Diusulkan lewat pull request (menggantikan protokol data simulasi); perlu review Penjaga Data dan persetujuan pembimbing.

| Butir | Isi |
|---|---|
| Kasus (dari kartu Kasus tim) | Deteksi anomali konsumsi daya (autoencoder): pompa menyala tanpa perintah = anomali |
| Sumber dataset | Sesi meja `sesi-20261113-124856` (13 Nov 2026, 160/160 kondisi); berkas dan sha256 di `data/MANIFEST.json`. Sesi uji 124151 dan 124243 tidak dipakai |
| Jenis tugas (`TUGAS`) | Deteksi anomali; ke penilai starter sebagai klasifikasi 2 kelas pada ambang |
| Target / label | pompa 0 → normal (0); pompa 255 → anomali (1), dari perintah skrip di `label.csv` |
| Masukan | INA219 kipas: `bus_mV` (tegangan rel 5 V) dan `arus_mA` (arus kipas) → `[2, 400]` |
| Penyimpangan dari rencana | Rencana meminta arus total/kipas/pompa; per 12 Nov hanya INA219 kipas (0x40) terpasang (diakui kartu Kasus tim). Pompa dideteksi tidak langsung lewat penurunan tegangan rel |
| Jendela | 2 s (400 sampel @ 200 Hz jam ESP32), geser 1 s, detik pertama tiap kondisi dibuang; jendela dengan celah sampel > 50 ms ditolak |
| Kondisi dikeluarkan | 12, 32, 46, 86, 136, 153: perintah kipas/pompa di `aktuator.csv` tidak sesuai label |
| Koreksi | Perintah lampu tercatat terbalik (info dosen 13 Nov); `label.csv` tidak diubah, keadaan fisik dikoreksi saat dibaca (`lampu_fisik`) |
| Satuan pemisahan | Ulangan (1–20), `split.indeks` seed 2026: uji 1, 2, 7, 10, 14, 15, 19; validasi 3, 6, 17; latih 4, 5, 8, 9, 11, 12, 13, 16, 18, 20 |
| Ukuran latih / validasi / uji | 1.754 / 518 / 1.171 jendela (normal/anomali: 899/855; 267/251; 625/546) |
| Metrik utama untuk komparasi | AUROC per kondisi (skor kondisi = skor jendela tertinggi), AUROC per jendela, F1 anomali pada ambang; rata-rata ± simpangan 3 seed; ONNX (KB) dan latensi (ms) |
| Praproses yang sama untuk semua model | Normalisasi per kanal dari data latih saja |
| Anggaran komputasi per run | ≤ 20 menit CPU laptop |
| Evaluasi data uji | Sekali, oleh Penjaga Data, pada sprint terakhir (`--final`) |
| `data_hash` | `f232927dd68ea75affe1962704e487fd` |
