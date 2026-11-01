# Protokol Evaluasi Tim — DataSquad

| Butir | Isi |
|---|---|
| Kasus (dari kartu Kasus tim) | Deteksi anomali konsumsi daya (Autoencoder) |
| Sumber dataset | Data simulasi sintetis lokal untuk pengujian pipeline; dataset utama menggunakan rekaman ESP32; [capstone-dl-2026-starter](https://github.com/sains-data/capstone-dl-2026-starter) sebagai kerangka kode |
| Jenis tugas (`TUGAS`) | Klasifikasi |
| Target / label | 0 = Normal (pompa = 0); 1 = Anomali (pompa > 0) |
| Ukuran data latih / validasi / uji | 2.016 / 224 / belum tersedia |
| Cara split | Split melalui `src.common.split`, seed 2026, proporsi validasi 10% |
| Metrik utama untuk komparasi | AUROC dan F1 sesuai protokol kasus; baseline saat ini menghasilkan accuracy = 1.000 dan macro-F1 = 1.000 |
| Praproses yang sama untuk semua model | Arus 3 kanal, jendela 2 detik (400 sampel), buang 1 detik pertama, normalisasi berdasarkan data latih saja |
| Anggaran komputasi per run | ≤ 20 menit CPU laptop |
| Evaluasi data uji | Sekali, oleh Penjaga Data, pada sprint terakhir (`--final`) |
| `data_hash` saat dibekukan | `52af3ea6225c46ba2e00b2e422dd7a1d` |