# Catatan Eksperimen Isolation Forest

**NIM:** 123450105  
**Nama:** Arielva Simon Siahaan  
**Jalur:** Non-DL (Isolation Forest)  
**Jenis Pembelajaran:** Unsupervised Anomaly Detection  
**Dataset:** Data Primer Meja Tim 23  
**Klasifikasi:** Normal (0) dan Anomali (1)

## 1. Arsitektur Model

Model menggunakan Isolation Forest sebagai pembanding non-Deep Learning untuk mendeteksi anomali pada sinyal konsumsi daya.

Spesifikasi model:

- Algoritma: Isolation Forest.
- Input awal: 2 kanal × 400 sampel.
- Kanal: `bus_mV` dan `arus_mA`.
- Jumlah fitur hasil ekstraksi: 23 fitur.
- Fitur setiap kanal: mean, standar deviasi, minimum, maksimum, rentang, median, kuartil 25%, kuartil 75%, interquartile range, root mean square (RMS), dan rata-rata perubahan absolut antar-sampel.
- Fitur tambahan: korelasi antara kanal tegangan dan arus.
- Jumlah estimator: 300 pohon.
- Max samples: `auto`.
- Contamination: `auto`.
- Random state: 2026.
- Parallel processing: `n_jobs=-1`.

## 2. Preprocessing dan Ekstraksi Fitur

Setiap jendela sinyal memiliki bentuk `[2, 400]`, yang terdiri dari dua kanal dengan masing-masing 400 sampel.

Sinyal mentah diubah menjadi fitur statistik agar pola konsumsi daya dapat direpresentasikan dalam bentuk numerik yang lebih ringkas.

Fitur yang diekstraksi dari masing-masing kanal:

- Mean.
- Standar deviasi.
- Minimum dan maksimum.
- Rentang nilai.
- Median.
- Kuartil 25% dan 75%.
- Interquartile range.
- Root Mean Square (RMS).
- Rata-rata nilai absolut perubahan antar-sampel.

Selain itu, ditambahkan satu fitur korelasi antara tegangan dan arus.

Total fitur yang digunakan adalah 23 fitur untuk setiap jendela.

Tidak dilakukan normalisasi tambahan pada fitur.

## 3. Konfigurasi Training

- Algoritma: Isolation Forest.
- Jumlah estimator: 300.
- Max samples: `auto`.
- Contamination: `auto`.
- Random state: 2026.
- Perangkat: CPU laptop.
- Data training: 1.754 jendela.
- Data validasi: 518 jendela.
- Data test: 1.171 jendela.

Model dilatih hanya menggunakan sampel kelas normal (label 0) dari data training. Sampel anomali tidak digunakan dalam proses fitting model.

## 4. Penentuan Threshold

Isolation Forest menghasilkan anomaly score berdasarkan nilai negatif dari `decision_function`.

Semakin tinggi skor, semakin besar indikasi bahwa sampel merupakan anomali.

Threshold ditentukan menggunakan validation set melalui pencarian nilai yang menghasilkan F1-score terbaik.

Prediksi kelas dilakukan menggunakan aturan:

- Skor lebih besar atau sama dengan threshold: anomali (1).
- Skor lebih kecil dari threshold: normal (0).

Data test tidak digunakan untuk menentukan threshold.

Threshold akhir yang diperoleh adalah **-0.051065**.

## 5. Hasil Eksperimen Validasi

Hasil evaluasi validasi:

| Metrik | Hasil |
|---|---:|
| Validation AUROC | 0.657535 |
| Validation F1 | 0.661017 |
| Threshold | -0.051065 |
| Jumlah data training | 1.754 |
| Jumlah data validasi | 518 |
| Jumlah data test | 1.171 |

Hasil tersebut menunjukkan kemampuan awal Isolation Forest dalam membedakan kondisi normal dan anomali pada data primer meja.

## 6. Hasil Evaluasi Final

Evaluasi final dilakukan menggunakan perintah:

`python models/non-dl/run.py --final`

Threshold yang telah ditentukan berdasarkan validation set digunakan kembali untuk menghasilkan prediksi pada data test.

Hasil evaluasi:

| Metrik | Validasi | Test |
|---|---:|---:|
| AUROC | 0.657535 | 0.751889 |
| F1-score | 0.661017 | 0.628811 |
| Accuracy | - | 0.469684 |

Informasi tambahan:

- Random state: 2026.
- Threshold: -0.051065.
- Jumlah data validasi: 518.
- Jumlah data test: 1.171.
- Perangkat: CPU laptop.

Test AUROC sebesar 0.751889 menunjukkan kemampuan model dalam membedakan sampel normal dan anomali berdasarkan anomaly score. Test F1-score sebesar 0.628811 menunjukkan performa klasifikasi pada threshold yang ditentukan dari validation set.

## 7. File Hasil

Implementasi model disimpan pada:

`models/non-dl/`

File:
- `run.py`: Implementasi Isolation Forest dan ekstraksi fitur.
- `catatan.md`: Dokumentasi eksperimen.

Hasil eksperimen disimpan pada:

`results/baseline/`

File:
- `metrics.json`: Metrik evaluasi model.
- `pred_val.npy`: Prediksi kelas data validasi.
- `score_val.npy`: Anomaly score data validasi.
- `pred_test.npy`: Prediksi kelas data test.
- `score_test.npy`: Anomaly score data test.

Prediksi data test dan metrik evaluasi final dihasilkan ketika argumen `--final` digunakan.

## 8. Kesimpulan

Isolation Forest berhasil diimplementasikan sebagai model pembanding non-Deep Learning untuk tugas deteksi anomali pada data primer meja Tim 23.

Model menggunakan 23 fitur hasil ekstraksi statistik dari dua kanal sinyal, yaitu tegangan dan arus kipas.

Pelatihan dilakukan hanya menggunakan sampel normal pada data training, sedangkan validation set digunakan untuk menentukan threshold berdasarkan F1-score terbaik.

Pada evaluasi validasi, model memperoleh AUROC sebesar 0.657535 dan F1-score sebesar 0.661017.

Pada evaluasi final menggunakan data test, model memperoleh AUROC sebesar 0.751889, F1-score sebesar 0.628811, dan Accuracy sebesar 0.469684.

Hasil tersebut digunakan sebagai baseline pembanding bagi model Deep Learning, khususnya Transformer Encoder, dengan menggunakan pembagian data uji yang sama.

Seluruh hasil evaluasi final tersimpan pada `results/baseline/metrics.json`.