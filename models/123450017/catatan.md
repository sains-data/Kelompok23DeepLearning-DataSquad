
# Catatan Model CNN 1D
NIM: 123450017

## 1. Alasan Pemilihan Optimizer

Model CNN 1D menggunakan optimizer Adam dengan learning rate sebesar 0.001. Adam dipilih karena mengadaptasi learning rate berdasarkan estimasi momen pertama dan kedua gradien, sehingga proses optimasi dapat berlangsung secara adaptif.

Optimizer ini digunakan untuk memperbarui bobot pada setiap iterasi pelatihan dengan meminimalkan nilai Cross Entropy Loss.

## 2. Regularisasi

Regularisasi yang digunakan dalam model meliputi:

- Dropout sebesar 0.2 pada bagian classifier untuk mengurangi risiko overfitting.
- Weight decay sebesar 0.0001 pada optimizer Adam untuk memberikan penalti terhadap bobot yang terlalu besar.
- Early stopping dengan patience 8 berdasarkan validation loss.
- Batch Normalization pada setiap blok konvolusi untuk membantu menstabilkan proses pelatihan.

## 3. Konfigurasi Hyperparameter

Konfigurasi yang digunakan pada eksperimen awal:

| Hyperparameter | Nilai |
|---|---:|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 64 |
| Dropout | 0.2 |
| Maximum epoch | 50 |
| Patience | 8 |
| Seed | 1 |

Catatan: Konfigurasi di atas merupakan konfigurasi eksperimen awal. Pencarian hyperparameter secara sistematis belum dilakukan.

## 4. Ablation Study

Ablation study belum dilakukan pada eksperimen ini.

Dua rancangan pengujian yang dapat dilakukan selanjutnya:

1. Menghilangkan Dropout untuk mengamati pengaruh regularisasi Dropout terhadap performa validation.
2. Menghilangkan Weight Decay untuk mengamati pengaruh regularisasi bobot terhadap performa validation.

Hasil kedua pengujian belum tersedia sehingga belum dapat dibandingkan.

## 5. Hasil Eksperimen Seed

Eksperimen yang telah dilakukan menggunakan seed 1.

| Metrik | Hasil |
|---|---:|
| Validation Accuracy | 0.924710 |
| Validation Macro F1 | 0.924404 |
| Validation AUROC | 0.987400 |
| Epoch terbaik | 26 |
| Durasi training | 23.67 detik |
| Jumlah parameter | 9,394 |

Eksperimen menggunakan data training sebanyak 1.754 observasi dan validation sebanyak 518 observasi.

Eksperimen seed 2 dan seed 3 belum dilakukan.

## 6. Arsitektur Model

Model menggunakan CNN 1D dengan tiga blok konvolusi:

1. Conv1D dengan 16 filter dan kernel size 7.
2. Conv1D dengan 32 filter dan kernel size 5.
3. Conv1D dengan 64 filter dan kernel size 3.

Setiap blok menggunakan Batch Normalization dan ReLU. Max Pooling digunakan pada dua blok pertama, sedangkan Adaptive Average Pooling digunakan untuk merangkum fitur sebelum klasifikasi.

Bagian classifier menggunakan Flatten, Dropout, dan Linear Layer dengan dua output kelas, yaitu normal dan anomali.

## 7. Catatan Evaluasi

Evaluasi dilakukan menggunakan evaluator resmi Tim 23 melalui `src.common.evaluate`.

Data test belum digunakan. Seluruh hasil yang dilaporkan pada dokumen ini merupakan hasil evaluasi validation untuk seed 1.