# Catatan Model 123450025

## Arsitektur Model

Model menggunakan Gated Recurrent Unit (GRU) untuk klasifikasi
dua kelas, yaitu normal (0) dan anomali (1).

* Input: 2 kanal dengan 400 titik waktu.
* Normalisasi: mean dan standar deviasi dari data latih.
* Downsampling: average pooling, 400 menjadi 100 titik waktu.
* RNN: GRU, 2 lapis, hidden size 64.
* Dropout: 0,2.
* Fully connected: 64 ke 32 ke 2 kelas.
* Fungsi aktivasi: ReLU pada lapisan tersembunyi.
* Loss: Cross Entropy Loss.
* Optimizer: Adam, learning rate 0,001.
* Weight decay: 0,0001.
* Batch size: 64.
* Maksimum epoch: 60.
* Early stopping: patience 10 berdasarkan validation loss.

## Alasan Pemilihan Optimizer

Adam dipilih karena dapat menyesuaikan learning rate setiap parameter
selama proses pelatihan. Optimizer ini cukup sesuai untuk pelatihan
GRU pada data deret waktu.

## Regularisasi

Dropout 0,2 digunakan untuk mengurangi ketergantungan model terhadap
neuron tertentu. Weight decay 0,0001 juga digunakan untuk membantu
mengurangi overfitting. Early stopping dilakukan berdasarkan
validation loss.

## Pencarian Hyperparameter

|Parameter|Konfigurasi awal|
|-|-|
|Hidden size|64|
|Jumlah layer|2|
|Dropout|0,2|
|Learning rate|0,001|
|Batch size|64|
|Weight decay|0,0001|

## Ablation Study

Eksperimen berikut direncanakan untuk membandingkan pengaruh
komponen model. Hasil diisi setelah eksperimen benar-benar dijalankan.

|Eksperimen|Perubahan|Hasil|
|-|-|-|
|Ablation 1|Hidden size 32 dibandingkan 64|Belum dijalankan|
|Ablation 2|Tanpa downsampling dibandingkan downsampling 4x|Belum dijalankan|

## Hasil Tiga Seed

|Seed|Accuracy|Macro F1|
|-|-|-|
|1|0,847490	|0,847444|
|2|Belum dijalankan|Belum dijalankan|
|3|Belum dijalankan|Belum dijalankan|

## Hasil dan Evaluasi

Hasil evaluasi akan diperbarui berdasarkan metrics.json dari
penilai starter setelah eksperimen dijalankan.

Model GRU memperoleh validation accuracy sebesar 84,75% dan validation macro F1 sebesar 84,74% pada seed 1. Proses training berhenti pada epoch ke-27 menggunakan early stopping.

