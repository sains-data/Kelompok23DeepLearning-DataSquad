\# Catatan Eksperimen Transformer Encoder



\*\*NIM:\*\* 123450105  

\*\*Nama:\*\* Arielva Simon Siahaan  

\*\*Jalur:\*\* TRF (Transformer Encoder)  

\*\*Jenis Pembelajaran:\*\* Supervised Learning  

\*\*Dataset:\*\* Data Primer Meja Tim 23  

\*\*Klasifikasi:\*\* Normal (0) dan Anomali (1)



\## 1. Arsitektur Model



Model menggunakan Transformer Encoder untuk mengklasifikasikan jendela sinyal konsumsi daya.



Spesifikasi arsitektur:



\- Input: 2 kanal × 400 sampel.

\- Patch: 20 patch.

\- Ukuran setiap patch: 20 sampel × 2 kanal.

\- Dimensi setiap patch: 40 fitur.

\- Patch embedding: Linear 40 → 64.

\- Positional embedding: 20 posisi.

\- Transformer Encoder: 2 layer.

\- Multi-head attention: 4 head.

\- Feed-forward dimension: 128.

\- Pooling: Global Average Pooling (Mean Pooling).

\- Output: 2 kelas (normal dan anomali).

\- Jumlah parameter model: 71.106.



\## 2. Preprocessing



Normalisasi dilakukan menggunakan rata-rata dan standar deviasi data training saja.



Data validasi dan data uji menggunakan parameter normalisasi yang sama dengan data training untuk menghindari data leakage.



\## 3. Optimizer dan Regularisasi



\- Optimizer: AdamW.

\- Learning rate: 0.0003.

\- Weight decay: 0.0001.

\- Loss function: Weighted Cross Entropy.



Regularisasi:



\- Dropout: 0.15.

\- Gradient clipping dengan maksimum norm 1.0.

\- Early stopping dengan patience 15.

\- ReduceLROnPlateau untuk penyesuaian learning rate.



\## 4. Konfigurasi Training



\- Epoch maksimum: 100.

\- Batch size: 64.

\- Seed eksperimen: 1, 2, dan 3.

\- Perangkat: CPU laptop.

\- Checkpoint terbaik dipilih berdasarkan validation loss terendah.



\## 5. Alasan Pemilihan Arsitektur



Transformer Encoder dipilih karena mekanisme self-attention memungkinkan model mempelajari hubungan antarbagian sinyal dalam satu jendela waktu.



Pembagian sinyal menjadi patch membantu model mempelajari pola temporal secara lebih terstruktur.



AdamW dipilih karena menggunakan weight decay yang terpisah dari mekanisme pembaruan parameter.



Dropout dan early stopping digunakan untuk mengurangi risiko overfitting.



\## 6. Hasil Eksperimen



Hasil evaluasi validasi dari tiga eksperimen menggunakan seed berbeda:



| Metrik | Seed 1 | Seed 2 | Seed 3 |

|---|---:|---:|---:|

| Best Epoch | 8 | 11 | 11 |

| Best Validation Loss | 0.309822 | 0.316178 | 0.316527 |

| AUROC | 0.947804 | 0.956101 | 0.951027 |

| Validation Accuracy | 0.835907 | 0.855212 | 0.851351 |

| Validation Macro F1 | 0.835804 | 0.854926 | 0.850670 |

| Train Loss Akhir | 0.057204 | 0.056478 | 0.060587 |



\### Rata-rata Hasil Tiga Seed



| Metrik | Rata-rata |

|---|---:|

| Best Validation Loss | 0.314176 |

| Validation Accuracy | 0.847490 |

| Validation Macro F1 | 0.847133 |

| Validation AUROC | 0.951644 |

| Train Loss Akhir | 0.058090 |



Seed 2 memperoleh hasil terbaik berdasarkan Validation Accuracy, Macro F1, dan AUROC. Sementara itu, Seed 1 menghasilkan Best Validation Loss terendah sebesar 0.309822.



\## 7. Hasil Evaluasi Final



Evaluasi final dijalankan menggunakan perintah:



`python models/123450105/run.py --seed 3 --final`



Hasil evaluasi menggunakan evaluator resmi dosen:



| Metrik | Validasi | Test |

|---|---:|---:|

| Accuracy | 0.851351 | 0.886422 |

| Macro F1 | 0.850670 | 0.885484 |

| AUROC | 0.951027 | - |



Informasi tambahan evaluasi final:



\- Best Epoch: 11.

\- Durasi training: 45.61 detik.

\- Jumlah data validasi: 518.

\- Jumlah data test: 1.171.

\- Perangkat: CPU laptop.

\- Model: TRF (Transformer Encoder).



Test Accuracy sebesar 0.886422 menunjukkan bahwa model memperoleh akurasi sekitar 88.64% pada data uji. Sementara itu, Test Macro F1 sebesar 0.885484 menunjukkan keseimbangan performa klasifikasi pada kedua kelas.



\## 8. File Hasil



Seluruh hasil eksperimen disimpan pada folder:



`results/123450105/`



File yang dihasilkan:



\- `pred\_val.npy`: Prediksi kelas untuk data validasi, berukuran 518.

\- `prob\_val.npy`: Probabilitas prediksi untuk data validasi, berukuran 518.

\- `pred\_test.npy`: Prediksi kelas untuk data test, berukuran 1.171.

\- `metrics.json`: Hasil evaluasi resmi dosen.

\- `tambahan\_seed1.json`: Informasi tambahan eksperimen seed 1.

\- `tambahan\_seed2.json`: Informasi tambahan eksperimen seed 2.

\- `tambahan\_seed3.json`: Informasi tambahan eksperimen seed 3.



Prediksi data uji dibuat ketika argumen `--final` digunakan.



Seluruh file prediksi telah diperiksa dan berhasil dibaca menggunakan NumPy dengan bentuk array:



\- `pred\_val.npy`: (518,)

\- `prob\_val.npy`: (518,)

\- `pred\_test.npy`: (1171,)



\## 9. Kesimpulan



Model Transformer Encoder berhasil dilatih dan dievaluasi menggunakan tiga seed eksperimen, yaitu seed 1, seed 2, dan seed 3.



Berdasarkan hasil pengujian, rata-rata Validation Accuracy mencapai 84.75%, rata-rata Validation Macro F1 mencapai 84.71%, dan rata-rata Validation AUROC mencapai 95.16%.



Seed 2 memberikan performa validasi terbaik dengan Accuracy sebesar 85.52%, Macro F1 sebesar 85.49%, dan AUROC sebesar 95.61%. Sementara itu, Seed 1 menghasilkan nilai validation loss terendah sebesar 0.309822.



Pada evaluasi final menggunakan seed 3, model memperoleh Test Accuracy sebesar 88.64% dan Test Macro F1 sebesar 88.55%.



Hasil tersebut menunjukkan bahwa model Transformer Encoder mampu melakukan klasifikasi antara kondisi normal dan anomali pada data sinyal konsumsi daya dengan performa yang baik.

