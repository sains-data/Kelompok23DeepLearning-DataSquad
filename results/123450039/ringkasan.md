# Ringkasan hasil AE (validasi; uji belum disentuh)

| Konfigurasi | n seed | AUROC kondisi | AUROC jendela | F1 anomali | presisi | recall | alarm palsu (FP) | parameter | latensi (ms) | ONNX (KB) |
|---|---|---|---|---|---|---|---|---|---|---|
| ablation-arus-tanpa-konteks | 3 | 0.431 ± 0.027 | 0.390 ± 0.014 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 32.333 ± 1.155 | 12921 | 0.155 ± 0.025 | 52.9 |
| ablation-both-tanpa-konteks | 3 | 0.625 ± 0.008 | 0.644 ± 0.010 | 0.467 ± 0.038 | 0.740 ± 0.034 | 0.344 ± 0.051 | 31.000 ± 10.392 | 13146 | 0.152 ± 0.014 | 53.8 |
| ablation-bus-tanpa-konteks | 3 | 0.331 ± 0.021 | 0.368 ± 0.019 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 44.333 ± 2.082 | 12921 | 0.152 ± 0.020 | 52.9 |
| utama | 3 | 0.742 ± 0.038 | 0.749 ± 0.038 | 0.512 ± 0.110 | 0.803 ± 0.045 | 0.380 ± 0.113 | 22.333 ± 1.528 | 13370 | 0.165 ± 0.027 | 54.7 |

## AUROC per kondisi menurut keadaan kipas × lampu fisik

| Konfigurasi | Keadaan | AUROC kondisi |
|---|---|---|
| ablation-arus-tanpa-konteks | kipas0_lampu0 | 0.704 ± 0.128 |
| ablation-arus-tanpa-konteks | kipas0_lampu255 | 0.556 ± 0.294 |
| ablation-arus-tanpa-konteks | kipas160_lampu0 | 0.000 ± 0.000 |
| ablation-arus-tanpa-konteks | kipas160_lampu255 | 0.444 ± 0.096 |
| ablation-both-tanpa-konteks | kipas0_lampu0 | 0.000 ± 0.000 |
| ablation-both-tanpa-konteks | kipas0_lampu255 | 1.000 ± 0.000 |
| ablation-both-tanpa-konteks | kipas160_lampu0 | 0.111 ± 0.096 |
| ablation-both-tanpa-konteks | kipas160_lampu255 | 0.722 ± 0.096 |
| ablation-bus-tanpa-konteks | kipas0_lampu0 | 0.000 ± 0.000 |
| ablation-bus-tanpa-konteks | kipas0_lampu255 | 0.481 ± 0.064 |
| ablation-bus-tanpa-konteks | kipas160_lampu0 | 0.111 ± 0.096 |
| ablation-bus-tanpa-konteks | kipas160_lampu255 | 0.556 ± 0.096 |
| utama | kipas0_lampu0 | 1.000 ± 0.000 |
| utama | kipas0_lampu255 | 1.000 ± 0.000 |
| utama | kipas160_lampu0 | 0.222 ± 0.096 |
| utama | kipas160_lampu255 | 0.778 ± 0.192 |
