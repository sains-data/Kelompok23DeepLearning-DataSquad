
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
)

# Pastikan root proyek masuk ke Python path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.protokol import data


def ekstrak_fitur(X: np.ndarray) -> np.ndarray:
    """Ubah X [N, 2, 400] menjadi fitur statistik [N, F]."""
    fitur = []

    for jendela in X:
        baris = []

        for kanal in jendela:
            q25, median, q75 = np.percentile(
                kanal, [25, 50, 75]
            )

            baris.extend([
                np.mean(kanal),
                np.std(kanal),
                np.min(kanal),
                np.max(kanal),
                np.ptp(kanal),
                median,
                q25,
                q75,
                q75 - q25,
                np.sqrt(np.mean(kanal ** 2)),
                np.mean(np.abs(np.diff(kanal))),
            ])

        # Korelasi antara tegangan dan arus
        korelasi = np.corrcoef(jendela[0], jendela[1])[0, 1]
        if not np.isfinite(korelasi):
            korelasi = 0.0

        baris.append(korelasi)
        fitur.append(baris)

    return np.asarray(fitur, dtype=np.float64)


def cari_ambang(y_true, skor):
    """Pilih threshold dengan F1 terbaik pada validation set."""
    precision, recall, thresholds = precision_recall_curve(
        y_true, skor
    )

    if len(thresholds) == 0:
        raise ValueError("Tidak ditemukan threshold untuk validation set.")

    f1 = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    indeks = int(np.argmax(f1))
    return float(thresholds[indeks]), float(f1[indeks])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final",
        action="store_true",
        help="Evaluasi data uji satu kali setelah threshold ditetapkan.",
    )
    args = parser.parse_args()

    # Memuat split resmi dari protokol proyek
    X_train, y_train, X_val, y_val, X_test, y_test = data.muat()

    print("Ukuran data:")
    print("Train:", X_train.shape, y_train.shape)
    print("Val  :", X_val.shape, y_val.shape)
    print("Test :", X_test.shape, y_test.shape)

    # Ekstraksi fitur
    F_train = ekstrak_fitur(X_train)
    F_val = ekstrak_fitur(X_val)

    # Isolation Forest dilatih hanya menggunakan kelas normal
    normal = F_train[y_train == 0]
    if len(normal) == 0:
        raise ValueError("Tidak ada sampel normal pada data training.")

    model = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination="auto",
        random_state=2026,
        n_jobs=-1,
    )
    model.fit(normal)

    # Skor lebih tinggi berarti lebih anomali
    skor_val = -model.decision_function(F_val)

    # Threshold hanya ditentukan dari validation set
    threshold, val_f1 = cari_ambang(y_val, skor_val)
    pred_val = (skor_val >= threshold).astype(np.int64)

    hasil = {
        "model": "IsolationForest",
        "seed": 2026,
        "metrik": {
            "val_auroc": float(roc_auc_score(y_val, skor_val)),
            "val_f1": float(f1_score(y_val, pred_val, zero_division=0)),
        },
        "threshold": threshold,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
    }

    output = ROOT / "results" / "baseline"
    output.mkdir(parents=True, exist_ok=True)

    np.save(output / "pred_val.npy", pred_val)
    np.save(output / "score_val.npy", skor_val)

    # Evaluasi test hanya jika --final diberikan
    if args.final:
        F_test = ekstrak_fitur(X_test)
        skor_test = -model.decision_function(F_test)
        pred_test = (skor_test >= threshold).astype(np.int64)

        hasil["metrik"].update({
            "test_auroc": float(roc_auc_score(y_test, skor_test)),
            "test_f1": float(
                f1_score(y_test, pred_test, zero_division=0)
            ),
            "test_accuracy": float(accuracy_score(y_test, pred_test)),
        })

        np.save(output / "pred_test.npy", pred_test)
        np.save(output / "score_test.npy", skor_test)

    with open(output / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(hasil, f, indent=2, ensure_ascii=False)

    print("\nHasil evaluasi:")
    print(json.dumps(hasil, indent=2, ensure_ascii=False))
    print(f"\nHasil tersimpan di: {output}")


if __name__ == "__main__":
    main()
