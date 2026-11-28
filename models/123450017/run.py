
"""CNN 1D untuk klasifikasi anomali konsumsi daya Tim 23.

Jalankan:
    python models/123450017/run.py --seed 1
"""

from __future__ import annotations

# PyTorch harus diimpor sebelum NumPy dan modul protokol data.
import torch
import torch.nn as nn

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


# ============================================================
# KONFIGURASI
# ============================================================

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.protokol import data  # noqa: E402

NIM = Path(__file__).resolve().parent.name
JALUR = "cnn"

DEVICE = torch.device("cpu")


# ============================================================
# PERHITUNGAN AUROC
# ============================================================

def auroc(y: np.ndarray, skor: np.ndarray) -> float:
    """Menghitung AUROC menggunakan metode peringkat."""

    y = np.asarray(y).astype(bool)
    skor = np.asarray(skor, dtype=np.float64)

    n1 = int(y.sum())
    n0 = int((~y).sum())

    if n1 == 0 or n0 == 0:
        return float("nan")

    urut = np.argsort(skor, kind="mergesort")
    peringkat = np.empty(len(skor), dtype=np.float64)
    s = skor[urut]

    i = 0

    while i < len(s):
        j = i

        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1

        peringkat[urut[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1

    return float(
        (peringkat[y].sum() - n1 * (n1 + 1) / 2)
        / (n1 * n0)
    )


# ============================================================
# ARSITEKTUR CNN 1D
# ============================================================

def buat_model():

    class CNN1D(nn.Module):

        def __init__(self):
            super().__init__()

            self.features = nn.Sequential(

                # Blok konvolusi pertama
                nn.Conv1d(2, 16, kernel_size=7, padding=3),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(2),

                # Blok konvolusi kedua
                nn.Conv1d(16, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),

                # Blok konvolusi ketiga
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),

                # Menghasilkan panjang fitur satu
                nn.AdaptiveAvgPool1d(1),
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(64, 2),
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    return CNN1D()


# ============================================================
# PREDIKSI PROBABILITAS
# ============================================================

def prediksi_probabilitas(model, X, device):

    model.eval()
    hasil = []

    with torch.no_grad():

        for i in range(0, len(X), 128):

            x = torch.from_numpy(
                X[i:i + 128]
            ).to(device)

            logits = model(x)
            probabilitas = torch.softmax(logits, dim=1)

            hasil.append(probabilitas.cpu().numpy())

    return np.concatenate(hasil, axis=0)


# ============================================================
# PELATIHAN DAN VALIDASI CNN
# ============================================================

def latih_dan_prediksi(seed: int):

    # Mengatur seed agar eksperimen dapat direproduksi.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --------------------------------------------------------
    # 1. MEMUAT DATA
    # --------------------------------------------------------

    Xtr, ytr, Xva, yva, _, _ = data.muat()

    Xtr = np.asarray(Xtr, dtype=np.float32)
    Xva = np.asarray(Xva, dtype=np.float32)

    ytr = np.asarray(ytr, dtype=np.int64)
    yva = np.asarray(yva, dtype=np.int64)

    print("=== INFORMASI DATA ===")
    print("X_train:", Xtr.shape)
    print("X_val  :", Xva.shape)
    print("y_train:", ytr.shape)
    print("y_val  :", yva.shape)

    # Memastikan data tidak mengandung NaN atau infinity.
    if not np.isfinite(Xtr).all():
        raise ValueError("Data training mengandung NaN atau infinity.")

    if not np.isfinite(Xva).all():
        raise ValueError("Data validation mengandung NaN atau infinity.")

    # --------------------------------------------------------
    # 2. NORMALISASI DATA
    # --------------------------------------------------------

    # Statistik normalisasi hanya dihitung dari training.
    mu = Xtr.mean(axis=(0, 2), keepdims=True)

    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6

    Xtr = ((Xtr - mu) / sd).astype(np.float32)
    Xva = ((Xva - mu) / sd).astype(np.float32)

    X_train = torch.from_numpy(Xtr)
    y_train = torch.from_numpy(ytr)

    # --------------------------------------------------------
    # 3. INISIALISASI MODEL DAN OPTIMIZER
    # --------------------------------------------------------

    model = buat_model().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )

    criterion = nn.CrossEntropyLoss()

    batch_size = 64
    max_epoch = 50
    patience = 8

    generator = torch.Generator().manual_seed(seed)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    tanpa_perbaikan = 0
    riwayat = []

    # --------------------------------------------------------
    # 4. PROSES TRAINING
    # --------------------------------------------------------

    print("\n=== TRAINING CNN 1D ===")

    for epoch in range(max_epoch):

        model.train()

        urutan = torch.randperm(
            len(X_train),
            generator=generator,
        )

        total_loss = 0.0

        for i in range(0, len(X_train), batch_size):

            idx = urutan[i:i + batch_size]

            xb = X_train[idx].to(DEVICE)
            yb = y_train[idx].to(DEVICE)

            optimizer.zero_grad()

            logits = model(xb)

            loss = criterion(logits, yb)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Training loss tidak valid. "
                    "Periksa kembali data dan normalisasi."
                )

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)

        train_loss = total_loss / len(X_train)

        # ----------------------------------------------------
        # 5. EVALUASI VALIDATION
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            xv = torch.from_numpy(Xva).to(DEVICE)
            yv = torch.from_numpy(yva).to(DEVICE)

            val_logits = model(xv)

            val_loss = criterion(
                val_logits,
                yv,
            ).item()

        if not np.isfinite(val_loss):
            raise RuntimeError(
                "Validation loss tidak valid."
            )

        riwayat.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        })

        print(
            f"Epoch {epoch + 1:02d}/{max_epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # ----------------------------------------------------
        # 6. EARLY STOPPING
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch + 1

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            tanpa_perbaikan = 0

        else:

            tanpa_perbaikan += 1

        if tanpa_perbaikan >= patience:

            print("Early stopping dijalankan.")
            break

    if best_state is None:
        raise RuntimeError(
            "Model tidak menghasilkan bobot terbaik."
        )

    # Mengembalikan bobot terbaik.
    model.load_state_dict(best_state)

    # --------------------------------------------------------
    # 7. PREDIKSI VALIDATION
    # --------------------------------------------------------

    pred_val = prediksi_probabilitas(
        model,
        Xva,
        DEVICE,
    )

    # Probabilitas kelas 1 = anomali.
    skor_anomali = pred_val[:, 1]

    nilai_auroc = auroc(
        yva,
        skor_anomali,
    )

    # --------------------------------------------------------
    # 8. INFORMASI TAMBAHAN MODEL
    # --------------------------------------------------------

    tambahan = {

        "arsitektur": "CNN 1D tiga blok konvolusi",

        "optimizer": "Adam",

        "learning_rate": 0.001,

        "weight_decay": 0.0001,

        "dropout": 0.2,

        "batch_size": batch_size,

        "max_epoch": max_epoch,

        "patience": patience,

        "epoch_terbaik": best_epoch,

        "validation_loss_terbaik": float(best_val_loss),

        "auroc_val": float(nilai_auroc),

        "n_parameter": int(
            sum(
                p.numel()
                for p in model.parameters()
            )
        ),

        "riwayat_loss": riwayat,
    }

    return pred_val, tambahan


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    ap = argparse.ArgumentParser(
        description="Pelatihan CNN 1D Tim 23."
    )

    ap.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    ap.add_argument(
        "--final",
        action="store_true",
        help="Evaluasi data uji pada sprint terakhir.",
    )

    ap.add_argument(
        "--perangkat",
        default="cpu laptop",
    )

    a = ap.parse_args()

    t0 = time.time()

    # Sprint 1: hanya training dan validation.
    pred_val, tambahan = latih_dan_prediksi(a.seed)

    durasi = time.time() - t0

    # --------------------------------------------------------
    # 1. MENYIMPAN HASIL
    # --------------------------------------------------------

    out = AKAR / "results" / NIM

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    jalur_pred = out / "pred_val.npy"

    np.save(
        jalur_pred,
        pred_val,
    )

    tambahan["seed"] = a.seed
    tambahan["durasi_detik"] = round(durasi, 2)

    (out / "tambahan.json").write_text(
        json.dumps(
            tambahan,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 2. MENJALANKAN EVALUATOR RESMI TIM
    # --------------------------------------------------------

    perintah = [

        sys.executable,

        "-m",
        "src.common.evaluate",

        "--nim",
        NIM,

        "--model",
        JALUR,

        "--seed",
        str(a.seed),

        "--pred-val",
        str(jalur_pred),

        "--durasi",
        f"{durasi:.1f}",

        "--perangkat",
        a.perangkat,
    ]

    subprocess.run(
        perintah,
        check=True,
        cwd=AKAR,
    )

    # --------------------------------------------------------
    # 3. RINGKASAN HASIL
    # --------------------------------------------------------

    print("\n=== HASIL VALIDASI CNN ===")

    print(
        f"AUROC        : {tambahan['auroc_val']:.4f}"
    )

    print(
        f"Epoch terbaik: {tambahan['epoch_terbaik']}"
    )

    print(
        f"Durasi       : {durasi:.2f} detik"
    )

    print(
        f"Parameter    : {tambahan['n_parameter']}"
    )

    print("\nFile hasil:")
    print(out / "pred_val.npy")
    print(out / "tambahan.json")
    print(out / "metrics.json")


if __name__ == "__main__":
    main()
