
"""Supervised Transformer Encoder - Tim 23
NIM: 123450105
Jalur: trf (klasifikasi normal/anomali)

Penggunaan:
    python models/123450105/run.py --seed 1
    python models/123450105/run.py --seed 1 --final
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.protokol import data

NIM = Path(__file__).resolve().parent.name
JALUR = "trf"


# =========================================================
# 1. TRANSFORMER ENCODER
# =========================================================

def buat_model(dropout=0.15):
    import torch
    import torch.nn as nn

    class TransformerKlasifikasi(nn.Module):

        def __init__(self):
            super().__init__()

            # 400 sampel dibagi menjadi 20 patch.
            # Setiap patch berisi 20 sampel x 2 kanal = 40 fitur.
            self.embedding = nn.Linear(40, 64)

            # Positional embedding untuk 20 patch.
            self.pos_embedding = nn.Parameter(
                torch.randn(1, 20, 64) * 0.02
            )

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=64,
                nhead=4,
                dim_feedforward=128,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu"
            )

            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=2
            )

            self.norm = nn.LayerNorm(64)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(64, 2)

        def forward(self, x):

            # Input: [batch, 2, 400]
            batch = x.shape[0]

            # [batch, 2, 20, 20]
            x = x.reshape(batch, 2, 20, 20)

            # [batch, 20, 2, 20]
            x = x.permute(0, 2, 1, 3)

            # [batch, 20, 40]
            x = x.reshape(batch, 20, 40)

            # Patch embedding + positional embedding
            x = self.embedding(x)
            x = x + self.pos_embedding

            # Transformer encoder
            x = self.encoder(x)

            # Global average pooling
            x = x.mean(dim=1)

            x = self.norm(x)
            x = self.dropout(x)

            # Output: [batch, 2]
            return self.classifier(x)

    return TransformerKlasifikasi()


# =========================================================
# 2. AUROC
# =========================================================

def hitung_auroc(y, skor):
    y = np.asarray(y).astype(bool)
    skor = np.asarray(skor)

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


# =========================================================
# 3. NORMALISASI DATA
# =========================================================

def normalisasi(X, mean, std):
    X = np.asarray(X, dtype=np.float32)

    return (
        (X - mean[None, :, None]) /
        std[None, :, None]
    ).astype(np.float32)


# =========================================================
# 4. TRAINING DAN PREDIKSI
# =========================================================

def latih_dan_prediksi(seed, final, epoch=100, batch=64):

    import torch
    import torch.nn.functional as F

    # Seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.set_num_threads(4)

    # Mengambil data sesuai protokol tim.
    Xtr, ytr, Xva, yva, Xte, yte = data.muat()

    Xtr = np.asarray(Xtr, dtype=np.float32)
    Xva = np.asarray(Xva, dtype=np.float32)

    ytr = np.asarray(ytr, dtype=np.int64).reshape(-1)
    yva = np.asarray(yva, dtype=np.int64).reshape(-1)

    print("\n========== DATASET ==========")
    print("X train:", Xtr.shape)
    print("y train:", ytr.shape)
    print("X validasi:", Xva.shape)
    print("y validasi:", yva.shape)

    if Xtr.ndim != 3 or Xtr.shape[1:] != (2, 400):
        raise ValueError("Bentuk data training harus [N, 2, 400].")

    if Xva.ndim != 3 or Xva.shape[1:] != (2, 400):
        raise ValueError("Bentuk data validasi harus [N, 2, 400].")

    if len(ytr) != len(Xtr) or len(yva) != len(Xva):
        raise ValueError("Jumlah data dan label tidak sama.")

    if not np.isfinite(Xtr).all() or not np.isfinite(Xva).all():
        raise ValueError("Data mengandung NaN atau infinity.")

    if not np.isin(ytr, [0, 1]).all():
        raise ValueError("Label training harus 0 atau 1.")

    if not np.isin(yva, [0, 1]).all():
        raise ValueError("Label validasi harus 0 atau 1.")

    print("Distribusi label train:", np.bincount(ytr, minlength=2))
    print("Distribusi label val:", np.bincount(yva, minlength=2))

    # Normalisasi hanya menggunakan data training.
    mean = Xtr.mean(axis=(0, 2))
    std = Xtr.std(axis=(0, 2)) + 1e-6

    Xtr = normalisasi(Xtr, mean, std)
    Xva = normalisasi(Xva, mean, std)

    # Tensor PyTorch.
    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr)

    Xv = torch.from_numpy(Xva)
    yv = torch.from_numpy(yva)

    # Class weight untuk mengatasi ketidakseimbangan kelas.
    jumlah = np.bincount(ytr, minlength=2).astype(np.float32)

    if np.any(jumlah == 0):
        raise ValueError("Data training harus memiliki kedua kelas.")

    bobot = len(ytr) / (2.0 * jumlah)
    class_weight = torch.tensor(bobot, dtype=torch.float32)

    # Model.
    model = buat_model(dropout=0.15)

    # Optimizer AdamW.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

    # Training.
    generator = torch.Generator().manual_seed(seed)

    best_loss = float("inf")
    best_state = None
    patience = 15
    tidak_membaik = 0
    riwayat = []

    print("\n========== TRAINING ==========")

    for ep in range(epoch):

        model.train()

        urutan = torch.randperm(
            len(Xt),
            generator=generator
        )

        total_loss = 0.0

        for i in range(0, len(Xt), batch):

            idx = urutan[i:i + batch]

            xb = Xt[idx]
            yb = yt[idx]

            optimizer.zero_grad()

            logits = model(xb)

            loss = F.cross_entropy(
                logits,
                yb,
                weight=class_weight
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item() * len(xb)

        train_loss = total_loss / len(Xt)

        # Evaluasi validasi.
        model.eval()

        with torch.no_grad():

            logits_val = model(Xv)

            val_loss = F.cross_entropy(
                logits_val,
                yv,
                weight=class_weight
            ).item()

        scheduler.step(val_loss)

        riwayat.append({
            "epoch": ep + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        if val_loss < best_loss:

            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep + 1
            tidak_membaik = 0

        else:
            tidak_membaik += 1

        if (ep + 1) % 10 == 0:
            print(
                f"Epoch {ep + 1:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

        if tidak_membaik >= patience:
            print("Early stopping aktif.")
            break

    # Menggunakan model dengan validation loss terbaik.
    model.load_state_dict(best_state)
    model.eval()

    # Prediksi validasi dan probabilitas anomali.
    with torch.no_grad():

        logits = model(Xv)

        probabilitas = torch.softmax(
            logits,
            dim=1
        ).numpy()

        pred_val = probabilitas.argmax(axis=1).astype(np.int64)

    # AUROC dari probabilitas kelas anomali.
    auroc = hitung_auroc(
        yva,
        probabilitas[:, 1]
    )

    # Final: prediksi data uji.
    pred_test = None

    if final:

        if Xte is None:
            raise ValueError("Data uji tidak tersedia.")

        Xte = np.asarray(Xte, dtype=np.float32)
        Xte = normalisasi(Xte, mean, std)

        with torch.no_grad():

            Xtst = torch.from_numpy(Xte)

            logits_test = model(Xtst)

            pred_test = (
                logits_test.argmax(dim=1)
                .numpy()
                .astype(np.int64)
            )

    tambahan = {
        "seed": seed,
        "arsitektur": "Transformer Encoder",
        "jalur": JALUR,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_loss),
        "auroc_val": float(auroc),
        "jumlah_parameter": sum(
            p.numel() for p in model.parameters()
        ),
        "train_loss_akhir": float(riwayat[-1]["train_loss"]),
        "riwayat_training": riwayat
    }

    return pred_val, pred_test, probabilitas[:, 1], tambahan


# =========================================================
# 5. MAIN DAN EVALUATOR DOSEN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--perangkat", default="cpu laptop")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)

    args = parser.parse_args()

    mulai = time.time()

    pred_val, pred_test, prob_val, tambahan = latih_dan_prediksi(
        args.seed,
        args.final,
        args.epoch,
        args.batch
    )

    durasi = time.time() - mulai

    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)

    # Simpan prediksi validasi.
    np.save(out / "pred_val.npy", pred_val)

    # Simpan probabilitas anomali untuk AUROC.
    np.save(out / "prob_val.npy", prob_val)

    # Simpan data uji jika --final digunakan.
    if pred_test is not None:
        np.save(out / "pred_test.npy", pred_test)

    # Simpan informasi eksperimen.
    (out / f"tambahan_seed{args.seed}.json").write_text(
        json.dumps(tambahan, indent=2),
        encoding="utf-8"
    )

    # Evaluator resmi dosen.
    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim", NIM,
        "--model", JALUR,
        "--seed", str(args.seed),
        "--pred-val", str(out / "pred_val.npy"),
        "--durasi", f"{durasi:.1f}",
        "--perangkat", args.perangkat
    ]

    if pred_test is not None:
        perintah += [
            "--pred-test",
            str(out / "pred_test.npy")
        ]

    print("\n========== EVALUATOR DOSEN ==========")

    subprocess.run(
        perintah,
        check=True,
        cwd=AKAR
    )

    print("\n========== SELESAI ==========")
    print("NIM:", NIM)
    print("Model:", JALUR)
    print("Seed:", args.seed)
    print("Jumlah prediksi validasi:", len(pred_val))
    print("AUROC validasi:", round(tambahan["auroc_val"], 4))
    print("Epoch terbaik:", tambahan["best_epoch"])
    print("Durasi:", round(durasi, 2), "detik")


if __name__ == "__main__":
    main()
