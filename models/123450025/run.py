from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.protokol import data  # noqa: E402

NIM = Path(__file__).resolve().parent.name
JALUR = "rnn"


class ModelGRU(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        _, hidden = self.gru(x)
        fitur = hidden[-1]
        return self.classifier(fitur)


def auroc(y, skor) -> float:
    """AUROC berbasis peringkat (Mann-Whitney U), menangani nilai kembar.

    Sumber: fungsi auroc() di models/123450039/run.py (Feby), disalin ke sini.
    """
    y = np.asarray(y).astype(bool)
    skor = np.asarray(skor, dtype=np.float64)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    urut = np.argsort(skor, kind="mergesort")
    s = skor[urut]
    peringkat = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        peringkat[urut[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((peringkat[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def latih_dan_prediksi(seed: int, final: bool):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)

    Xtr, ytr, Xva, yva, Xte, _ = data.muat()

    # Normalisasi menggunakan statistik data latih saja.
    rata = Xtr.mean(axis=(0, 2), keepdims=True)
    std = Xtr.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-6)

    Xtr = (Xtr - rata) / std
    Xva = (Xva - rata) / std

    if final and Xte is not None:
        Xte = (Xte - rata) / std

    def siapkan_input(X):
        tensor = torch.tensor(np.asarray(X), dtype=torch.float32)

        # Downsampling 400 menjadi 100 titik waktu.
        tensor = torch.nn.functional.avg_pool1d(
            tensor, kernel_size=4, stride=4
        )

        # [N, 2, 100] menjadi [N, 100, 2].
        return tensor.transpose(1, 2).contiguous()

    x_train = siapkan_input(Xtr)
    x_val = siapkan_input(Xva)

    y_train = torch.tensor(
        np.asarray(ytr).reshape(-1), dtype=torch.long
    )

    y_val = np.asarray(yva).reshape(-1)

    dataset = TensorDataset(x_train, y_train)
    generator = torch.Generator().manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    model = ModelGRU()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )
    criterion = nn.CrossEntropyLoss()

    epochs = 60
    patience = 10
    terbaik = float("inf")
    model_terbaik = None
    tanpa_perbaikan = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            optimizer.zero_grad()

            output = model(xb)
            loss = criterion(output, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            output_val = model(x_val)
            loss_val = criterion(
                output_val,
                torch.tensor(y_val, dtype=torch.long),
            ).item()

        loss_train = total_loss / len(dataset)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1:02d} | "
                f"Train Loss: {loss_train:.4f} | "
                f"Val Loss: {loss_val:.4f}"
            )

        if loss_val < terbaik:
            terbaik = loss_val
            model_terbaik = copy.deepcopy(model.state_dict())
            tanpa_perbaikan = 0
        else:
            tanpa_perbaikan += 1

        if tanpa_perbaikan >= patience:
            print(f"Early stopping pada epoch {epoch + 1}")
            break

    model.load_state_dict(model_terbaik)
    model.eval()

    with torch.no_grad():
        logit_val = model(x_val)
        pred_val = logit_val.argmax(dim=1).numpy()
        # peluang kelas anomali (kelas 1) untuk menghitung AUROC
        prob_val = torch.softmax(logit_val, dim=1)[:, 1].numpy()

        pred_test = None
        if final and Xte is not None:
            x_test = siapkan_input(Xte)
            pred_test = model(x_test).argmax(dim=1).numpy()

    tambahan = {
        "seed": seed,
        "auroc_jendela": round(auroc(y_val, prob_val), 4),
        "n_parameter": int(sum(p.numel() for p in model.parameters())),
    }
    return pred_val, pred_test, prob_val, tambahan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--perangkat", default="cpu laptop")
    args = parser.parse_args()

    mulai = time.time()
    pred_val, pred_test, prob_val, tambahan = latih_dan_prediksi(args.seed, args.final)
    durasi = time.time() - mulai

    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "pred_val.npy", pred_val)
    np.save(out / "prob_val.npy", prob_val)
    (out / f"tambahan_seed{args.seed}.json").write_text(
        json.dumps(tambahan, indent=2) + "\n", encoding="utf-8"
    )
    print(f"AUROC validasi (per jendela): {tambahan['auroc_jendela']}")

    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim", NIM,
        "--model", JALUR,
        "--seed", str(args.seed),
        "--pred-val", str(out / "pred_val.npy"),
        "--durasi", f"{durasi:.1f}",
        "--perangkat", args.perangkat,
    ]

    if pred_test is not None:
        np.save(out / "pred_test.npy", pred_test)
        perintah += ["--pred-test", str(out / "pred_test.npy")]

    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()
