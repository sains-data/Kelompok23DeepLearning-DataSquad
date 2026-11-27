"""Jalur AE (jalur utama Tim 23): autoencoder konvolusi 1-D untuk deteksi anomali konsumsi daya.

Pemilik: Feby Angelina (123450039).

    python models/123450039/run.py --seed 1                         # konfigurasi utama (2 kanal + konteks)
    python models/123450039/run.py --seed 1 --tanpa-konteks         # ablation: tanpa keadaan kipas/lampu
    python models/123450039/run.py --seed 1 --tanpa-konteks --kanal bus   # ablation: hanya tegangan rel
    python models/123450039/run.py --seed 1 --tanpa-konteks --kanal arus  # ablation: hanya arus kipas
    python models/123450039/run.py --seed 1 --final            # sprint terakhir: + data uji (SEKALI)

Ide: AE hanya melihat jendela NORMAL (pompa = 0) saat latih, sehingga pandai menyusun ulang pola
normal. Jendela saat pompa menyala tersusun ulang lebih buruk -> galat rekonstruksi (skor) besar.
Ambang = persentil 95 skor jendela normal latih (dipilih tanpa melihat contoh anomali).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402  (protokol tim)

NIM = Path(__file__).resolve().parent.name
JALUR = "ae"


# ---- metrik tambahan (AUROC tidak dihitung penilai starter) ---------------------------
def auroc(y: np.ndarray, skor: np.ndarray) -> float:
    """AUROC berbasis peringkat (setara Mann-Whitney U), menangani nilai kembar."""
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    urut = np.argsort(skor, kind="mergesort")
    peringkat = np.empty(len(skor), dtype=np.float64)
    s = np.asarray(skor)[urut]
    i = 0
    while i < len(s):                       # peringkat rata-rata untuk nilai kembar
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        peringkat[urut[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((peringkat[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def f1_anomali(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    pres = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pres * rec / (pres + rec) if pres + rec else 0.0
    return {"f1_anomali": f1, "presisi": pres, "recall": rec, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def auroc_kondisi(y: np.ndarray, skor: np.ndarray, no: np.ndarray) -> float:
    """Skor satu kondisi (30 s) = skor jendela tertinggi di kondisi itu."""
    u = np.unique(no)
    return auroc(np.array([y[no == k][0] for k in u]), np.array([skor[no == k].max() for k in u]))


# ---- model -----------------------------------------------------------------------------
def buat_model(c_masuk: int, c_keluar: int, dropout: float):
    import torch.nn as nn

    class AE(nn.Module):
        """Encoder: 400 -> 200 -> 100 -> 50 langkah; laten 8 kanal x 50. Decoder: kebalikannya."""

        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv1d(c_masuk, 16, 7, stride=2, padding=3), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv1d(32, 8, 3, padding=1),                      # bottleneck
            )
            self.dec = nn.Sequential(
                nn.ConvTranspose1d(8, 32, 4, stride=2, padding=1), nn.ReLU(),
                nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
                nn.ConvTranspose1d(16, 16, 4, stride=2, padding=1), nn.ReLU(),
                nn.Conv1d(16, c_keluar, 7, padding=3),
            )

        def forward(self, x):
            return self.dec(self.enc(x))

    return AE()


# ---- persiapan masukan ------------------------------------------------------------------
def siapkan(X: np.ndarray, idx: np.ndarray, j: dict, kanal: list[int], konteks: bool,
            mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Normalisasi per kanal (mu/sd dari data latih normal) + kanal konteks opsional.

    Konteks = keadaan kipas dan lampu FISIK (diketahui pengendali dari perintahnya sendiri);
    keadaan pompa TIDAK pernah dimasukkan karena itulah yang dideteksi.
    """
    z = (X[:, kanal] - mu[None, :, None]) / sd[None, :, None]
    if konteks:
        k = np.broadcast_to((j["kipas"][idx] > 0).astype(np.float32)[:, None, None], (len(idx), 1, z.shape[2]))
        l = np.broadcast_to((j["lampu_fisik"][idx] > 0).astype(np.float32)[:, None, None], (len(idx), 1, z.shape[2]))
        z = np.concatenate([z, k, l], axis=1)
    return z.astype(np.float32)


def skor_rekonstruksi(model, Z: np.ndarray, c_sinyal: int) -> np.ndarray:
    import torch

    model.eval()
    hasil = []
    with torch.no_grad():
        for i in range(0, len(Z), 256):
            x = torch.from_numpy(Z[i:i + 256])
            r = model(x)
            hasil.append(((r - x[:, :c_sinyal]) ** 2).mean(dim=(1, 2)).numpy())
    return np.concatenate(hasil)


# ---- latih dan nilai ---------------------------------------------------------------------
def latih_dan_prediksi(a) -> tuple[np.ndarray, np.ndarray | None, dict]:
    import torch

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    j = data.jendela()
    il, iv, iu = data.bagian("latih"), data.bagian("val"), data.bagian("uji")
    kanal = {"both": [0, 1], "bus": [0], "arus": [1]}[a.kanal]
    c_sinyal = len(kanal)

    # HANYA jendela normal latih yang dipakai untuk melatih AE (kekeliruan yang dihindari di panduan)
    il_normal = il[j["y"][il] == 0]
    mu = j["X"][il_normal][:, kanal].mean(axis=(0, 2))
    sd = j["X"][il_normal][:, kanal].std(axis=(0, 2)) + 1e-6

    Z_lat = siapkan(j["X"][il_normal], il_normal, j, kanal, a.konteks, mu, sd)
    model = buat_model(Z_lat.shape[1], c_sinyal, a.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    gen = torch.Generator().manual_seed(a.seed)
    Xt = torch.from_numpy(Z_lat)
    riwayat = []
    for ep in range(a.epoch):
        model.train()
        urut = torch.randperm(len(Xt), generator=gen)
        total = 0.0
        for i in range(0, len(Xt), a.batch):
            x = Xt[urut[i:i + a.batch]]
            loss = ((model(x) - x[:, :c_sinyal]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        riwayat.append(total / len(Xt))

    # ambang: persentil 95 skor jendela normal latih (tanpa melihat anomali / validasi / uji)
    skor_lat = skor_rekonstruksi(model, Z_lat, c_sinyal)
    ambang = float(np.percentile(skor_lat, 95))

    Z_val = siapkan(j["X"][iv], iv, j, kanal, a.konteks, mu, sd)
    skor_val = skor_rekonstruksi(model, Z_val, c_sinyal)
    pred_val = (skor_val > ambang).astype(np.int64)
    y_val = j["y"][iv]

    # rincian AUROC per keadaan kipas x lampu fisik (untuk analisis kesalahan)
    per_keadaan = {}
    for kp in (0, 160):
        for lp in (0, 255):
            m = (j["kipas"][iv] == kp) & (j["lampu_fisik"][iv] == lp)
            per_keadaan[f"kipas{kp}_lampu{lp}"] = round(auroc_kondisi(y_val[m], skor_val[m], j["no"][iv][m]), 4)

    # latensi satu jendela (batch 1) di CPU
    model.eval()
    x1 = torch.from_numpy(Z_val[:1])
    with torch.no_grad():
        for _ in range(20):
            model(x1)
        t0 = time.perf_counter()
        for _ in range(200):
            model(x1)
    latensi_ms = (time.perf_counter() - t0) / 200 * 1000

    tambahan = {
        "konfigurasi": {"kanal": a.kanal, "konteks": a.konteks, "epoch": a.epoch, "lr": a.lr,
                        "weight_decay": a.weight_decay, "dropout": a.dropout, "batch": a.batch},
        "ulangan": {k: v.tolist() for k, v in zip(("latih", "val", "uji"), data.ulangan_split())},
        "n_latih_normal": int(len(il_normal)), "n_val": int(len(iv)),
        "ambang": ambang,
        "auroc_jendela": round(auroc(y_val, skor_val), 4),
        "auroc_kondisi": round(auroc_kondisi(y_val, skor_val, j["no"][iv]), 4),
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in f1_anomali(y_val, pred_val).items()},
        "auroc_kondisi_per_keadaan": per_keadaan,
        "n_parameter": int(sum(p.numel() for p in model.parameters())),
        "latensi_ms": round(latensi_ms, 3),
        "loss_latih_akhir": round(riwayat[-1], 5),
    }

    # ukuran ONNX (KB), bila paket onnx tersedia
    try:
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        out_dir = AKAR / "results" / NIM
        out_dir.mkdir(parents=True, exist_ok=True)
        jalur_onnx = out_dir / "_model.onnx"
        torch.onnx.export(model, x1, str(jalur_onnx), input_names=["x"], output_names=["r"],
                          dynamic_axes={"x": {0: "n"}, "r": {0: "n"}}, dynamo=False)
        tambahan["ukuran_onnx_kb"] = round(jalur_onnx.stat().st_size / 1024, 1)
    except Exception as e:  # onnx belum terpasang: tidak menggagalkan run
        tambahan["ukuran_onnx_kb"] = None
        tambahan["catatan_onnx"] = type(e).__name__

    pred_test = None
    if a.final:
        Z_uji = siapkan(j["X"][iu], iu, j, kanal, a.konteks, mu, sd)
        pred_test = (skor_rekonstruksi(model, Z_uji, c_sinyal) > ambang).astype(np.int64)
    return pred_val, pred_test, tambahan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    ap.add_argument("--kanal", choices=["both", "bus", "arus"], default="both")
    ap.add_argument("--tanpa-konteks", dest="konteks", action="store_false",
                    help="ablation: tanpa keadaan kipas & lampu fisik sebagai masukan (bawaan: dengan konteks)")
    ap.add_argument("--epoch", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    # konfigurasi utama dipilih dari validasi Sprint 1 (lihat catatan.md): 2 kanal + konteks
    nama = "utama" if (a.kanal == "both" and a.konteks) else f"ablation-{a.kanal}{'' if a.konteks else '-tanpa-konteks'}"
    t0 = time.time()
    pred_val, pred_test, tambahan = latih_dan_prediksi(a)
    durasi = time.time() - t0

    out = AKAR / "results" / NIM
    sub = out / nama / f"seed{a.seed}"
    sub.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", pred_val)
    (sub / "tambahan.json").write_text(json.dumps({"seed": a.seed, **tambahan}, indent=2) + "\n", encoding="utf-8")

    perintah = [sys.executable, "-m", "src.common.evaluate", "--nim", NIM, "--model", JALUR, "--seed", str(a.seed),
                "--pred-val", str(out / "pred_val.npy"), "--durasi", f"{durasi:.1f}", "--perangkat", a.perangkat,
                "--out", str(sub / "metrics.json")]
    if pred_test is not None:
        np.save(out / "pred_test.npy", pred_test)
        perintah += ["--pred-test", str(out / "pred_test.npy")]
    subprocess.run(perintah, check=True, cwd=AKAR)
    # metrics.json utama (yang diperiksa Local Runner) = run konfigurasi utama terakhir
    if nama == "utama":
        (out / "metrics.json").write_text((sub / "metrics.json").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[{nama} seed {a.seed}] AUROC kondisi {tambahan['auroc_kondisi']} | AUROC jendela "
          f"{tambahan['auroc_jendela']} | F1 anomali {tambahan['f1_anomali']} | {durasi:.0f} s")


if __name__ == "__main__":
    main()
