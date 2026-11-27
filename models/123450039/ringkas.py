"""Rangkum hasil semua run AE (3 seed per konfigurasi) menjadi tabel rata-rata ± simpangan baku.

    python models/123450039/ringkas.py     # menulis results/123450039/ringkasan.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
NIM = Path(__file__).resolve().parent.name
OUT = AKAR / "results" / NIM
KOLOM = ["auroc_kondisi", "auroc_jendela", "f1_anomali", "presisi", "recall"]


def ms(v: list[float]) -> str:
    v = np.asarray(v, dtype=float)
    return f"{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0:.3f}"


baris = ["# Ringkasan hasil AE (validasi; uji belum disentuh)", "",
         "| Konfigurasi | n seed | AUROC kondisi | AUROC jendela | F1 anomali | presisi | recall | alarm palsu (FP) | parameter | latensi (ms) | ONNX (KB) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
keadaan = []
for kfg in sorted(p for p in OUT.iterdir() if p.is_dir()):
    runs = [json.loads((s / "tambahan.json").read_text()) for s in sorted(kfg.glob("seed*")) if (s / "tambahan.json").exists()]
    if not runs:
        continue
    r0 = runs[-1]
    baris.append(f"| {kfg.name} | {len(runs)} | " + " | ".join(ms([r[k] for r in runs]) for k in KOLOM)
                 + f" | {ms([r['fp'] for r in runs])} | {r0['n_parameter']} | {ms([r['latensi_ms'] for r in runs])} | {r0.get('ukuran_onnx_kb')} |")
    for nama in r0["auroc_kondisi_per_keadaan"]:
        keadaan.append((kfg.name, nama, ms([r["auroc_kondisi_per_keadaan"][nama] for r in runs])))

baris += ["", "## AUROC per kondisi menurut keadaan kipas × lampu fisik", "", "| Konfigurasi | Keadaan | AUROC kondisi |", "|---|---|---|"]
baris += [f"| {a} | {b} | {c} |" for a, b, c in keadaan]
(OUT / "ringkasan.md").write_text("\n".join(baris) + "\n", encoding="utf-8")
print("\n".join(baris))
