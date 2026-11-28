"""Periksa artefak jalur AE: skema metrics.json dan data_hash terhadap data/MANIFEST.json.

    python models/123450039/periksa_artefak.py      # menulis results/123450039/periksa_artefak.md

Untuk setiap results/123450039/**/metrics.json: kunci wajib (metrik, seed, data_hash, perangkat,
durasi_detik) bertipe benar, semua metrik berupa angka, dan data_hash sama dengan sidik yang
dihitung ulang dari berkas di MANIFEST. Juga memeriksa sha256 label.csv terhadap MANIFEST.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.common import manifest, split  # noqa: E402

NIM = Path(__file__).resolve().parent.name
WAJIB = {"metrik": dict, "seed": int, "data_hash": str, "perangkat": str, "durasi_detik": (int, float)}

m = manifest.baca()
hash_manifest = manifest.data_hash(f"split:{split.SEED_SPLIT}".encode())
label = next(AKAR / b for b in m["berkas"] if b.endswith("label.csv"))
sha_label = hashlib.sha256(label.read_bytes()).hexdigest()

baris = ["# Pemeriksaan artefak jalur AE (123450039)", "",
         f"- data_hash dihitung ulang dari MANIFEST: `{hash_manifest}`",
         f"- sha256 label.csv: `{sha_label}` "
         f"({'cocok' if sha_label == m.get('sha256_label_csv') else 'TIDAK cocok'} dengan MANIFEST)", "",
         "| Berkas | seed | skema | data_hash | val_acc | val_macro_f1 |", "|---|---|---|---|---|---|"]
gagal = 0
for f in sorted((AKAR / "results" / NIM).rglob("metrics.json")):
    d = json.loads(f.read_text("utf-8"))
    skema = all(isinstance(d.get(k), t) for k, t in WAJIB.items()) and \
        all(isinstance(v, (int, float)) for v in d.get("metrik", {}).values())
    cocok = d.get("data_hash") == hash_manifest
    gagal += (not skema) + (not cocok)
    baris.append(f"| {f.relative_to(AKAR)} | {d.get('seed')} | {'OK' if skema else 'GAGAL'} | "
                 f"{'cocok' if cocok else 'BEDA'} | {d['metrik'].get('val_acc')} | {d['metrik'].get('val_macro_f1')} |")
baris += ["", "Semua artefak lolos." if gagal == 0 else f"{gagal} pemeriksaan gagal."]
out = AKAR / "results" / NIM / "periksa_artefak.md"
out.write_text("\n".join(baris) + "\n", encoding="utf-8")
print("\n".join(baris))
sys.exit(1 if gagal else 0)
