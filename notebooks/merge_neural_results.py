"""Merge Colab neural results back into the local repo, then you can re-run the reports.

After training on Colab and downloading `hsc_neural_results.zip` to the repo root:

    .venv/Scripts/python.exe notebooks/merge_neural_results.py hsc_neural_results.zip

It copies the neural metrics + per-example predictions into place and merges the neural
registry entries into models/registry.json WITHOUT touching the classical entries. Then
`hsc report` / `hsc analyze` / `hsc bias` include the neural models automatically.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(zip_path: str) -> None:
    zp = Path(zip_path)
    if not zp.is_absolute():
        zp = ROOT / zp
    if not zp.exists():
        raise SystemExit(f"not found: {zp}")

    (ROOT / "reports" / "metrics").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "predictions").mkdir(parents=True, exist_ok=True)

    neural_reg: dict = {}
    n_metrics = n_preds = 0
    with zipfile.ZipFile(zp) as z:
        for name in z.namelist():
            if name.startswith("reports/metrics/") and name.endswith(".json"):
                (ROOT / name).write_bytes(z.read(name))
                n_metrics += 1
            elif name.startswith("reports/predictions/") and name.endswith(".parquet"):
                (ROOT / name).write_bytes(z.read(name))
                n_preds += 1
            elif name.endswith("registry_neural.json") or name.endswith("registry.json"):
                neural_reg = json.loads(z.read(name).decode("utf-8"))

    # merge registry: neural entries win only for their own ids; classical untouched
    reg_path = ROOT / "models" / "registry.json"
    reg = json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.exists() else {}
    added = [k for k in neural_reg if reg.get(k) != neural_reg[k]]
    reg.update({k: v for k, v in neural_reg.items() if v.get("family") == "neural"})
    reg_path.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"merged: {n_metrics} metrics, {n_preds} prediction files, {len(added)} registry entries")
    print("now run:  hsc report  &&  hsc analyze  &&  hsc bias")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python notebooks/merge_neural_results.py <hsc_neural_results.zip>")
    main(sys.argv[1])
