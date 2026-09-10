"""Merge Colab neural results back into the local repo, then you can re-run the reports.

After training on Colab and downloading `hsc_neural_results.zip` to the repo root:

    .venv/Scripts/python.exe notebooks/merge_neural_results.py hsc_neural_results.zip

It copies the neural metrics + per-example predictions into place and merges the neural
registry entries into models/registry.json WITHOUT touching the classical entries. Then
`hsc report` / `hsc analyze` / `hsc bias` include the neural models automatically.

The merge always ends by reconciling the registry against reports/metrics/*.json, which
are the authority: a run whose zip lost registry_neural.json, or a partial policy run,
used to leave the registry quoting numbers from an older corpus while the metrics and the
predictions were already v5. Reconciling without a zip:

    .venv/Scripts/python.exe notebooks/merge_neural_results.py --resync
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


REG_PATH = ROOT / "models" / "registry.json"
# campos do registry que sao copia do que reports/metrics/<id>.json ja diz
FROM_METRICS = ("threshold", "val_macro_f1", "test_macro_f1")


def resync_from_metrics(verbose: bool = True) -> int:
    """Realinha as entradas neurais do registry com reports/metrics/. Devolve quantas mudaram.

    O registry e indice; o arquivo de metricas e o resultado da corrida. Quando os dois
    discordam, quem esta errado e o indice. Vale so para family == neural: as classicas
    sao escritas pelo proprio pipeline local, na mesma passada que gera as metricas.
    """
    if not REG_PATH.exists():
        return 0
    reg = json.loads(REG_PATH.read_text(encoding="utf-8"))
    mudou = 0
    for mid, entry in reg.items():
        if entry.get("family") != "neural":
            continue
        mpath = ROOT / "reports" / "metrics" / f"{mid}.json"
        if not mpath.exists():
            continue
        m = json.loads(mpath.read_text(encoding="utf-8"))
        novo = {
            "threshold": m.get("threshold"),
            "val_macro_f1": m["splits"]["val"]["macro_f1"],
            "test_macro_f1": m["splits"]["test"]["macro_f1"],
        }
        for campo in FROM_METRICS:
            velho = entry.get(campo)
            if novo[campo] is not None and velho != novo[campo]:
                if verbose:
                    print(f"  {mid}.{campo}: {velho} -> {novo[campo]}")
                entry[campo] = novo[campo]
                mudou += 1
    if mudou:
        REG_PATH.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    return mudou


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
    reg = json.loads(REG_PATH.read_text(encoding="utf-8")) if REG_PATH.exists() else {}
    added = [k for k in neural_reg if reg.get(k) != neural_reg[k]]
    reg.update({k: v for k, v in neural_reg.items() if v.get("family") == "neural"})
    REG_PATH.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"merged: {n_metrics} metrics, {n_preds} prediction files, {len(added)} registry entries")
    # rede de seguranca: o zip pode nao trazer o registry, ou trazer so uma politica
    n = resync_from_metrics()
    print(f"resync com reports/metrics: {n} campos corrigidos")
    print("now run:  hsc report  &&  hsc analyze  &&  hsc bias")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--resync":
        n = resync_from_metrics()
        print(f"resync com reports/metrics: {n} campos corrigidos")
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        raise SystemExit(
            "usage: python notebooks/merge_neural_results.py <hsc_neural_results.zip> | --resync"
        )
