"""Fatia adversarial do teste v5 (strict): macro-F1 por fonte para os modelos servíveis.

A fonte `vidgen` é o conjunto adversarial (frases escritas para enganar
classificadores) e responde ~1/3 do teste. Esta tabela é a metade "clássica" da
comparação justa clássico × neural; a outra metade vem do re-treino no Colab
(colab_neural_v5_painel). Os transformers v1 ficam de fora de propósito: foram
medidos em outro corpus/teste.

Uso: python scripts/adversarial_slice.py
Saída: reports/tables/adversarial_slice_v5_strict.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["hatebr", "hatexplain", "pt_fortuna", "toldbr", "tweets_ip", "vidgen"]
# sbert_* ficam de fora: métricas de corpus anterior (teste 7.649 linhas, sem
# vidgen/hatexplain), não comparáveis ao teste v5 de 16.261.
CLASSICAL = [
    "tfidf_logreg_strict_s42",
    "tfidf_svm_strict_s42",
    "tfidf_lgbm_strict_s42",
]


def row_from_metrics(mid: str) -> dict:
    m = json.load(open(ROOT / f"reports/metrics/{mid}.json", encoding="utf-8"))
    test = m["splits"]["test"]
    row = {"model": m["config"], "macro_f1_total": round(test["macro_f1"], 4),
           "recall_hate_total": round(test["recall_hate"], 4)}
    for r in test["by_source"]:
        row[f"f1_{r['source_dataset']}"] = r["macro_f1"]
    vid = next(r for r in test["by_source"] if r["source_dataset"] == "vidgen")
    row["recall_hate_vidgen"] = vid["recall_hate"]
    return row


def row_from_predictions(mid: str, label: str) -> dict:
    df = pd.read_parquet(ROOT / f"reports/predictions/{mid}_test.parquet")
    row = {"model": label,
           "macro_f1_total": round(f1_score(df.y_true, df.y_pred, average="macro"), 4),
           "recall_hate_total": round(recall_score(df.y_true, df.y_pred), 4)}
    for src, g in df.groupby("source_dataset"):
        row[f"f1_{src}"] = round(f1_score(g.y_true, g.y_pred, average="macro",
                                          zero_division=0), 4)
    vid = df[df.source_dataset == "vidgen"]
    row["recall_hate_vidgen"] = round(recall_score(vid.y_true, vid.y_pred), 4)
    row["n_vidgen"] = int(len(vid))
    row["n_test"] = int(len(df))
    return row


def main() -> None:
    rows = [row_from_predictions("stack_strict_s42", "stack (produto)")]
    rows += [row_from_metrics(mid) for mid in CLASSICAL]
    out = pd.DataFrame(rows)
    cols = (["model", "macro_f1_total", "recall_hate_total"]
            + [f"f1_{s}" for s in SOURCES] + ["recall_hate_vidgen", "n_vidgen", "n_test"])
    out = out[[c for c in cols if c in out.columns]]
    dest = ROOT / "reports/tables/adversarial_slice_v5_strict.csv"
    out.to_csv(dest, index=False)
    print(out.to_string(index=False))
    print("->", dest)


if __name__ == "__main__":
    main()
