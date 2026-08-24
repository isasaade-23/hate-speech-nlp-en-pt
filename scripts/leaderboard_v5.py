"""Leaderboard justo do corpus v5: clássico x neural no MESMO corpus e teste.

Só entram modelos medidos no teste v5 (16.261 linhas, ou a fatia de idioma dele).
Modelos do estudo v1 ficam de fora: outro corpus, outro teste, não comparáveis.

Modelos de um idioma só (BERTimbau em PT, BERTweet em EN) são comparados na
fatia correspondente, nunca no total, porque o total deles não é o mesmo teste.

Uso: python scripts/leaderboard_v5.py
Saídas: reports/tables/leaderboard_v5_strict.csv
        reports/tables/leaderboard_v5_pt.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "reports/metrics"
PRED = ROOT / "reports/predictions"
N_TEST_V5 = 16261          # teste strict do corpus v5
N_TEST_EN, N_TEST_PT = 11513, 4748


def do_stack() -> dict:
    """O produto servido hoje: métricas direto das predições."""
    df = pd.read_parquet(PRED / "stack_strict_s42_test.parquet")
    row = {"modelo": "stack (produto atual)", "familia": "classico",
           "n_test": len(df),
           "macro_f1": round(f1_score(df.y_true, df.y_pred, average="macro"), 4),
           "recall_hate": round(recall_score(df.y_true, df.y_pred), 4)}
    for lang in ("en", "pt"):
        g = df[df.language == lang]
        row[f"macro_f1_{lang}"] = round(f1_score(g.y_true, g.y_pred, average="macro"), 4)
        row[f"recall_hate_{lang}"] = round(recall_score(g.y_true, g.y_pred), 4)
    vid = df[df.source_dataset == "vidgen"]
    row["macro_f1_vidgen"] = round(f1_score(vid.y_true, vid.y_pred, average="macro"), 4)
    return row


def do_json(path: Path) -> dict | None:
    m = json.load(open(path, encoding="utf-8"))
    t = m["splits"]["test"]
    # descarta o que não foi medido no teste v5 (ou na fatia de idioma dele)
    if t["n"] not in (N_TEST_V5, N_TEST_EN, N_TEST_PT):
        return None
    row = {"modelo": m["config"], "familia": m.get("family", "classico"),
           "n_test": t["n"], "macro_f1": round(t["macro_f1"], 4),
           "recall_hate": round(t["recall_hate"], 4)}
    for r in t.get("by_language", []):
        row[f"macro_f1_{r['language']}"] = r["macro_f1"]
        row[f"recall_hate_{r['language']}"] = r["recall_hate"]
    for r in t.get("by_source", []):
        if r["source_dataset"] == "vidgen":
            row["macro_f1_vidgen"] = r["macro_f1"]
    return row


def main() -> None:
    rows = [do_stack()]
    for p in sorted(MET.glob("*_strict_s42.json")):
        r = do_json(p)
        if r:
            rows.append(r)

    df = pd.DataFrame(rows).drop_duplicates(subset="modelo")
    bilingue = df[df.n_test == N_TEST_V5].sort_values("macro_f1", ascending=False)
    cols = ["modelo", "familia", "macro_f1", "recall_hate", "macro_f1_en",
            "recall_hate_en", "macro_f1_pt", "recall_hate_pt", "macro_f1_vidgen"]
    bilingue = bilingue[[c for c in cols if c in bilingue.columns]]
    bilingue.to_csv(ROOT / "reports/tables/leaderboard_v5_strict.csv", index=False)

    # ranking de PT: bilíngues na fatia PT + modelos só-PT no total deles
    pt_rows = []
    for r in rows:
        if r["n_test"] == N_TEST_V5 and "macro_f1_pt" in r:
            pt_rows.append({"modelo": r["modelo"], "escopo": "bilingue (fatia PT)",
                            "macro_f1_pt": r["macro_f1_pt"],
                            "recall_hate_pt": r["recall_hate_pt"]})
        elif r["n_test"] == N_TEST_PT:
            pt_rows.append({"modelo": r["modelo"], "escopo": "so PT",
                            "macro_f1_pt": r["macro_f1"],
                            "recall_hate_pt": r["recall_hate"]})
    pt = pd.DataFrame(pt_rows).sort_values("recall_hate_pt", ascending=False)
    pt.to_csv(ROOT / "reports/tables/leaderboard_v5_pt.csv", index=False)

    print("=== teste v5 strict completo (16.261) ===")
    print(bilingue.to_string(index=False))
    print("\n=== portugues (o gargalo do produto) ===")
    print(pt.to_string(index=False))
    # "tem resultado v5" = json existe E foi medido no teste v5 (o json do
    # estudo v1 pode existir com o mesmo nome, medido em outro teste)
    feitos = set(df.modelo)
    faltam = [n for n in ("bertimbau_pt", "xlmr_multilingual", "twitter_xlmr",
                          "bertweet_en", "lfm25_encoder") if n not in feitos]
    if faltam:
        print("\nainda sem resultado no teste v5:", ", ".join(faltam))


if __name__ == "__main__":
    main()
