"""Diagnóstico do buraco de recall PT do stack v5 (strict), sem re-treino.

Perguntas:
1. Limiar único global prejudica o PT? -> ajusta limiar POR IDIOMA na validação
   (mesmo protocolo best-threshold do pipeline) e mede o efeito no teste.
2. Os falsos negativos PT estão perto do limiar (calibração) ou no fundo da
   escala (representação)?

Uso: python scripts/pt_gap.py
Saída: reports/tables/pt_gap_stack_v5_strict.csv (agregados; nenhum texto)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
GLOBAL_THR = 0.3371778039000763  # bundle do stack (calibrado, ajustado na validação)


def best_threshold(y_true, y_score):
    """Cópia do protocolo de hsc.evaluate (macro-F1 na validação)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    cands = np.unique(np.quantile(y_score, np.linspace(0.02, 0.98, 97)))
    best_t, best_f = 0.5, -1.0
    for t in cands:
        f = f1_score(y_true, (y_score >= t).astype(int), average="macro", zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


def block(g, thr):
    yp = (g.y_score >= thr).astype(int)
    return {"macro_f1": round(f1_score(g.y_true, yp, average="macro", zero_division=0), 4),
            "recall_hate": round(recall_score(g.y_true, yp, zero_division=0), 4),
            "precision_hate": round(
                float((yp[g.y_true == 1] == 1).sum() / max(yp.sum(), 1)), 4)}


def main() -> None:
    val = pd.read_parquet(ROOT / "reports/predictions/stack_strict_s42_val.parquet")
    test = pd.read_parquet(ROOT / "reports/predictions/stack_strict_s42_test.parquet")

    rows = []
    for lang in ("en", "pt"):
        va, te = val[val.language == lang], test[test.language == lang]
        thr_lang = best_threshold(va.y_true, va.y_score)
        g = block(te, GLOBAL_THR)
        l = block(te, thr_lang)
        rows.append({"idioma": lang, "n_test": len(te),
                     "limiar_global": round(GLOBAL_THR, 4),
                     "limiar_por_idioma (val)": round(thr_lang, 4),
                     "macro_f1 global": g["macro_f1"], "macro_f1 idioma": l["macro_f1"],
                     "recall_hate global": g["recall_hate"],
                     "recall_hate idioma": l["recall_hate"],
                     "precision_hate global": g["precision_hate"],
                     "precision_hate idioma": l["precision_hate"]})
    out = pd.DataFrame(rows)

    # anatomia dos falsos negativos PT no teste (limiar global)
    pt = test[test.language == "pt"]
    fn = pt[(pt.y_true == 1) & (pt.y_score < GLOBAL_THR)]
    q = fn.y_score.quantile([0.25, 0.5, 0.75]).round(4).to_dict()
    perto = (fn.y_score >= GLOBAL_THR - 0.1).mean()
    fundo = (fn.y_score < 0.1).mean()
    anat = pd.DataFrame([
        {"metrica": "n_hate_pt_teste", "valor": int((pt.y_true == 1).sum())},
        {"metrica": "n_fn_pt (limiar global)", "valor": int(len(fn))},
        {"metrica": "score FN p25/p50/p75",
         "valor": f"{q[0.25]}/{q[0.5]}/{q[0.75]}"},
        {"metrica": "% FN a menos de 0,1 do limiar (calibracao)", "valor": round(perto, 4)},
        {"metrica": "% FN com score < 0,1 (representacao)", "valor": round(fundo, 4)},
    ])
    fn_src = (fn.groupby("source_dataset").size() / pt[pt.y_true == 1]
              .groupby(pt.source_dataset).size()).round(4).rename("taxa_fn").reset_index()

    dest = ROOT / "reports/tables/pt_gap_stack_v5_strict.csv"
    with open(dest, "w", encoding="utf-8", newline="") as fh:
        out.to_csv(fh, index=False)
        fh.write("\n")
        anat.to_csv(fh, index=False)
        fh.write("\n")
        fn_src.to_csv(fh, index=False)

    print(out.to_string(index=False))
    print()
    print(anat.to_string(index=False))
    print()
    print("taxa de FN por fonte PT:")
    print(fn_src.to_string(index=False))
    print("->", dest)


if __name__ == "__main__":
    main()
