"""Alavancas para o recall de ódio em português, sem GPU.

O produto servido hoje recupera 31,6% do ódio em PT. Três caminhos testados no
MESMO teste v5 (fatia PT, 4.748 linhas), com limiar sempre ajustado na
VALIDAÇÃO e o teste tocado uma vez por configuração:

  A. limiar por idioma no stack atual (nada muda no modelo)
  B. modelo dedicado ao PT (TF-IDF + LogReg treinado só em PT: o vocabulário
     não é diluído pelo inglês, que é 2/3 do corpus)
  C. B com o limiar ajustado na validação PT

O roteamento por idioma é viável no produto: o site e a extensão já detectam
idioma antes de classificar.

Uso: python scripts/pt_boost.py
Saída: reports/tables/pt_boost_v5.csv + models/pt_logreg_strict_s42/model.joblib
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import FeatureUnion

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
GLOBAL_THR = 0.3371778039000763   # limiar do bundle servido


def best_threshold(y_true, y_score) -> float:
    """Mesmo protocolo do pipeline: varre quantis e maximiza macro-F1 na validação."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score, dtype=float)
    cands = np.unique(np.quantile(y_score, np.linspace(0.02, 0.98, 97)))
    best_t, best_f = 0.5, -1.0
    for t in cands:
        f = f1_score(y_true, (y_score >= t).astype(int), average="macro", zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


def scores(y_true, y_score, thr: float, nome: str) -> dict:
    yp = (np.asarray(y_score) >= thr).astype(int)
    return {"config": nome, "limiar": round(float(thr), 4),
            "macro_f1": round(f1_score(y_true, yp, average="macro", zero_division=0), 4),
            "recall_hate": round(recall_score(y_true, yp, zero_division=0), 4),
            "precision_hate": round(precision_score(y_true, yp, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_true, y_score), 4)}


def build_vectorizer() -> FeatureUnion:
    """Mesma receita do tfidf_logreg do estudo (word 1-2 + char_wb 3-5)."""
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3,
                                 max_features=150000, sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 max_features=150000, sublinear_tf=True, lowercase=True)),
    ])


def main() -> None:
    corpus = pd.read_parquet(ROOT / "data/processed/corpus_strict.parquet")
    pt = corpus[corpus.language == "pt"]
    tr, va, te = (pt[pt.split == s] for s in ("train", "val", "test"))
    print(f"PT: train={len(tr)} val={len(va)} test={len(te)} "
          f"| ódio no treino={int(tr.label.sum())} ({tr.label.mean():.1%})")

    rows = []

    # --- A. stack atual na fatia PT, limiar global vs limiar de idioma
    pred_va = pd.read_parquet(ROOT / "reports/predictions/stack_strict_s42_val.parquet")
    pred_te = pd.read_parquet(ROOT / "reports/predictions/stack_strict_s42_test.parquet")
    sva, ste = pred_va[pred_va.language == "pt"], pred_te[pred_te.language == "pt"]
    rows.append(scores(ste.y_true, ste.y_score, GLOBAL_THR, "stack servido (limiar global)"))
    thr_pt = best_threshold(sva.y_true, sva.y_score)
    rows.append(scores(ste.y_true, ste.y_score, thr_pt, "stack + limiar de idioma"))

    # --- B/C. modelo dedicado ao PT
    vec = build_vectorizer()
    Xtr = vec.fit_transform(tr.text_clean)          # fit SÓ no treino
    Xva, Xte = vec.transform(va.text_clean), vec.transform(te.text_clean)
    print(f"vocabulário PT: {Xtr.shape[1]} dims")
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                             solver="liblinear", random_state=SEED)
    clf.fit(Xtr, tr.label.values)
    s_va = clf.predict_proba(Xva)[:, 1]
    s_te = clf.predict_proba(Xte)[:, 1]
    rows.append(scores(te.label.values, s_te, 0.5, "modelo PT (limiar 0,5)"))
    thr_b = best_threshold(va.label.values, s_va)
    rows.append(scores(te.label.values, s_te, thr_b, "modelo PT + limiar na validação"))

    out = pd.DataFrame(rows)
    dest = ROOT / "reports/tables/pt_boost_v5.csv"
    out.to_csv(dest, index=False)
    print("\n=== fatia PT do teste v5 (4.748 linhas) ===")
    print(out.to_string(index=False))
    print("->", dest)

    # guarda o modelo PT para o produto poder rotear por idioma
    mdir = ROOT / "models/pt_logreg_strict_s42"
    mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"kind": "single", "vectorizer": vec, "estimator": clf,
                 "threshold": float(thr_b), "language": "pt",
                 "config": {"name": "pt_logreg", "policy": "strict", "seed": SEED}},
                mdir / "model.joblib")
    json.dump({"model_id": "pt_logreg_strict_s42", "config": "pt_logreg",
               "family": "classical", "policy": "strict", "seed": SEED,
               "language": "pt", "n_train": int(len(tr)),
               "threshold": round(float(thr_b), 4),
               "test_pt": rows[-1]}, open(mdir / "metrics.json", "w"), indent=2)
    print(f"-> {mdir / 'model.joblib'} "
          f"({(mdir / 'model.joblib').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
