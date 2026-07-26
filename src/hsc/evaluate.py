"""Evaluation metrics shared by every model family (classical and neural), so the
comparison is apples-to-apples.

Primary metric: macro-F1. Also: per-class precision/recall (recall-on-hate is the
ethically important number), ROC-AUC, PR-AUC, bootstrap CI on macro-F1, McNemar's test
for paired model comparison, and per-slice (language / source) breakdowns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

POS = 1  # hate


def classification_metrics(y_true, y_pred, y_score=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    out = {
        "n": int(len(y_true)),
        "n_hate": int((y_true == POS).sum()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "precision_hate": float(p[1]),
        "recall_hate": float(r[1]),
        "f1_hate": float(f[1]),
        "precision_nothate": float(p[0]),
        "recall_nothate": float(r[0]),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        y_score = np.asarray(y_score)
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    return out


def bootstrap_macro_f1_ci(y_true, y_pred, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lo, hi


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """Paired comparison of two models on the same test set (exact binomial)."""
    from scipy.stats import binomtest

    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true
    b01 = int(np.sum(a_ok & ~b_ok))  # a right, b wrong
    b10 = int(np.sum(~a_ok & b_ok))  # a wrong, b right
    n = b01 + b10
    p = binomtest(min(b01, b10), n, 0.5).pvalue if n > 0 else 1.0
    return {"a_only_correct": b01, "b_only_correct": b10, "p_value": float(p)}


def confusion(y_true, y_pred) -> list[list[int]]:
    return confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()


def breakdown(df: pd.DataFrame, y_true_col: str, y_pred_col: str, by: str) -> pd.DataFrame:
    """Per-slice macro-F1 and recall-on-hate. `by` is e.g. 'language' or 'source_dataset'."""
    rows = []
    for key, g in df.groupby(by):
        m = classification_metrics(g[y_true_col], g[y_pred_col])
        rows.append(
            {
                by: key,
                "n": m["n"],
                "n_hate": m["n_hate"],
                "macro_f1": round(m["macro_f1"], 4),
                "recall_hate": round(m["recall_hate"], 4),
            }
        )
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True)
