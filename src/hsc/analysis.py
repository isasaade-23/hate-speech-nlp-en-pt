"""Fase 9 post-hoc analysis over trained models: paired significance (McNemar) and
probability calibration. Operates on saved per-example predictions, so it never
retrains and stays consistent with the frozen splits.

Comparisons are made WITHIN a label policy only: strict and broad have different test
rows and different ground truth, so a paired test across them is meaningless. Within a
policy all models share the frozen test split, so predictions align by corpus `id`.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hsc.config import resolve
from hsc.evaluate import calibration_curve_bins, mcnemar
from hsc.predictions import load_predictions
from hsc.utils import ensure_dir, get_logger, read_json

log = get_logger("hsc.analysis")


def _registry_by_policy() -> dict[str, list[str]]:
    reg = read_json(resolve("models") / "registry.json")
    out: dict[str, list[str]] = {}
    for mid, entry in reg.items():
        out.setdefault(entry["policy"], []).append(mid)
    for policy in out:
        out[policy].sort()
    return out


def _holm(pvals: list[float]) -> list[bool]:
    """Holm-Bonferroni: reject in ascending p order while p_(k) <= alpha/(m-k)."""
    alpha = 0.05
    m = len(pvals)
    order = np.argsort(pvals)
    reject = [False] * m
    for k, i in enumerate(order):
        if pvals[i] <= alpha / (m - k):
            reject[i] = True
        else:
            break
    return reject


def run_significance(split: str = "test") -> pd.DataFrame:
    """Pairwise McNemar across every model pair within each policy. Writes a long-form
    table (one row per pair) with Holm-corrected significance."""
    by_policy = _registry_by_policy()
    rows = []
    for policy, models in by_policy.items():
        preds = {m: load_predictions(m, split).set_index("id") for m in models}
        pairs, pvals = [], []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a, b = models[i], models[j]
                joined = preds[a][["y_true", "y_pred"]].join(
                    preds[b][["y_pred"]], rsuffix="_b", how="inner"
                )
                mc = mcnemar(joined["y_true"], joined["y_pred"], joined["y_pred_b"])
                pairs.append((a, b, mc))
                pvals.append(mc["p_value"])
        rejects = _holm(pvals) if pvals else []
        for (a, b, mc), pval, rej in zip(pairs, pvals, rejects):
            rows.append(
                {
                    "policy": policy,
                    "model_a": a,
                    "model_b": b,
                    "a_only_correct": mc["a_only_correct"],
                    "b_only_correct": mc["b_only_correct"],
                    "p_value": round(pval, 5),
                    "significant_holm": bool(rej),
                }
            )
    df = pd.DataFrame(rows)
    out_dir = ensure_dir(resolve("reports/tables"))
    df.to_csv(out_dir / f"mcnemar_{split}.csv", index=False)
    log.info("wrote reports/tables/mcnemar_%s.csv (%d pairs)", split, len(df))
    return df


def run_calibration(split: str = "test", n_bins: int = 10) -> pd.DataFrame:
    """ECE / MCE / Brier per model + one reliability diagram per policy."""
    by_policy = _registry_by_policy()
    out_dir = ensure_dir(resolve("reports/tables"))
    fig_dir = ensure_dir(resolve("reports/figures"))
    rows = []
    for policy, models in by_policy.items():
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect")
        for m in models:
            p = load_predictions(m, split)
            cal = calibration_curve_bins(p["y_true"], p["y_score"], n_bins=n_bins)
            rows.append(
                {
                    "model_id": m,
                    "policy": policy,
                    "ece": cal["ece"],
                    "mce": cal["mce"],
                    "brier": cal["brier"],
                }
            )
            pts = [(b["confidence"], b["accuracy"]) for b in cal["bins"] if b["count"] > 0]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", ms=4, lw=1.2, label=m.replace(f"_{policy}_s42", ""))
        ax.set_xlabel("mean predicted confidence")
        ax.set_ylabel("empirical accuracy")
        ax.set_title(f"Reliability — {policy}")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(fig_dir / f"calibration_{policy}.png", dpi=130)
        plt.close(fig)
        log.info("wrote reports/figures/calibration_%s.png", policy)
    df = pd.DataFrame(rows).sort_values(["policy", "ece"]).reset_index(drop=True)
    df.to_csv(out_dir / f"calibration_{split}.csv", index=False)
    log.info("wrote reports/tables/calibration_%s.csv (%d models)", split, len(df))
    return df


def run_all(split: str = "test") -> None:
    sig = run_significance(split)
    cal = run_calibration(split)
    print("\n===== McNEMAR (paired, Holm-corrected) =====")
    print(sig.to_string(index=False) if len(sig) else "(no pairs)")
    print("\n===== CALIBRATION =====")
    print(cal.to_string(index=False))
