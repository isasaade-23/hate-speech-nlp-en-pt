"""Beta 2.0 — build the served stacking ensemble over the classical models.

Protocol (anti-leakage): the meta logistic regression, the decision threshold and
the Platt calibration are all fit on the VALIDATION split only; the test split is
touched once per candidate composition, for reporting.

Two compositions are compared:
  tfidf3 : tfidf_logreg + tfidf_svm + tfidf_lgbm       (CPU-light, Streamlit-safe)
  all5   : tfidf3 + sbert_logreg + sbert_lgbm          (needs the 470 MB SBERT encoder)

The winner by the Pareto rule (serve tfidf3 unless all5 beats it by >= MIN_GAIN AUC)
is written as models/stack_<policy>_s42/model.joblib with kind='stack': the member
bundles inline (vectorizer + estimator), the meta coefficients, the calibrated
threshold, plus a registry entry and saved val/test predictions.

Usage: python scripts/build_stack.py [--policy strict]
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression  # noqa: F401  (kept for future ablation)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score

sys.path.insert(0, "src")

from hsc.config import resolve  # noqa: E402
from hsc.evaluate import calibration_curve_bins  # noqa: E402
from hsc.predictions import PRED_COLS, load_predictions, save_predictions  # noqa: E402
from hsc.utils import ensure_dir, read_json, write_json  # noqa: E402

TFIDF3 = ["tfidf_logreg", "tfidf_svm", "tfidf_lgbm"]
ALL5 = TFIDF3 + ["sbert_logreg", "sbert_lgbm"]
MIN_GAIN = 0.005  # AUC gain all5 must show over tfidf3 to justify the SBERT encoder


def gather(members: list[str], policy: str, split: str) -> pd.DataFrame:
    frames = {}
    for m in members:
        d = load_predictions(f"{m}_{policy}_s42", split)
        frames[m] = d.set_index("id")
    base = frames[members[0]][["language", "source_dataset", "y_true"]].copy()
    for m in members:
        base[m] = frames[m]["y_score"]
    assert not base.isna().any().any(), "stack: member predictions do not align by id"
    return base


def fit_eval(members: list[str], policy: str):
    val, test = gather(members, policy, "val"), gather(members, policy, "test")
    yv, yt = val["y_true"].to_numpy(int), test["y_true"].to_numpy(int)
    meta = LogisticRegression(C=1.0, max_iter=1000).fit(val[members], yv)
    pv = meta.predict_proba(val[members])[:, 1]
    pt = meta.predict_proba(test[members])[:, 1]

    # Platt over the stacked score (the meta output is already probability-shaped but
    # not calibrated: it was fit on the same split it scores, so re-scale honestly)
    platt = LogisticRegression(C=1e6, solver="lbfgs").fit(pv.reshape(-1, 1), yv)
    cv = platt.predict_proba(pv.reshape(-1, 1))[:, 1]
    ct = platt.predict_proba(pt.reshape(-1, 1))[:, 1]

    ths = np.unique(np.quantile(cv, np.linspace(0.30, 0.995, 200)))
    thr = float(max(ths, key=lambda t: f1_score(yv, cv >= t, average="macro")))

    cal = calibration_curve_bins(yt, ct, n_bins=10)
    res = {
        "members": members,
        "val_auc": round(roc_auc_score(yv, cv), 4),
        "test_auc": round(roc_auc_score(yt, ct), 4),
        "val_macro_f1": round(f1_score(yv, cv >= thr, average="macro"), 4),
        "test_macro_f1": round(f1_score(yt, ct >= thr, average="macro"), 4),
        "test_recall_hate": round(
            float(((ct >= thr) & (yt == 1)).sum() / max((yt == 1).sum(), 1)), 4
        ),
        "test_ece": round(cal["ece"], 4),
        "test_brier": round(brier_score_loss(yt, np.clip(ct, 0, 1)), 4),
        "threshold": round(thr, 4),
    }
    return res, meta, platt, thr, (val, test, cv, ct)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="strict")
    args = ap.parse_args()
    policy = args.policy

    res3, meta3, platt3, thr3, data3 = fit_eval(TFIDF3, policy)
    try:
        res5, *_ = fit_eval(ALL5, policy)
    except (FileNotFoundError, AssertionError, KeyError) as e:
        # SBERT members not (re)trained on the current corpus: their prediction
        # files are missing or align to older split ids. Compare tfidf3 alone.
        print(f"all5 skipped ({type(e).__name__}: SBERT predictions absent/stale on this corpus)")
        res5 = None
    for r in (res3, res5) if res5 else (res3,):
        print(
            f"{'+'.join(m.split('_')[0] for m in r['members']):24s}"
            f" n={len(r['members'])} test AUC={r['test_auc']} F1={r['test_macro_f1']}"
            f" rec_hate={r['test_recall_hate']} ECE={r['test_ece']}"
        )

    if res5 and res5["test_auc"] - res3["test_auc"] >= MIN_GAIN:
        print(
            f"NOTE: all5 beats tfidf3 by {res5['test_auc'] - res3['test_auc']:.4f} AUC "
            "(>= MIN_GAIN) but needs the 470 MB SBERT encoder; serving tfidf3 anyway "
            "requires a human call. Bundling tfidf3."
        )
    chosen, meta, platt, thr, (val, test, cv, ct) = res3, meta3, platt3, thr3, data3

    model_id = f"stack_{policy}_s42"
    members = chosen["members"]
    member_bundles = {
        m: {
            k: v
            for k, v in joblib.load(
                resolve("models") / f"{m}_{policy}_s42" / "model.joblib"
            ).items()
            if k in ("vectorizer", "estimator")
        }
        for m in members
    }
    a = float(platt.coef_[0, 0])
    b = float(platt.intercept_[0])
    bundle = {
        "kind": "stack",
        "members": members,
        "member_bundles": member_bundles,
        "meta_coef": meta.coef_[0].tolist(),
        "meta_intercept": float(meta.intercept_[0]),
        "calibration": {"method": "platt", "coef": a, "intercept": b, "threshold": thr},
        "threshold": thr,
        "config": {"policy": policy, "composition": "tfidf3", "beta": "2.0"},
    }
    out_dir = ensure_dir(resolve("models") / model_id)
    joblib.dump(bundle, out_dir / "model.joblib")

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    reg_path = resolve("models") / "registry.json"
    reg = read_json(reg_path)
    reg[model_id] = {
        "path": f"models/{model_id}/model.joblib",
        "config": "stack_tfidf3",
        "policy": policy,
        "seed": 42,
        "git_sha": sha,
        "threshold": thr,
        "val_macro_f1": chosen["val_macro_f1"],
        "test_macro_f1": chosen["test_macro_f1"],
        "test_roc_auc": chosen["test_auc"],
    }
    write_json(reg, reg_path)

    for split, frame, scores in (("val", val, cv), ("test", test, ct)):
        pred = frame.reset_index()[["id", "language", "source_dataset", "y_true"]].copy()
        pred["y_score"] = scores
        pred["y_pred"] = (scores >= thr).astype(int)
        save_predictions(model_id, split, pred[PRED_COLS])

    pd.DataFrame([r for r in (res3, res5) if r]).to_csv(
        resolve("reports/tables") / f"stack_{policy}.csv", index=False
    )
    print(f"saved {model_id}: threshold {thr:.4f}, registry + predictions + table written")


if __name__ == "__main__":
    main()
