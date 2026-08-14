"""Calibration probe for the served demo model (tfidf_logreg_strict_s42).

Protocol (anti-leakage):
  - calibrators are FIT on the validation split only;
  - test is touched once per method, for reporting;
  - macro-F1 must not move: monotone calibration preserves ranking, the decision
    threshold is mapped through the fitted calibrator.

Methods: Platt (sigmoid on the raw score) and isotonic regression.
Outputs reports/tables/calibration_demo.csv and a console summary.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score

sys.path.insert(0, "src")

from hsc.config import resolve  # noqa: E402
from hsc.evaluate import calibration_curve_bins  # noqa: E402
from hsc.predictions import load_predictions  # noqa: E402
from hsc.utils import read_json  # noqa: E402

MODEL_ID = "tfidf_logreg_strict_s42"


def report(tag: str, y_true, p, threshold: float) -> dict:
    cal = calibration_curve_bins(y_true, p, n_bins=10)
    y_pred = (p >= threshold).astype(int)
    return {
        "method": tag,
        "threshold": round(float(threshold), 4),
        "ece": round(cal["ece"], 4),
        "mce": round(cal["mce"], 4),
        "brier": round(brier_score_loss(y_true, np.clip(p, 0, 1)), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
    }


def main() -> None:
    val = load_predictions(MODEL_ID, "val")
    test = load_predictions(MODEL_ID, "test")
    thr = float(read_json(resolve("models") / "registry.json")[MODEL_ID]["threshold"])

    yv, sv = val["y_true"].to_numpy(int), val["y_score"].to_numpy(float)
    yt, st = test["y_true"].to_numpy(int), test["y_score"].to_numpy(float)

    rows = [report("raw", yt, st, thr)]

    platt = LogisticRegression(C=1e6, solver="lbfgs")
    platt.fit(sv.reshape(-1, 1), yv)
    thr_p = float(platt.predict_proba([[thr]])[0, 1])
    rows.append(report("platt", yt, platt.predict_proba(st.reshape(-1, 1))[:, 1], thr_p))

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(sv, yv)
    # isotonic is flat in places: pick the calibrated threshold that reproduces the
    # raw decision on val exactly, then reuse it on test
    pv = iso.predict(sv)
    raw_pos = sv >= thr
    thr_i = float(pv[raw_pos].min()) if raw_pos.any() else 1.0
    rows.append(report("isotonic", yt, iso.predict(st), thr_i))

    df = pd.DataFrame(rows)
    out = resolve("reports/tables") / "calibration_demo.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
