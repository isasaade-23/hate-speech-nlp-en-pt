"""Fit Platt calibration for the served demo model and store it in the bundle.

Fit on the validation split only (anti-leakage). Stores {coef, intercept,
threshold} under bundle["calibration"]; the raw score, raw threshold and the
registry are untouched, so every existing analysis stays reproducible.
Applied at inference time by HateClassifier when the key is present.

Updates both bundle copies: the research repo and the Streamlit space.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "src")

from hsc.config import resolve  # noqa: E402
from hsc.predictions import load_predictions  # noqa: E402
from hsc.utils import read_json  # noqa: E402

MODEL_ID = "tfidf_logreg_strict_s42"
SPACE_MODELS = Path("C:/Users/Renato/hate-speech-space/models")


def main() -> None:
    val = load_predictions(MODEL_ID, "val")
    thr = float(read_json(resolve("models") / "registry.json")[MODEL_ID]["threshold"])
    sv = val["y_score"].to_numpy(float).reshape(-1, 1)
    yv = val["y_true"].to_numpy(int)

    platt = LogisticRegression(C=1e6, solver="lbfgs")
    platt.fit(sv, yv)
    a = float(platt.coef_[0, 0])
    b = float(platt.intercept_[0])
    thr_cal = float(1.0 / (1.0 + np.exp(-(a * thr + b))))
    cal = {"method": "platt", "coef": a, "intercept": b, "threshold": thr_cal}
    print(f"platt: p = sigmoid({a:.4f}*s + {b:.4f}); threshold {thr:.4f} -> {thr_cal:.4f}")

    for models_dir in (resolve("models"), SPACE_MODELS):
        path = models_dir / MODEL_ID / "model.joblib"
        bundle = joblib.load(path)
        bundle["calibration"] = cal
        joblib.dump(bundle, path)
        print(f"updated {path}")


if __name__ == "__main__":
    main()
