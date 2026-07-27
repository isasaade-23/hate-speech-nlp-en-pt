"""Per-example predictions: the substrate for paired significance (McNemar) and
calibration analysis.

Aggregate metrics in reports/metrics/*.json cannot answer "do models A and B disagree
significantly?" or "are the scores calibrated?" — those need the actual per-row
(y_true, y_score, y_pred). This module persists them during training and can
reconstruct them for any registered model from its frozen joblib + the frozen corpus,
so the six TF-IDF models trained earlier get predictions without retraining.

Alignment guarantee: within one label policy every model is evaluated on the same
frozen split, so rows align by the corpus `id`. McNemar and any paired test join on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hsc.config import resolve
from hsc.utils import ensure_dir, get_logger, read_json

log = get_logger("hsc.predictions")

PRED_COLS = ["id", "language", "source_dataset", "y_true", "y_score", "y_pred"]


def predict_scores(estimator, X):
    """Positive-class scores for ROC/PR/calibration; falls back to decision_function."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    raise AttributeError("estimator exposes neither predict_proba nor decision_function")


def predictions_path(model_id: str, split: str):
    return resolve("reports/predictions") / f"{model_id}_{split}.parquet"


def save_predictions(model_id: str, split: str, frame: pd.DataFrame) -> None:
    ensure_dir(resolve("reports/predictions"))
    frame[PRED_COLS].to_parquet(predictions_path(model_id, split), index=False)


def build_prediction_frame(part: pd.DataFrame, y_score, threshold: float) -> pd.DataFrame:
    y_score = np.asarray(y_score, dtype=float)
    return pd.DataFrame(
        {
            "id": part["id"].values,
            "language": part["language"].values,
            "source_dataset": part["source_dataset"].values,
            "y_true": part["label"].values.astype(int),
            "y_score": y_score,
            "y_pred": (y_score >= threshold).astype(int),
        }
    )


def _reconstruct(model_id: str, split: str) -> pd.DataFrame:
    """Regenerate predictions from a registered model's frozen joblib + frozen corpus."""
    import joblib

    reg = read_json(resolve("models") / "registry.json")
    if model_id not in reg:
        raise KeyError(f"{model_id} not in registry")
    entry = reg[model_id]
    bundle = joblib.load(resolve(entry["path"]))
    vec, est = bundle["vectorizer"], bundle["estimator"]
    threshold = float(bundle.get("threshold", entry.get("threshold", 0.5)))
    text_col = bundle.get("config", {}).get("text_column", "text_clean")

    corpus = pd.read_parquet(resolve("data/processed") / f"corpus_{entry['policy']}.parquet")
    part = corpus[corpus["split"] == split]
    y_score = predict_scores(est, vec.transform(part[text_col].values))
    frame = build_prediction_frame(part, y_score, threshold)
    save_predictions(model_id, split, frame)
    log.info("reconstructed predictions for %s [%s] (%d rows)", model_id, split, len(frame))
    return frame


def load_predictions(model_id: str, split: str = "test") -> pd.DataFrame:
    """Load saved per-example predictions; reconstruct from the model if absent."""
    path = predictions_path(model_id, split)
    if path.exists():
        return pd.read_parquet(path)
    return _reconstruct(model_id, split)
