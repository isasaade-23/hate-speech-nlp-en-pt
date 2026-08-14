"""Fase 11 — inference wrapper for the product API.

Loads a trained bundle (vectorizer + estimator + tuned threshold) from the registry,
runs the SAME cleaning as training (light profile) and the language-detection front-end,
and returns a structured prediction. Used by api/service.py and demo/app.py.
"""

from __future__ import annotations

import numpy as np

from hsc.clean import clean_text
from hsc.config import data_config, resolve
from hsc.langid import detect as detect_lang
from hsc.utils import read_json


def _resolve_model_id(model_id: str | None) -> str:
    reg_path = resolve("models") / "registry.json"
    if not reg_path.exists():
        raise FileNotFoundError("models/registry.json not found — train a model first.")
    reg = read_json(reg_path)
    if not reg:
        raise RuntimeError("registry is empty — train a model first.")
    if model_id:
        if model_id not in reg:
            raise KeyError(f"model_id {model_id} not in registry")
        return model_id
    # Default: best test macro-F1 among LOCALLY-SERVABLE models. Neural entries live in the
    # registry for the leaderboard, but their weights are a Colab HF dir, not a local
    # model.joblib — serving them needs torch/GPU infra, so inference defaults to the best
    # classical model that is actually loadable here (the product recommendation).
    servable = {k: v for k, v in reg.items() if (resolve("models") / k / "model.joblib").exists()}
    if not servable:
        raise FileNotFoundError(
            "no locally-servable model.joblib found — train a classical model first "
            "(neural weights are not served locally)."
        )
    return max(servable, key=lambda k: servable[k].get("test_macro_f1", 0.0))


def _score(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


class HateClassifier:
    def __init__(self, model_id: str | None = None):
        import joblib

        self.model_id = _resolve_model_id(model_id)
        bundle = joblib.load(resolve("models") / self.model_id / "model.joblib")
        self.vectorizer = bundle["vectorizer"]
        self.estimator = bundle["estimator"]
        self.threshold = float(bundle.get("threshold", 0.5))
        self.config = bundle.get("config", {})
        # optional Platt calibration (fit on val): served scores become probabilities
        cal = bundle.get("calibration")
        self._cal = (float(cal["coef"]), float(cal["intercept"])) if cal else None
        if cal:
            self.threshold = float(cal["threshold"])
        self._profile = data_config()["clean"]["profiles"]["light"]

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict]:
        cleaned = [clean_text(t, self._profile) for t in texts]
        X = self.vectorizer.transform(cleaned)
        scores = np.asarray(_score(self.estimator, X))
        if self._cal is not None:
            a, b = self._cal
            scores = 1.0 / (1.0 + np.exp(-(a * scores + b)))
        out = []
        for text, s in zip(texts, scores):
            code, conf = detect_lang(text)
            label = int(s >= self.threshold)
            out.append(
                {
                    "text": text,
                    "label": "hate" if label else "not_hate",
                    "score": float(s),
                    "language": {"detected": code, "confidence": round(float(conf), 4)},
                    "model_version": self.model_id,
                }
            )
        return out


_CLASSIFIER: HateClassifier | None = None


def get_classifier(model_id: str | None = None) -> HateClassifier:
    """Process-wide singleton so the API loads the model once at startup."""
    global _CLASSIFIER
    if _CLASSIFIER is None or (model_id and model_id != _CLASSIFIER.model_id):
        _CLASSIFIER = HateClassifier(model_id)
    return _CLASSIFIER
