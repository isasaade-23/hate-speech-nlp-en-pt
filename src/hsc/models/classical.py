"""Classical estimators built from config. All expose predict_proba so evaluation
(ROC-AUC, PR-AUC, calibration) is uniform across model families.
"""

from __future__ import annotations


def build_estimator(cfg_model: dict, seed: int):
    t = cfg_model["type"]
    params = dict(cfg_model.get("params", {}))

    if t == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(random_state=seed, **params)

    if t == "svm":
        # LinearSVC has no predict_proba; wrap for calibrated probabilities.
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import LinearSVC

        base = LinearSVC(random_state=seed, **params)
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)

    if t == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(random_state=seed, **params)

    if t == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            random_state=seed,
            eval_metric="logloss",
            tree_method="hist",
            **params,
        )

    raise ValueError(f"unknown model type: {t}")
