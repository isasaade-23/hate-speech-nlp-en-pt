"""Unified, config-driven training entrypoint.

Same interface for classical and (later) neural models so the comparison is fair:
identical frozen splits, identical metrics, identical seeds. Anti-leakage is enforced
here: the vectorizer is fit ONLY on train rows.

Outputs per run:
  - reports/metrics/<model_id>.json     (source of the paper's tables)
  - models/<model_id>/model.joblib + model_card.md
  - models/registry.json entry
  - MLflow run under logs/mlruns (if mlflow is installed)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from hsc.config import load_yaml, resolve
from hsc.evaluate import bootstrap_macro_f1_ci, breakdown, classification_metrics, confusion
from hsc.features.tfidf import build_tfidf
from hsc.models.classical import build_estimator
from hsc.utils import ensure_dir, get_logger, read_json, set_all_seeds, write_json

log = get_logger("hsc.train")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(resolve(".")), text=True
        ).strip()
    except Exception:
        return "nogit"


def _load_split_corpus(policy: str) -> pd.DataFrame:
    path = resolve("data/processed") / f"corpus_{policy}.parquet"
    df = pd.read_parquet(path)
    if "split" not in df.columns:
        raise RuntimeError(
            f"{path} has no 'split' column. Run `hsc split --policy {policy}` first (Fase 4)."
        )
    return df


def _predict_scores(estimator, X):
    """Positive-class scores for ROC/PR; falls back to decision_function."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return None


def train_from_config(config_path: str, policy_override: str | None = None) -> dict:
    cfg = load_yaml(config_path)
    policy = policy_override or cfg.get("policy", "strict")
    seed = int(cfg.get("seed", 42))
    text_col = cfg.get("text_column", "text_clean")
    set_all_seeds(seed)

    df = _load_split_corpus(policy)
    tr = df[df["split"] == "train"]
    va = df[df["split"] == "val"]
    te = df[df["split"] == "test"]
    log.info("train=%d val=%d test=%d [%s]", len(tr), len(va), len(te), policy)

    # Features: fit ON TRAIN ONLY
    vec = build_tfidf(cfg["features"])
    Xtr = vec.fit_transform(tr[text_col].values)
    est = build_estimator(cfg["model"], seed)
    est.fit(Xtr, tr["label"].values)

    model_id = f"{cfg['name']}_{policy}_s{seed}"
    result = {
        "model_id": model_id,
        "config": cfg["name"],
        "family": cfg.get("family", "classical"),
        "policy": policy,
        "seed": seed,
        "git_sha": _git_sha(),
        "n_train": int(len(tr)),
        "splits": {},
    }

    for name, part in [("val", va), ("test", te)]:
        X = vec.transform(part[text_col].values)
        y = part["label"].values
        y_pred = est.predict(X)
        y_score = _predict_scores(est, X)
        m = classification_metrics(y, y_pred, y_score)
        lo, hi = bootstrap_macro_f1_ci(y, y_pred, seed=seed)
        m["macro_f1_ci95"] = [round(lo, 4), round(hi, 4)]
        m["confusion"] = confusion(y, y_pred)

        pred_df = part[["language", "source_dataset", "label"]].copy()
        pred_df["pred"] = y_pred
        m["by_language"] = breakdown(pred_df, "label", "pred", "language").to_dict("records")
        m["by_source"] = breakdown(pred_df, "label", "pred", "source_dataset").to_dict("records")
        result["splits"][name] = m
        log.info(
            "%s [%s]: macro-F1=%.4f (CI %.3f-%.3f) recall_hate=%.4f",
            model_id, name, m["macro_f1"], lo, hi, m["recall_hate"],
        )

    _persist(result, cfg, vec, est, model_id)
    _mlflow_log(result, cfg)
    return result


def _persist(result: dict, cfg: dict, vec, est, model_id: str) -> None:
    import joblib

    metrics_dir = ensure_dir(resolve("reports/metrics"))
    write_json(result, metrics_dir / f"{model_id}.json")

    model_dir = ensure_dir(resolve("models") / model_id)
    joblib.dump({"vectorizer": vec, "estimator": est, "config": cfg}, model_dir / "model.joblib")
    _write_model_card(result, cfg, model_dir / "model_card.md")

    # registry
    reg_path = resolve("models") / "registry.json"
    reg = read_json(reg_path) if reg_path.exists() else {}
    reg[model_id] = {
        "path": f"models/{model_id}/model.joblib",
        "config": cfg["name"],
        "policy": result["policy"],
        "seed": result["seed"],
        "git_sha": result["git_sha"],
        "val_macro_f1": result["splits"]["val"]["macro_f1"],
        "test_macro_f1": result["splits"]["test"]["macro_f1"],
    }
    write_json(reg, reg_path)
    log.info("saved model + registry entry: %s", model_id)


def _write_model_card(result: dict, cfg: dict, path: Path) -> None:
    te = result["splits"]["test"]
    lines = [
        f"# Model card — {result['model_id']}",
        "",
        f"- Family: {result['family']}  |  Config: {result['config']}  |  Policy: {result['policy']}",
        f"- Seed: {result['seed']}  |  git: {result['git_sha']}  |  train rows: {result['n_train']}",
        "",
        "## Test metrics",
        f"- macro-F1: {te['macro_f1']:.4f} (95% CI {te['macro_f1_ci95']})",
        f"- recall (hate): {te['recall_hate']:.4f}  |  precision (hate): {te['precision_hate']:.4f}",
        f"- ROC-AUC: {te.get('roc_auc', float('nan')):.4f}  |  PR-AUC: {te.get('pr_auc', float('nan')):.4f}",
        "",
        "## Intended use & limitations",
        "Research classifier for EN/PT social-media hate speech. Probabilistic; not a",
        "moderation oracle. Training-data licenses restrict commercial use (see",
        "methodology/data_provenance.md). See methodology/limitations.md.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _mlflow_log(result: dict, cfg: dict) -> None:
    # Best-effort: a tracking hiccup must never kill a training run.
    try:
        import os

        # newer mlflow blocks the file store unless this opt-out is set
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow

        mlflow.set_tracking_uri((resolve("logs/mlruns")).as_uri())
        mlflow.set_experiment("hate-speech-classical")
        with mlflow.start_run(run_name=result["model_id"]):
            mlflow.log_params(
                {
                    "config": cfg["name"],
                    "family": result["family"],
                    "policy": result["policy"],
                    "seed": result["seed"],
                    "model_type": cfg["model"]["type"],
                }
            )
            for split, m in result["splits"].items():
                for k in ("macro_f1", "accuracy", "recall_hate", "precision_hate"):
                    mlflow.log_metric(f"{split}_{k}", m[k])
                if "roc_auc" in m:
                    mlflow.log_metric(f"{split}_roc_auc", m["roc_auc"])
    except Exception as e:  # pragma: no cover
        log.warning("mlflow logging skipped: %s", e)
