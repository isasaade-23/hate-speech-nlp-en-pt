"""Bayesian hyperparameter search (Optuna TPE) for the product model tfidf_logreg.

Two stages, both leakage-safe (vectorizer fit on TRAIN only, test untouched):

  1. Text-normalization ablation (--stage ablation): a small fixed grid over cleaning
     profile (light/heavy), strip_accents, binary TF and sublinear TF, holding the
     baseline hyperparameters fixed. Answers "does normalization/encoding help?".
  2. Bayesian search (--stage tune): TPE over TF-IDF (n-gram ranges, min_df,
     max_features, sublinear/binary) + LogReg (C, penalty, class_weight) + the
     text-normalization choices, optimizing val macro-F1 with the same tuned-threshold
     protocol as train.py. SQLite storage so runs are resumable.

The test set is only touched by --stage final, once, for the chosen config.

Usage (from repo root, venv python):
    python scripts/tune_tfidf_logreg.py --stage ablation
    python scripts/tune_tfidf_logreg.py --stage tune --trials 80
    python scripts/tune_tfidf_logreg.py --stage final
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion

from hsc.clean import clean_series
from hsc.config import resolve
from hsc.evaluate import best_threshold
from hsc.utils import get_logger

log = get_logger("hsc.tune")

POLICY = "strict"
SEED = 42
OUT_DIR = Path(resolve("reports/tables"))
STUDY_DB = Path(resolve("logs")) / "optuna_tfidf_logreg.db"
STUDY_NAME = f"tfidf_logreg_{POLICY}"

# Cleaning variants: keyed presets built over the raw `text` column. `light` must
# reproduce the corpus text_clean (the current baseline).
PROFILES = {
    "light": {
        "lowercase": False, "url_token": "<url>", "user_token": "<user>",
        "strip_rt": True, "keep_emoji": True, "demojize": True,
        "strip_hashtag_symbol": True,
    },
    "heavy": {
        "lowercase": True, "url_token": "<url>", "user_token": "<user>",
        "strip_rt": True, "keep_emoji": False, "demojize": True,
        "strip_hashtag_symbol": True, "strip_punct_noise": True,
    },
}


def load_data():
    df = pd.read_parquet(resolve("data/processed") / f"corpus_{POLICY}.parquet")
    tr = df[df["split"] == "train"]
    va = df[df["split"] == "val"]
    te = df[df["split"] == "test"]
    return tr, va, te


_CLEAN_CACHE: dict[tuple[str, int], pd.Series] = {}


def cleaned(part: pd.DataFrame, profile_key: str) -> np.ndarray:
    key = (profile_key, id(part))
    if key not in _CLEAN_CACHE:
        _CLEAN_CACHE[key] = clean_series(part["text"].values, PROFILES[profile_key])
    return _CLEAN_CACHE[key].values


def build_union(p: dict) -> FeatureUnion:
    parts = []
    if p.get("use_word", True):
        parts.append(("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=tuple(p["word_ngram"]),
            min_df=p["word_min_df"],
            max_features=p["word_max_features"],
            sublinear_tf=p["sublinear_tf"],
            binary=p["binary"],
            strip_accents=p["strip_accents"],
            lowercase=True,
        )))
    if p.get("use_char", True):
        parts.append(("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=tuple(p["char_ngram"]),
            min_df=p["char_min_df"],
            max_features=p["char_max_features"],
            sublinear_tf=p["sublinear_tf"],
            binary=p["binary"],
            strip_accents=p["strip_accents"],
            lowercase=True,
        )))
    return FeatureUnion(parts)


BASELINE = {
    "profile": "light",
    "word_ngram": [1, 2], "word_min_df": 3, "word_max_features": 50000,
    "char_ngram": [3, 5], "char_min_df": 3, "char_max_features": 50000,
    "sublinear_tf": True, "binary": False, "strip_accents": None,
    "C": 1.0, "penalty": "l2", "class_weight": "balanced",
}


def evaluate_params(p: dict, tr, va) -> dict:
    """Fit on train, tuned-threshold macro-F1 on val (same protocol as train.py)."""
    t0 = time.time()
    vec = build_union(p)
    Xtr = vec.fit_transform(cleaned(tr, p["profile"]))
    Xva = vec.transform(cleaned(va, p["profile"]))
    est = LogisticRegression(
        C=p["C"], penalty=p["penalty"], class_weight=p["class_weight"],
        solver="liblinear", max_iter=1000, random_state=SEED,
    )
    est.fit(Xtr, tr["label"].values)
    val_score = est.predict_proba(Xva)[:, 1]
    thr = best_threshold(va["label"].values, val_score)
    f1 = f1_score(va["label"].values, (val_score >= thr).astype(int),
                  average="macro", zero_division=0)
    return {"val_macro_f1": float(f1), "threshold": float(thr),
            "n_features": int(Xtr.shape[1]), "fit_seconds": round(time.time() - t0, 1)}


def stage_ablation(tr, va):
    """Normalization/encoding grid around the baseline, one factor at a time."""
    variants = {
        "baseline_light": {},
        "clean_heavy": {"profile": "heavy"},
        "strip_accents": {"strip_accents": "unicode"},
        "binary_tf": {"binary": True},
        "no_sublinear": {"sublinear_tf": False},
        "heavy+accents": {"profile": "heavy", "strip_accents": "unicode"},
    }
    rows = []
    for name, delta in variants.items():
        p = {**BASELINE, **delta}
        r = evaluate_params(p, tr, va)
        rows.append({"variant": name, **r})
        log.info("%-16s val macro-F1=%.4f thr=%.3f feats=%d (%.0fs)",
                 name, r["val_macro_f1"], r["threshold"], r["n_features"], r["fit_seconds"])
    out = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    out.to_csv(OUT_DIR / "tuning_normalization_ablation.csv", index=False)
    print(out.to_string(index=False))


def stage_tune(tr, va, n_trials: int):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        p = {
            "profile": trial.suggest_categorical("profile", ["light", "heavy"]),
            "strip_accents": trial.suggest_categorical("strip_accents", [None, "unicode"]),
            "word_ngram": json.loads(trial.suggest_categorical(
                "word_ngram", ["[1, 1]", "[1, 2]", "[1, 3]"])),
            "word_min_df": trial.suggest_int("word_min_df", 1, 5),
            "word_max_features": trial.suggest_categorical(
                "word_max_features", [30000, 50000, 100000, 200000]),
            "char_ngram": json.loads(trial.suggest_categorical(
                "char_ngram", ["[2, 4]", "[3, 5]", "[2, 5]", "[3, 6]"])),
            "char_min_df": trial.suggest_int("char_min_df", 1, 5),
            "char_max_features": trial.suggest_categorical(
                "char_max_features", [30000, 50000, 100000, 200000]),
            "sublinear_tf": trial.suggest_categorical("sublinear_tf", [True, False]),
            "binary": trial.suggest_categorical("binary", [True, False]),
            "C": trial.suggest_float("C", 0.05, 50.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2", "l1"]),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        }
        r = evaluate_params(p, tr, va)
        trial.set_user_attr("threshold", r["threshold"])
        trial.set_user_attr("n_features", r["n_features"])
        return r["val_macro_f1"]

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_DB.as_posix()}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
        load_if_exists=True,
    )
    # Seed the search with the known-good baseline so TPE starts from it.
    if not study.trials:
        study.enqueue_trial({
            "profile": "light", "strip_accents": None,
            "word_ngram": "[1, 2]", "word_min_df": 3, "word_max_features": 50000,
            "char_ngram": "[3, 5]", "char_min_df": 3, "char_max_features": 50000,
            "sublinear_tf": True, "binary": False,
            "C": 1.0, "penalty": "l2", "class_weight": "balanced",
        })
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    log.info("BEST val macro-F1=%.4f (trial %d) params=%s",
             best.value, best.number, best.params)
    rows = [{"trial": t.number, "val_macro_f1": t.value, **t.params}
            for t in study.trials if t.value is not None]
    pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False).to_csv(
        OUT_DIR / "tuning_trials.csv", index=False)
    print(json.dumps({"best_val_macro_f1": best.value, "params": best.params}, indent=2))


def stage_cv(tr, va, top_k: int = 10):
    """Re-rank the study's top-k trials (plus baseline) by 5-fold stratified CV over
    train+val. A single val split rewards selection overfitting (80 trials picking on
    the same 5.6k rows); CV mean is the stabler selection criterion."""
    import optuna
    from sklearn.model_selection import StratifiedKFold

    study = optuna.load_study(study_name=STUDY_NAME,
                              storage=f"sqlite:///{STUDY_DB.as_posix()}")
    done = [t for t in study.trials if t.value is not None]
    done.sort(key=lambda t: t.value, reverse=True)
    seen, cands = set(), [("baseline", BASELINE)]
    for t in done[: top_k * 2]:
        key = json.dumps(t.params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        p = {**BASELINE, **t.params,
             "word_ngram": json.loads(t.params["word_ngram"]),
             "char_ngram": json.loads(t.params["char_ngram"])}
        cands.append((f"trial_{t.number}", p))
        if len(cands) >= top_k + 1:
            break

    pool = pd.concat([tr, va], ignore_index=True)
    y = pool["label"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, params in cands:
        texts = clean_series(pool["text"].values, PROFILES[params["profile"]]).values
        scores = []
        for tr_idx, va_idx in skf.split(texts, y):
            vec = build_union(params)
            Xtr = vec.fit_transform(texts[tr_idx])
            est = LogisticRegression(
                C=params["C"], penalty=params["penalty"],
                class_weight=params["class_weight"],
                solver="liblinear", max_iter=1000, random_state=SEED,
            )
            est.fit(Xtr, y[tr_idx])
            s = est.predict_proba(vec.transform(texts[va_idx]))[:, 1]
            thr = best_threshold(y[va_idx], s)
            scores.append(f1_score(y[va_idx], (s >= thr).astype(int),
                                   average="macro", zero_division=0))
        mean, std = float(np.mean(scores)), float(np.std(scores))
        rows.append({"config": name, "cv_macro_f1_mean": round(mean, 4),
                     "cv_macro_f1_std": round(std, 4)})
        log.info("%-10s CV macro-F1 = %.4f ± %.4f", name, mean, std)
    out = pd.DataFrame(rows).sort_values("cv_macro_f1_mean", ascending=False)
    out.to_csv(OUT_DIR / "tuning_cv_rerank.csv", index=False)
    print(out.to_string(index=False))


def stage_final(tr, va, te):
    """Evaluate the study's best params on test, ONCE, next to the baseline."""
    import optuna

    study = optuna.load_study(study_name=STUDY_NAME,
                              storage=f"sqlite:///{STUDY_DB.as_posix()}")
    bp = dict(study.best_trial.params)
    p = {**BASELINE, **bp,
         "word_ngram": json.loads(bp["word_ngram"]),
         "char_ngram": json.loads(bp["char_ngram"])}

    rows = []
    for name, params in [("baseline", BASELINE), ("tuned", p)]:
        vec = build_union(params)
        Xtr = vec.fit_transform(cleaned(tr, params["profile"]))
        est = LogisticRegression(
            C=params["C"], penalty=params["penalty"], class_weight=params["class_weight"],
            solver="liblinear", max_iter=1000, random_state=SEED,
        )
        est.fit(Xtr, tr["label"].values)
        val_score = est.predict_proba(vec.transform(cleaned(va, params["profile"])))[:, 1]
        thr = best_threshold(va["label"].values, val_score)
        te_score = est.predict_proba(vec.transform(cleaned(te, params["profile"])))[:, 1]
        te_f1 = f1_score(te["label"].values, (te_score >= thr).astype(int),
                         average="macro", zero_division=0)
        va_f1 = f1_score(va["label"].values, (val_score >= thr).astype(int),
                         average="macro", zero_division=0)
        rows.append({"config": name, "val_macro_f1": round(float(va_f1), 4),
                     "test_macro_f1": round(float(te_f1), 4),
                     "threshold": round(float(thr), 4)})
        log.info("%s: val=%.4f test=%.4f thr=%.3f", name, va_f1, te_f1, thr)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "tuning_final_test.csv", index=False)
    print(out.to_string(index=False))
    print("\nbest params:", json.dumps(bp, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["ablation", "tune", "cv", "final"], required=True)
    ap.add_argument("--trials", type=int, default=60)
    args = ap.parse_args()

    tr, va, te = load_data()
    log.info("train=%d val=%d test=%d [%s]", len(tr), len(va), len(te), POLICY)
    if args.stage == "ablation":
        stage_ablation(tr, va)
    elif args.stage == "tune":
        stage_tune(tr, va, args.trials)
    elif args.stage == "cv":
        stage_cv(tr, va)
    else:
        stage_final(tr, va, te)


if __name__ == "__main__":
    main()
