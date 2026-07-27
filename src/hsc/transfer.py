"""Fase 9 headline experiment: cross-domain and cross-lingual generalization.

Trains on one slice of the corpus and tests on a disjoint slice to measure domain shift
and zero-shot language transfer directly, instead of only reporting in-distribution
numbers. Two axes:

  * cross-domain  (same language, different source): tweets_ip <-> memotion, both EN.
  * cross-lingual (zero-shot, different language): EN sources -> pt_fortuna, and back.

Fair protocol, identical to the main pipeline: features fit on the TRAIN slice only;
decision threshold tuned on a held-out VAL slice within the train source; evaluated once
on the TEST source. Slices reuse the frozen split column, and train/test sources are
different datasets, so there is no leakage across the transfer boundary.

The point of contrast is the feature family. TF-IDF word features do not share
vocabulary across languages, so they collapse cross-lingually; frozen multilingual
sentence embeddings (SBERT) place EN and PT in one space and should transfer. Running
both makes that the result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hsc.config import resolve
from hsc.evaluate import best_threshold, classification_metrics
from hsc.features import build_features
from hsc.models.classical import build_estimator
from hsc.predictions import predict_scores
from hsc.utils import ensure_dir, get_logger, set_all_seeds

log = get_logger("hsc.transfer")

TEXT_COL = "text_clean"

# Feature families compared. SBERT is included only when sentence-transformers is present.
TFIDF_FEATURES = {
    "type": "tfidf",
    "word": {"ngram_range": [1, 2], "min_df": 3, "max_features": 30000, "sublinear_tf": True},
    "char": {"analyzer": "char_wb", "ngram_range": [3, 5], "min_df": 3, "max_features": 30000, "sublinear_tf": True},
}
SBERT_FEATURES = {"type": "embeddings"}

LOGREG = {"type": "logreg", "params": {"class_weight": "balanced", "max_iter": 2000, "C": 1.0}}

# Each experiment: label, sources used to TRAIN, sources used to TEST, kind.
EXPERIMENTS = [
    {"name": "EN_tweets->EN_memes", "train": ["tweets_ip"], "test": ["memotion"], "kind": "cross-domain"},
    {"name": "EN_memes->EN_tweets", "train": ["memotion"], "test": ["tweets_ip"], "kind": "cross-domain"},
    {"name": "EN_all->PT (zero-shot)", "train": ["tweets_ip", "memotion", "multioff"], "test": ["pt_fortuna"], "kind": "cross-lingual"},
    {"name": "PT->EN_tweets (zero-shot)", "train": ["pt_fortuna"], "test": ["tweets_ip"], "kind": "cross-lingual"},
]


def _sbert_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("sentence_transformers") is not None


def _slice(corpus: pd.DataFrame, sources: list[str], splits: list[str]) -> pd.DataFrame:
    return corpus[corpus["source_dataset"].isin(sources) & corpus["split"].isin(splits)]


def _run_one(corpus: pd.DataFrame, exp: dict, features_cfg: dict, feat_name: str, seed: int) -> dict:
    set_all_seeds(seed)
    tr = _slice(corpus, exp["train"], ["train"])
    va = _slice(corpus, exp["train"], ["val"])
    te = _slice(corpus, exp["test"], ["test"])
    if len(va) < 20:  # tiny train sources (e.g. multioff alone) — fall back to train for threshold
        va = tr

    vec = build_features(features_cfg)
    Xtr = vec.fit_transform(tr[TEXT_COL].values)
    est = build_estimator(LOGREG, seed)
    est.fit(Xtr, tr["label"].values)

    thr = best_threshold(va["label"].values, predict_scores(est, vec.transform(va[TEXT_COL].values)))
    y_score = predict_scores(est, vec.transform(te[TEXT_COL].values))
    y_pred = (np.asarray(y_score) >= thr).astype(int)
    m = classification_metrics(te["label"].values, y_pred, y_score)
    return {
        "experiment": exp["name"],
        "kind": exp["kind"],
        "features": feat_name,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "test_hate_rate": round(float(np.mean(te["label"].values)), 4),
        "macro_f1": round(m["macro_f1"], 4),
        "recall_hate": round(m["recall_hate"], 4),
        "precision_hate": round(m["precision_hate"], 4),
        "roc_auc": round(m.get("roc_auc", float("nan")), 4),
        "threshold": round(thr, 4),
    }


def run_transfer(policy: str = "strict", seed: int = 42) -> pd.DataFrame:
    corpus = pd.read_parquet(resolve("data/processed") / f"corpus_{policy}.parquet")
    families = [("tfidf", TFIDF_FEATURES)]
    if _sbert_available():
        families.append(("sbert", SBERT_FEATURES))
    else:
        log.warning("sentence-transformers not installed — running TF-IDF transfer only")

    rows = []
    for feat_name, fcfg in families:
        for exp in EXPERIMENTS:
            # skip experiments whose sources are absent under this policy (e.g. multioff not in strict)
            present = set(corpus["source_dataset"].unique())
            if not (set(exp["test"]) & present) or not (set(exp["train"]) & present):
                continue
            row = _run_one(corpus, exp, fcfg, feat_name, seed)
            row["policy"] = policy
            rows.append(row)
            log.info(
                "[%s|%s] %s: macroF1=%.3f recall_hate=%.3f (n_test=%d)",
                feat_name, policy, exp["name"], row["macro_f1"], row["recall_hate"], row["n_test"],
            )
    df = pd.DataFrame(rows)
    out_dir = ensure_dir(resolve("reports/tables"))
    df.to_csv(out_dir / f"transfer_{policy}.csv", index=False)
    log.info("wrote reports/tables/transfer_%s.csv (%d rows)", policy, len(df))
    return df


def run_all(seed: int = 42) -> None:
    frames = [run_transfer(p, seed) for p in ("strict", "broad")]
    df = pd.concat(frames, ignore_index=True)
    cols = ["policy", "kind", "experiment", "features", "n_train", "n_test",
            "test_hate_rate", "macro_f1", "recall_hate", "roc_auc", "threshold"]
    print("\n===== CROSS-DOMAIN / CROSS-LINGUAL TRANSFER =====")
    print(df[cols].to_string(index=False))
