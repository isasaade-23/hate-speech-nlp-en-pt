"""Fase 11 — product model selection by Pareto trade-off.

Picking the served model is not "highest macro-F1". A product weighs quality (macro-F1,
recall-on-hate) against cost and risk: inference latency, on-disk size, probability
calibration (the API returns a score), identity-term bias, and — decisively — the data
license. This module assembles every axis into one table, computes the Pareto front among
the CPU-deployable candidates, and prints a recommendation.

Quality/calibration/bias come from the frozen reports (so neural models, whose weights
live on Colab, are compared on quality). Latency and size are measured live from the
local joblib models; neural rows are marked not-locally-deployable.

License reality: every current model is trained on the FULL corpus, which mixes
non-commercial sources — so all are research-only. A commercially-clean model must be
retrained on the commercial whitelist (labels.yaml: commercial_whitelist).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from hsc.config import labels_config, resolve
from hsc.utils import get_logger, read_json

log = get_logger("hsc.product")

# Novel texts (EN+PT) for latency timing — must NOT be in the embedding cache, so SBERT
# encoding time is measured honestly rather than served from cache.
_LATENCY_TEXTS = [
    "this brand new sentence has never appeared anywhere in the corpus xyzzy",
    "esta frase totalmente inédita nunca apareceu no corpus quux plugh",
    "another fresh unseen string to time the model foobar 12345",
    "mais uma sentença nova para medir a latência do modelo blargh 67890",
]


def _quality_rows() -> pd.DataFrame:
    rows = []
    for f in sorted(resolve("reports/metrics").glob("*.json")):
        r = read_json(f)
        te = r["splits"]["test"]
        rows.append(
            {
                "model_id": r["model_id"],
                "family": r.get("family", "?"),
                "policy": r.get("policy", "?"),
                "test_macro_f1": round(te["macro_f1"], 4),
                "recall_hate": round(te["recall_hate"], 4),
            }
        )
    return pd.DataFrame(rows)


def _calibration_map() -> dict:
    out = {}
    for pol in ("strict", "broad"):
        p = resolve("reports/tables") / f"calibration_test.csv"
        if p.exists():
            df = pd.read_csv(p)
            for _, r in df.iterrows():
                out[r["model_id"]] = round(float(r["ece"]), 4)
            break
    return out


def _bias_map() -> dict:
    """Mean identity-term FPR gap per model (over-flagging bias; higher = worse)."""
    out: dict[str, float] = {}
    for pol in ("strict", "broad"):
        p = resolve("reports/tables") / f"bias_identity_fpr_{pol}.csv"
        if p.exists():
            df = pd.read_csv(p)
            for mid, g in df.groupby("model_id"):
                out[mid] = round(float(g["fpr_gap"].mean()), 4)
    return out


def _size_and_latency(model_id: str) -> tuple[float | None, float | None, float | None]:
    """(size_mb, latency_p50_ms, latency_p95_ms) for a local joblib model, else NAs."""
    jl = resolve("models") / model_id / "model.joblib"
    if not jl.exists():
        return None, None, None
    import joblib

    size_mb = round(jl.stat().st_size / 1_048_576, 2)
    bundle = joblib.load(jl)
    vec, est = bundle["vectorizer"], bundle["estimator"]
    # honest SBERT timing: disable the per-text cache so we measure real encoding
    for attr in ("cache",):
        if hasattr(vec, attr):
            setattr(vec, attr, False)

    def infer(text):
        X = vec.transform([text])
        if hasattr(est, "predict_proba"):
            est.predict_proba(X)
        else:
            est.decision_function(X)

    infer(_LATENCY_TEXTS[0])  # warmup (loads SBERT encoder etc.)
    times = []
    for i in range(24):
        t = _LATENCY_TEXTS[i % len(_LATENCY_TEXTS)] + f" n{i}"
        t0 = time.perf_counter()
        infer(t)
        times.append((time.perf_counter() - t0) * 1000)
    return size_mb, round(float(np.percentile(times, 50)), 1), round(float(np.percentile(times, 95)), 1)


def _pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated set. Better = higher {f1, recall}, lower {ece, bias, latency, size}."""
    def dominates(a, b):
        ge = (
            a["test_macro_f1"] >= b["test_macro_f1"]
            and a["recall_hate"] >= b["recall_hate"]
            and a["ece"] <= b["ece"]
            and a["bias_gap"] <= b["bias_gap"]
            and a["latency_p95_ms"] <= b["latency_p95_ms"]
            and a["size_mb"] <= b["size_mb"]
        )
        strict = (
            a["test_macro_f1"] > b["test_macro_f1"]
            or a["recall_hate"] > b["recall_hate"]
            or a["ece"] < b["ece"]
            or a["bias_gap"] < b["bias_gap"]
            or a["latency_p95_ms"] < b["latency_p95_ms"]
            or a["size_mb"] < b["size_mb"]
        )
        return ge and strict

    keep = []
    recs = df.to_dict("records")
    for i, a in enumerate(recs):
        if not any(dominates(b, a) for j, b in enumerate(recs) if j != i):
            keep.append(a["model_id"])
    return df[df["model_id"].isin(keep)]


def run_selection(policy: str = "strict") -> pd.DataFrame:
    q = _quality_rows()
    q = q[q["policy"] == policy].copy()
    cal, bias = _calibration_map(), _bias_map()
    whitelist = set(labels_config().get("commercial_whitelist", []))

    recs = []
    for _, r in q.iterrows():
        size, p50, p95 = _size_and_latency(r["model_id"])
        recs.append(
            {
                **r,
                "ece": cal.get(r["model_id"], float("nan")),
                "bias_gap": bias.get(r["model_id"], float("nan")),
                "size_mb": size,
                "latency_p50_ms": p50,
                "latency_p95_ms": p95,
                "deployable_cpu": size is not None,
                # trained on the full corpus (mixed licenses) -> research-only regardless
                "data_license": "research-only",
            }
        )
    df = pd.DataFrame(recs).sort_values("test_macro_f1", ascending=False).reset_index(drop=True)
    out = resolve("reports/tables") / f"product_selection_{policy}.csv"
    df.to_csv(out, index=False)
    log.info("wrote %s (%d models)", out, len(df))

    deployable = df[df["deployable_cpu"]].dropna(
        subset=["ece", "bias_gap", "latency_p95_ms", "size_mb"]
    )
    front = _pareto_front(deployable) if len(deployable) else deployable

    print(f"\n===== PRODUCT SELECTION [{policy}] =====")
    cols = ["model_id", "family", "test_macro_f1", "recall_hate", "ece", "bias_gap",
            "size_mb", "latency_p50_ms", "latency_p95_ms", "deployable_cpu"]
    print(df[cols].to_string(index=False))
    print("\nQuality leader (all):", df.iloc[0]["model_id"], f"(F1={df.iloc[0]['test_macro_f1']})")
    print("CPU-deployable Pareto front:", ", ".join(front["model_id"].tolist()) or "(none)")
    print(f"License: all models research-only (trained on full corpus). Commercial whitelist = {whitelist or '{}'}.")
    print("Note: neural weights live on Colab -> not benchmarked/deployable locally yet.")
    return df


def run_all() -> None:
    for pol in ("strict", "broad"):
        run_selection(pol)
