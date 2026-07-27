"""Fase 10 — assemble the paper's comparison tables from reports/metrics/*.json.

Every figure/table is generated from the run metrics, never hand-made, so re-running
an experiment updates the paper automatically. This module builds the model leaderboard
and the per-source / per-language breakdowns.
"""

from __future__ import annotations

import pandas as pd

from hsc.config import resolve
from hsc.utils import ensure_dir, get_logger, read_json

log = get_logger("hsc.report")


def _load_all_metrics() -> list[dict]:
    metrics_dir = resolve("reports/metrics")
    return [read_json(f) for f in sorted(metrics_dir.glob("*.json"))]


def build_leaderboard(write: bool = True) -> pd.DataFrame:
    rows = []
    for r in _load_all_metrics():
        te = r["splits"]["test"]
        va = r["splits"]["val"]
        rows.append(
            {
                "model_id": r["model_id"],
                "family": r.get("family", "?"),
                "config": r.get("config", "?"),
                "policy": r.get("policy", "?"),
                "seed": r.get("seed", "?"),
                "val_macroF1": round(va["macro_f1"], 4),
                "test_macroF1": round(te["macro_f1"], 4),
                "test_CI95": f'[{te["macro_f1_ci95"][0]}, {te["macro_f1_ci95"][1]}]',
                "recall_hate": round(te["recall_hate"], 4),
                "roc_auc": round(te.get("roc_auc", float("nan")), 4),
            }
        )
    if not rows:
        log.warning("no metrics found in reports/metrics/*.json")
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(
        ["policy", "test_macroF1"], ascending=[True, False]
    ).reset_index(drop=True)

    if write:
        tdir = ensure_dir(resolve("reports/tables"))
        df.to_csv(tdir / "leaderboard.csv", index=False)
        (tdir / "leaderboard.md").write_text(_to_markdown(df), encoding="utf-8")
        log.info("leaderboard:\n%s", df.to_string(index=False))
    return df


def _to_markdown(df: pd.DataFrame) -> str:
    """Render a GitHub-flavored markdown table without requiring `tabulate`."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        cols = list(df.columns)
        head = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        body = [
            "| " + " | ".join(str(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)
        ]
        return "\n".join([head, sep, *body])


def build_breakdowns(split: str = "test", write: bool = True) -> pd.DataFrame:
    """Per-source macro-F1 for every model (exposes the language/domain confound)."""
    rows = []
    for r in _load_all_metrics():
        for rec in r["splits"][split].get("by_source", []):
            rows.append(
                {
                    "model_id": r["model_id"],
                    "policy": r.get("policy", "?"),
                    "source_dataset": rec["source_dataset"],
                    "n": rec["n"],
                    "macro_f1": rec["macro_f1"],
                    "recall_hate": rec["recall_hate"],
                }
            )
    df = pd.DataFrame(rows)
    if write and not df.empty:
        tdir = ensure_dir(resolve("reports/tables"))
        df.to_csv(tdir / f"breakdown_by_source_{split}.csv", index=False)
        log.info("wrote per-source breakdown (%d rows)", len(df))
    return df


def build_seed_aggregate(write: bool = True) -> pd.DataFrame:
    """Aggregate across seeds: mean +/- std of test macro-F1 and recall-on-hate per
    (family, config, policy). With a single seed this reports that value and std 0; with
    multiple seeds it is the paper's confidence-interval-style summary."""
    rows = []
    for r in _load_all_metrics():
        te = r["splits"]["test"]
        rows.append(
            {
                "family": r.get("family", "?"),
                "config": r.get("config", "?"),
                "policy": r.get("policy", "?"),
                "seed": r.get("seed", 0),
                "test_macro_f1": te["macro_f1"],
                "recall_hate": te["recall_hate"],
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["family", "config", "policy"])
        .agg(
            n_seeds=("seed", "nunique"),
            macro_f1_mean=("test_macro_f1", "mean"),
            macro_f1_std=("test_macro_f1", "std"),
            recall_hate_mean=("recall_hate", "mean"),
            recall_hate_std=("recall_hate", "std"),
        )
        .reset_index()
    )
    for c in ("macro_f1_mean", "macro_f1_std", "recall_hate_mean", "recall_hate_std"):
        agg[c] = agg[c].round(4)
    agg = agg.sort_values(["policy", "macro_f1_mean"], ascending=[True, False]).reset_index(drop=True)
    if write:
        tdir = ensure_dir(resolve("reports/tables"))
        agg.to_csv(tdir / "leaderboard_agg.csv", index=False)
        log.info("seed-aggregate leaderboard (%d seeds max):\n%s",
                 int(agg["n_seeds"].max()), agg.to_string(index=False))
    return agg


def build_all() -> None:
    build_leaderboard()
    build_seed_aggregate()
    build_breakdowns("test")
