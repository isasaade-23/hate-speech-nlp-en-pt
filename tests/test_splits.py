"""Anti-leakage gate. These tests must pass for any result to be trustworthy."""

from __future__ import annotations

import random

import pandas as pd

from hsc.splits import (
    drop_exact_duplicates,
    make_splits,
    near_dup_cluster_ids,
    normalize_for_dedup,
)

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

_WORDS = [f"word{w}" for w in range(120)]  # distinct-enough vocabulary pool


def _varied_corpus(n_per_label: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for label in (0, 1):
        for _ in range(n_per_label):
            text = " ".join(rng.sample(_WORDS, 6))
            rows.append(
                {
                    "id": f"r{len(rows)}",
                    "text": text,
                    "text_clean": text,
                    "label": label,
                    "label_original": str(label),
                    "label_confidence": "high",
                    "label_policy": "strict",
                    "language": "en",
                    "source_dataset": "tweets_ip",
                    "domain": "tweet",
                }
            )
    return pd.DataFrame(rows)


def test_drop_exact_duplicates_prefers_high_confidence():
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "text_clean": ["Hello World", "hello world", "unique text"],
            "label_confidence": ["low", "high", "high"],
            "label": [0, 1, 0],
        }
    )
    kept, removed = drop_exact_duplicates(df)
    assert removed == 1  # "Hello World" / "hello world" collapse
    # the surviving duplicate is the high-confidence one (id "b")
    survivor = kept[kept["text_clean"].str.lower() == "hello world"]
    assert survivor["label_confidence"].iloc[0] == "high"


def test_near_dup_groups_case_and_whitespace_variants():
    texts = [
        "Hello world this is a test sentence",
        "hello   world this is a test sentence",  # same after normalization
        "completely unrelated vocabulary nothing shared zzz qqq",
    ]
    clusters = near_dup_cluster_ids(texts, threshold=0.9)
    assert clusters[0] == clusters[1]  # variants cluster together
    assert clusters[2] != clusters[0]  # distinct text stays apart


def test_make_splits_no_text_or_cluster_leakage():
    df = _varied_corpus()
    # inject a genuine near-duplicate pair (long shared body, 1-char different tail)
    base = "this is a deliberately long shared sentence used to force a near duplicate pair tail "
    inject = pd.DataFrame(
        [
            {
                "id": "dupA",
                "text": base + "A",
                "text_clean": base + "A",
                "label": 0,
                "label_original": "0",
                "label_confidence": "high",
                "label_policy": "strict",
                "language": "en",
                "source_dataset": "tweets_ip",
                "domain": "tweet",
            },
            {
                "id": "dupB",
                "text": base + "B",
                "text_clean": base + "B",
                "label": 0,
                "label_original": "0",
                "label_confidence": "high",
                "label_policy": "strict",
                "language": "en",
                "source_dataset": "tweets_ip",
                "domain": "tweet",
            },
        ]
    )
    df = pd.concat([df, inject], ignore_index=True)

    out, stats = make_splits(df, RATIOS, seed=42)

    # (a) no normalized text appears in more than one split
    norm = out["text_clean"].map(normalize_for_dedup)
    per_text_splits = out.assign(_n=norm).groupby("_n")["split"].nunique()
    assert (per_text_splits == 1).all(), "a text leaked across splits"

    # (b) no cluster spans more than one split
    per_cluster_splits = out.groupby("dup_cluster_id")["split"].nunique()
    assert (per_cluster_splits == 1).all(), "a dup cluster leaked across splits"

    # (c) the injected near-dup pair shares a cluster and thus a split
    pair = out[out["id"].isin(["dupA", "dupB"])]
    assert pair["dup_cluster_id"].nunique() == 1
    assert pair["split"].nunique() == 1

    # (d) all three splits are non-empty
    assert set(out["split"].unique()) == {"train", "val", "test"}
