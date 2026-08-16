"""Vidgen et al. 2021 (ACL), Dynamically Generated Hate Speech Dataset v0.2.3:
41,144 SYNTHETIC English entries written by trained annotators adversarially
against a model-in-the-loop over 4 rounds (Dynabench).

Why it enters in Beta 2.0 phase 2: the corpus was 8% hate; this source is 54%
hate, and its hate is exactly the hard kind (implicit, perturbed, obfuscated)
that our error analysis flags as the main false-negative bucket. Rounds 2-4
carry original/perturbation PAIRS (acl.id.matched): near-dup clustering plus
the frozen group split keeps each pair inside one split, so no leakage.

Label is binary (hate / nothate); `nothate` includes deliberately-hard benign
content (identity mentions, reclaimed slurs) but has no offensive tier, so it
folds to `neither`:
    hate    -> hate               (22,175)
    nothate -> neither            (18,969)
strict and broad coincide for this source (no offensive_nothate bucket).

License CC BY 4.0 -> first large source eligible for the commercial whitelist
alongside multioff. All content is synthetic: no real user text, no PII.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hsc.ingest.base import build_interim, read_raw_csv

SOURCE = "vidgen"

LABEL_MAP = {"hate": "hate", "nothate": "neither"}


def load(cfg_source: dict, raw_root: Path) -> pd.DataFrame:
    df = read_raw_csv(SOURCE, cfg_source["member"], cfg_source["encoding"], raw_root)
    unknown = set(df["label"]) - set(LABEL_MAP)
    assert not unknown, f"vidgen: unknown labels {unknown}"
    return build_interim(
        source=SOURCE,
        language="en",
        domain="synthetic",
        text=df["text"],
        label_original=df["label"].map(LABEL_MAP),
    )
