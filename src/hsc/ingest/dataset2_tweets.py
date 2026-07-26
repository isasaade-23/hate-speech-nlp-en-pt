"""Hate/offensive tweets (EN). Text = `tweet`; label = `label` (1/2/3, provisional)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hsc.ingest.base import build_interim, read_raw_csv

SOURCE = "tweets_ip"


def load(cfg_source: dict, raw_root: Path) -> pd.DataFrame:
    df = read_raw_csv(SOURCE, cfg_source["member"], cfg_source["encoding"], raw_root)
    return build_interim(
        source=SOURCE,
        language="en",
        domain="tweet",
        text=df["tweet"],
        label_original=df["label"],
    )
