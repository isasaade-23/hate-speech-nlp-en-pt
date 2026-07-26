"""Memotion 7k (EN memes). Text = OCR; label = `offensive` (4 levels)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hsc.ingest.base import build_interim, read_raw_csv

SOURCE = "memotion"


def load(cfg_source: dict, raw_root: Path) -> pd.DataFrame:
    df = read_raw_csv(SOURCE, cfg_source["member"], cfg_source["encoding"], raw_root)
    corrected = df["text_corrected"].fillna("").astype(str)
    ocr = df["text_ocr"].fillna("").astype(str)
    # Prefer human-corrected OCR; fall back to raw OCR when corrected is empty.
    text = corrected.where(corrected.str.strip().str.len() > 0, ocr)
    return build_interim(
        source=SOURCE,
        language="en",
        domain="meme_ocr",
        text=text,
        label_original=df["offensive"],
    )
