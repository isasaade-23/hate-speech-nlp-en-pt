"""Shared pytest fixtures. Tiny synthetic frames keep unit tests fast and free of
any dependency on the real (large) source data."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def tiny_memotion_raw() -> pd.DataFrame:
    """Mimics memotion labels.csv (subset of columns used by the loader)."""
    return pd.DataFrame(
        {
            "text_corrected": ["  hello world ", "", "  "],
            "text_ocr": ["hello world raw", "fallback ocr text", "another ocr"],
            "offensive": ["not_offensive", "hateful_offensive", "very_offensive"],
        }
    )
