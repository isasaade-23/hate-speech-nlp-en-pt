"""Unit tests for the ingestion layer (no dependency on the real source zips)."""

from __future__ import annotations

import pandas as pd

from hsc.ingest.base import build_interim, validate_interim
from hsc.schema import INTERIM_COLUMNS, VALID_DOMAINS, VALID_LANGUAGES, VALID_SOURCES


def test_build_interim_schema_and_ids():
    df = build_interim(
        source="pt_fortuna",
        language="pt",
        domain="web_comment",
        text=pd.Series(["  olá mundo ", "segundo texto"]),
        label_original=pd.Series(["1", "0"]),
    )
    assert list(df.columns) == INTERIM_COLUMNS
    assert df["id"].tolist() == ["pt_fortuna_0", "pt_fortuna_1"]
    assert df["id"].is_unique
    # text is stripped
    assert df.loc[0, "text"] == "olá mundo"
    # constant metadata columns
    assert set(df["language"]) == {"pt"}
    assert set(df["source_dataset"]) == {"pt_fortuna"}
    assert set(df["domain"]) == {"web_comment"}


def test_validate_interim_stats():
    df = build_interim(
        source="tweets_ip",
        language="en",
        domain="tweet",
        text=pd.Series(["a real tweet", "   "]),
        label_original=pd.Series(["1", "3"]),
    )
    st = validate_interim(df, "tweets_ip")
    assert st["rows"] == 2
    assert st["empty_text"] == 1  # the whitespace-only row becomes empty after strip
    assert st["labels"] == {"1": 1, "3": 1}


def test_schema_vocabularies_consistent():
    assert VALID_SOURCES == {"memotion", "tweets_ip", "pt_fortuna", "multioff", "hatebr"}
    assert VALID_LANGUAGES == {"en", "pt"}
    assert "meme_ocr" in VALID_DOMAINS
