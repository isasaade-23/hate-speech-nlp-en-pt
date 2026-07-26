"""The common schema every ingest loader emits, and the source registry.

Keeping the schema in one place means data/interim/*.parquet, the unified corpus,
and the tests all agree on column names and dtypes.
"""

from __future__ import annotations

# Columns of data/interim/<source>.parquet (per-dataset, pre-harmonization label view)
INTERIM_COLUMNS = [
    "id",              # stable unique id: "<source>_<rownum>"
    "text",            # raw text before cleaning
    "label_original",  # original label value as a string (traceability)
    "language",        # asserted source language: "en" | "pt"
    "source_dataset",  # "memotion" | "tweets_ip" | "pt_fortuna" | "multioff"
    "domain",          # "tweet" | "web_comment" | "meme_ocr"
]

# Extra columns added during harmonization (data/processed/corpus.parquet)
CORPUS_EXTRA_COLUMNS = [
    "text_clean",       # cleaned text (profile chosen downstream)
    "label",            # harmonized binary: 1 = hate, 0 = not-hate
    "label_confidence", # "high" | "low"
    "label_policy",     # "strict" | "broad"
    "lang_pred",        # language predicted by langid (filled in Fase 5)
    "lang_conf",        # langid confidence
    "dup_cluster_id",   # near-duplicate cluster (grouping key for splits)
    "split",            # "train" | "val" | "test"
]

VALID_LANGUAGES = {"en", "pt"}
VALID_DOMAINS = {"tweet", "web_comment", "meme_ocr"}
VALID_SOURCES = {"memotion", "tweets_ip", "pt_fortuna", "multioff"}
POSITIVE_LABEL = 1  # hate
