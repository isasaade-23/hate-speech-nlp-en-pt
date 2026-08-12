"""HateBR (Vargas et al. 2022): 7k Brazilian Instagram comments, expert-annotated.

The source column `hate_speech` is a category CODE, not a binary flag:
    "0"  -> non-offensive                              (3500 rows)
    "-1" -> offensive but NOT hate speech              (2798 rows)
    else -> hate (codes 1..9 = the nine hate categories; decimals like "5,8" = multi-category)  (702 rows)

We fold this into a 3-way `label_original` {neither, offensive_nothate, hate} so that a
SINGLE interim column drives BOTH policies in configs/labels.yaml:
    strict : only `hate` -> 1 (702 positives; true hate only)
    broad  : offensive folds to hate (== offensive_language; 3500 positives)
This is exactly the offensive-vs-hate distinction that is this project's scientific crux,
here given natively by the annotators instead of inferred.

Domain is web_comment (Instagram comments share the short user-comment register with
pt_fortuna; source_dataset keeps them separable in the breakdowns). Research-only license
(Sinch): kept OUT of the commercial whitelist (see methodology/data_provenance.md).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hsc.ingest.base import build_interim, read_raw_csv

SOURCE = "hatebr"


def _three_way(hate_speech: pd.Series, offensive: pd.Series) -> pd.Series:
    """Map the `hate_speech` category code to {neither, offensive_nothate, hate}."""
    code = hate_speech.astype(str).str.strip()
    lab = pd.Series("hate", index=code.index)
    lab[code == "0"] = "neither"
    lab[code == "-1"] = "offensive_nothate"
    # Integrity: the 3-way partition must agree with the binary offensive flag, or the
    # source schema drifted (fail loud rather than harmonize a corrupted mapping).
    off = offensive.astype(str).str.strip()
    assert ((lab == "neither") == (off == "0")).all(), "hatebr: 'neither' != offensive_language==0"
    assert ((lab != "neither") == (off == "1")).all(), "hatebr: offensive buckets != offensive_language==1"
    return lab


def load(cfg_source: dict, raw_root: Path) -> pd.DataFrame:
    df = read_raw_csv(SOURCE, cfg_source["member"], cfg_source["encoding"], raw_root)
    label_original = _three_way(df["hate_speech"], df["offensive_language"])
    return build_interim(
        source=SOURCE,
        language="pt",
        domain="web_comment",
        text=df["instagram_comments"],
        label_original=label_original,
    )
