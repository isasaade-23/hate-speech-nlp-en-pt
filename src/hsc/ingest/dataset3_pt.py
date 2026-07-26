"""Portuguese hate speech (Fortuna et al. 2019). Text = `text`; label = `hatespeech_comb`.

Read with latin-1 (see DECISOES_METODOLOGICAS 2026-07-26): utf-8 mojibakes the accents.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hsc.ingest.base import build_interim, read_raw_csv

SOURCE = "pt_fortuna"


def load(cfg_source: dict, raw_root: Path) -> pd.DataFrame:
    df = read_raw_csv(SOURCE, cfg_source["member"], cfg_source["encoding"], raw_root)
    return build_interim(
        source=SOURCE,
        language="pt",
        domain="web_comment",
        text=df["text"],
        label_original=df["hatespeech_comb"],
    )
