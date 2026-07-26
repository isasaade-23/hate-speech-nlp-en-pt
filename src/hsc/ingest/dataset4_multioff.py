"""MultiOFF offensive memes (EN). Text = `sentence`; label = `offensive`/`Non-offensiv`.

The three official split files are concatenated here and re-split later by our own
leakage-safe splitter (Fase 4). Label string 'Non-offensiv' is truncated in the source;
matching is by prefix during harmonization (configs/labels.yaml: label_match: prefix).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hsc.ingest.base import build_interim, read_raw_csv

SOURCE = "multioff"


def load(cfg_source: dict, raw_root: Path) -> pd.DataFrame:
    frames = []
    for member in cfg_source["members"]:
        frames.append(read_raw_csv(SOURCE, member, cfg_source["encoding"], raw_root))
    df = pd.concat(frames, ignore_index=True)
    return build_interim(
        source=SOURCE,
        language="en",
        domain="meme_ocr",
        text=df["sentence"],
        label_original=df["label"],
    )
