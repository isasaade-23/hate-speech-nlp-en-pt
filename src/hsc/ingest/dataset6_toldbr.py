"""ToLD-Br (Leite et al. 2020): 21k Brazilian Portuguese tweets, crowd-annotated.

Each tweet carries SIX category columns (homophobia, obscene, insult, racism,
misogyny, xenophobia), each the COUNT of annotators (0..3, three per tweet) who
assigned that category. Four categories are identity-directed hate; obscene and
insult are offensiveness without a protected target. Majority vote (>= 2 of 3)
folds this into the same 3-way `label_original` as HateBR, so one interim column
drives both policies in configs/labels.yaml:
    hate              : >= 2 votes on any identity category (376 rows, 1.8%)
    offensive_nothate : else >= 2 votes on obscene/insult   (3,687 rows)
    neither           : everything else                     (16,937 rows)
strict -> only `hate` is 1; broad -> offensive folds to 1.

First PT tweets in the corpus: until now PT was only web_comment (pt_fortuna,
hatebr), so this source deconfounds language from domain on the PT side.
Crowd annotation with moderate agreement (not expert like HateBR): minority
votes (1 of 3) fold to the lower bucket, hence `medium` confidence outside
`hate`. Data license CC BY-SA 4.0; kept out of the commercial whitelist because
share-alike on derived models is untested legal ground.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from hsc.ingest.base import build_interim, read_raw_csv

SOURCE = "toldbr"

IDENTITY = ["homophobia", "racism", "misogyny", "xenophobia"]
OFFENSE = ["obscene", "insult"]
MAJORITY = 2  # of 3 annotators


def _three_way(df: pd.DataFrame) -> pd.Series:
    votes = df[IDENTITY + OFFENSE].astype(float)
    assert votes.notna().all().all(), "toldbr: NaN votes"
    assert ((votes >= 0) & (votes <= 3)).all().all(), "toldbr: vote counts outside 0..3"
    lab = pd.Series("neither", index=df.index)
    lab[votes[OFFENSE].max(axis=1) >= MAJORITY] = "offensive_nothate"
    lab[votes[IDENTITY].max(axis=1) >= MAJORITY] = "hate"
    return lab


def load(cfg_source: dict, raw_root: Path) -> pd.DataFrame:
    df = read_raw_csv(SOURCE, cfg_source["member"], cfg_source["encoding"], raw_root)
    return build_interim(
        source=SOURCE,
        language="pt",
        domain="tweet",
        text=df["text"],
        label_original=_three_way(df),
    )
