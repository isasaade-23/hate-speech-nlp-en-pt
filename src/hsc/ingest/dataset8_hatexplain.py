"""HateXplain (Mathew et al. 2021, AAAI): 20,148 English Twitter + Gab posts,
3 crowd annotators each, with a native 3-way label per annotator
(hatespeech / offensive / normal) that maps one-to-one onto our scheme.

Majority vote (>= 2 of 3), same rule as ToLD-Br:
    hatespeech -> hate               (5,935)
    offensive  -> offensive_nothate  (5,480)
    normal     -> neither            (7,814)
    no majority (3-way tie)          (919, DROPPED - no honest label exists)

Text is the space-join of `post_tokens` (the dataset ships pre-tokenized,
lowercased, with URLs/users already scrubbed by the authors); the heavy
cleaning profile lowercases anyway, so this costs the TF-IDF models nothing.
Domain is recorded as `web_comment`: the Gab half is forum-style and the
tokenization already erased tweet-specific surface (RT, @user, #).

License MIT -> eligible for the commercial whitelist.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from hsc.ingest.base import build_interim

SOURCE = "hatexplain"

LABEL_MAP = {"hatespeech": "hate", "offensive": "offensive_nothate", "normal": "neither"}
MAJORITY = 2  # of 3 annotators


def load(cfg_source: dict, raw_root: Path) -> pd.DataFrame:
    path = raw_root / SOURCE / Path(cfg_source["member"]).name
    with open(path, encoding=cfg_source["encoding"]) as f:
        data = json.load(f)

    texts, labels = [], []
    for post in data.values():
        votes = Counter(a["label"] for a in post["annotators"])
        label, count = votes.most_common(1)[0]
        if count < MAJORITY:
            continue  # 3-way tie: no honest label
        assert label in LABEL_MAP, f"hatexplain: unknown label {label}"
        texts.append(" ".join(post["post_tokens"]))
        labels.append(LABEL_MAP[label])

    return build_interim(
        source=SOURCE,
        language="en",
        domain="web_comment",
        text=pd.Series(texts),
        label_original=pd.Series(labels),
    )
