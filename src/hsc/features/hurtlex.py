"""HurtLex lexicon features (Bassignana et al. 2018), Beta 2.0 phase 2.

The survey guiding Beta 2.0 (Gandhi et al. 2024) reports large gains from
stacking affective-lexicon features onto surface features. HurtLex gives one
multilingual lexicon (EN 8,228 + PT 3,901 lemmas here) organized in 17
categories of offensive/hurtful language (slurs, animals-as-insult, moral
defects, ...), each entry tagged conservative (core) or inclusive (extended).

The transformer is STATELESS with respect to the corpus: the lexicon is a
static external resource under data/external/hurtlex, so fitting is a no-op
and there is nothing to leak. Per text it emits, per (category, level):
count of lemma hits normalized by token count, plus two totals — a small
dense block (2*|categories| + 2 dims) hstacked after the TF-IDF blocks.
Matching is exact on lowercased unigrams and bigrams (HurtLex lemmas are
1-2 words in the vast majority; longer lemmas are ignored).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

# Fixed category order so the feature layout is stable across runs.
CATEGORIES = [
    "an", "asf", "asm", "cds", "ddf", "ddp", "dmc", "is", "om",
    "or", "pa", "pr", "ps", "qas", "rci", "re", "svp",
]
LEVELS = ["conservative", "inclusive"]

_TOKEN_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


def _load_lexicon(root: Path) -> dict[str, list[tuple[int, int]]]:
    """lemma -> list of (category_index, level_index), EN and PT merged."""
    cat_idx = {c: i for i, c in enumerate(CATEGORIES)}
    lvl_idx = {v: i for i, v in enumerate(LEVELS)}
    lex: dict[str, list[tuple[int, int]]] = {}
    for lang in ("EN", "PT"):
        path = root / f"hurtlex_{lang}.tsv"
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                lemma = row["lemma"].strip().lower()
                if not lemma or len(lemma.split()) > 2:
                    continue
                key = (cat_idx.get(row["category"]), lvl_idx.get(row["level"]))
                if key[0] is None or key[1] is None:
                    continue
                lex.setdefault(lemma, []).append(key)
    return lex


class HurtLexFeatures(BaseEstimator, TransformerMixin):
    """Dense-but-sparse-matrix block: hit rates per (category, level) + totals."""

    def __init__(self, lexicon_dir: str = "data/external/hurtlex"):
        self.lexicon_dir = lexicon_dir

    def fit(self, X, y=None):
        self._lex = _load_lexicon(Path(self.lexicon_dir))
        return self

    def transform(self, X):
        if not hasattr(self, "_lex"):
            self.fit(X)
        n_cat, n_lvl = len(CATEGORIES), len(LEVELS)
        out = np.zeros((len(X), n_cat * n_lvl + 2), dtype=np.float64)
        for i, text in enumerate(X):
            tokens = _TOKEN_RE.findall(str(text).lower())
            n = len(tokens)
            if n == 0:
                continue
            grams = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
            hits = 0
            for g in grams:
                for ci, li in self._lex.get(g, ()):
                    out[i, ci * n_lvl + li] += 1.0
                    hits += 1
            out[i, :-2] /= n
            out[i, -2] = hits / n          # overall hit rate
            out[i, -1] = 1.0 if hits else 0.0  # any-hit flag
        return sparse.csr_matrix(out)

    def get_feature_names_out(self, input_features=None):
        names = [f"hurtlex_{c}_{v}" for c in CATEGORIES for v in LEVELS]
        return np.array(names + ["hurtlex_rate", "hurtlex_any"])
