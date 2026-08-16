"""TF-IDF features: a union of word n-grams and character n-grams.

Char n-grams (char_wb) are essential for noisy social/PT text (misspellings,
morphology, code-switching). The vectorizer is fit ON TRAIN ONLY by train.py.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def _vectorizer(kind: str, spec: dict) -> TfidfVectorizer:
    if kind == "word":
        from hsc.features.stopwords import resolve_stopwords

        return TfidfVectorizer(
            analyzer="word",
            ngram_range=tuple(spec.get("ngram_range", [1, 2])),
            min_df=spec.get("min_df", 1),
            max_features=spec.get("max_features"),
            sublinear_tf=spec.get("sublinear_tf", True),
            lowercase=spec.get("lowercase", True),
            stop_words=resolve_stopwords(spec.get("stop_words")),  # None = keep all (baseline)
        )
    if kind == "char":
        return TfidfVectorizer(
            analyzer=spec.get("analyzer", "char_wb"),
            ngram_range=tuple(spec.get("ngram_range", [3, 5])),
            min_df=spec.get("min_df", 1),
            max_features=spec.get("max_features"),
            sublinear_tf=spec.get("sublinear_tf", True),
            lowercase=spec.get("lowercase", True),
        )
    raise ValueError(f"unknown tfidf part: {kind}")


def build_tfidf(cfg_features: dict) -> FeatureUnion:
    """Return an UNFITTED FeatureUnion(word, char). Caller fits on train only."""
    parts = []
    if "word" in cfg_features:
        parts.append(("word", _vectorizer("word", cfg_features["word"])))
    if "char" in cfg_features:
        parts.append(("char", _vectorizer("char", cfg_features["char"])))
    if "hurtlex" in cfg_features:
        from hsc.features.hurtlex import HurtLexFeatures

        spec = cfg_features["hurtlex"] or {}
        parts.append(("hurtlex", HurtLexFeatures(**spec)))
    if not parts:
        raise ValueError("tfidf features need at least one of: word, char")
    return FeatureUnion(parts)
