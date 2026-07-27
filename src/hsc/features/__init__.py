"""Feature builders: TF-IDF (word + char) and multilingual sentence embeddings.

`build_features` is the single factory `train.py` calls; the `type` key in a config's
`features` block selects the family. Both return an UNFITTED, sklearn-compatible
transformer that is fit ON TRAIN ONLY.
"""

from __future__ import annotations


def build_features(cfg_features: dict):
    kind = cfg_features.get("type", "tfidf")
    if kind == "tfidf":
        from hsc.features.tfidf import build_tfidf

        return build_tfidf(cfg_features)
    if kind in ("embeddings", "sbert"):
        from hsc.features.embeddings import build_embeddings

        return build_embeddings(cfg_features)
    raise ValueError(f"unknown features.type: {kind!r} (expected 'tfidf' or 'embeddings')")
