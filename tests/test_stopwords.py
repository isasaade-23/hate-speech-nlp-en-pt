"""Tests for the bilingual stop-word ablation helper and its TF-IDF wiring."""

from __future__ import annotations

import pytest

from hsc.features.stopwords import resolve_stopwords
from hsc.features.tfidf import build_tfidf


def test_resolve_none_is_none():
    assert resolve_stopwords(None) is None


def test_resolve_enpt_has_both_languages_and_no_negations():
    sw = resolve_stopwords("enpt")
    assert isinstance(sw, list) and sw == sorted(sw)
    # EN + PT function words present
    for w in ("the", "with", "they", "those"):
        assert w in sw
    for w in ("com", "vocês", "aquele", "porque"):
        assert w in sw
    # negations deliberately excluded (removing them would invert meaning)
    for neg in ("not", "no", "never", "não", "nao", "nunca", "nem"):
        assert neg not in sw


def test_resolve_explicit_list_passes_through_sorted():
    assert resolve_stopwords(["b", "a"]) == ["a", "b"]


def test_resolve_unknown_key_raises():
    with pytest.raises(ValueError):
        resolve_stopwords("klingon")


def test_build_tfidf_with_stopwords_fits_and_drops_them():
    docs = ["they are with those people", "you and the others"]
    feats = {"word": {"ngram_range": [1, 1], "min_df": 1, "stop_words": "enpt"}}
    vec = build_tfidf(feats)
    vec.fit(docs)
    vocab = vec.transformer_list[0][1].vocabulary_
    # function words removed; content words kept
    assert "people" in vocab and "others" in vocab
    assert "they" not in vocab and "with" not in vocab and "the" not in vocab


def test_build_tfidf_without_stopwords_keeps_them():
    docs = ["they are with those people", "you and the others"]
    feats = {"word": {"ngram_range": [1, 1], "min_df": 1}}
    vec = build_tfidf(feats)
    vec.fit(docs)
    vocab = vec.transformer_list[0][1].vocabulary_
    assert "they" in vocab and "with" in vocab  # baseline keeps function words
