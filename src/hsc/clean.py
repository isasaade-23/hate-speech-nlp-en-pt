"""Text cleaning with two profiles (see configs/data.yaml -> clean.profiles):

    light : minimal normalization for transformers (keep casing/emoji).
    heavy : aggressive normalization for TF-IDF (lowercase, strip punctuation noise).

Order of operations is fixed and each step is toggleable so cleaning is a documented
methodological choice, not an accident. `emoji` is an optional dependency; if absent,
demojize is skipped (logged once).
"""

from __future__ import annotations

import re
import unicodedata

from hsc.utils import get_logger

log = get_logger("hsc.clean")

try:
    import emoji as _emoji

    _HAS_EMOJI = True
except ImportError:  # pragma: no cover - optional dep
    _HAS_EMOJI = False

# Precompiled patterns
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION = re.compile(r"@\w+")
_RT = re.compile(r"^\s*RT\b[:\s]*", re.IGNORECASE)
_HASHTAG = re.compile(r"#(\w+)")
# C0 controls (except tab/newline/cr) and C1 controls (0x80-0x9f). The C1 range
# matters for the latin-1-decoded Portuguese text, where stray bytes land there.
_CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_WS = re.compile(r"\s+")
# keep letters (incl. accented), digits, whitespace, and a few social tokens
_PUNCT_NOISE = re.compile(r"[^\w\s<>:_]", re.UNICODE)

_EMOJI_WARNED = False


def _maybe_demojize(text: str, keep_emoji: bool, demojize: bool) -> str:
    global _EMOJI_WARNED
    if not demojize:
        if not keep_emoji and _HAS_EMOJI:
            return _emoji.replace_emoji(text, replace="")
        return text
    if not _HAS_EMOJI:
        if not _EMOJI_WARNED:
            log.warning("emoji package not installed; demojize skipped")
            _EMOJI_WARNED = True
        return text
    # emoji -> :name: token (preserves signal in a tokenizer-friendly form)
    return _emoji.demojize(text, delimiters=(" :", ": "))


def clean_text(text: str, profile: dict) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    t = _CONTROL.sub(" ", t)

    if profile.get("strip_rt", True):
        t = _RT.sub("", t)

    url_tok = profile.get("url_token", "<url>")
    t = _URL.sub(f" {url_tok} " if url_tok else " ", t)

    user_tok = profile.get("user_token", "<user>")
    t = _MENTION.sub(f" {user_tok} " if user_tok else " ", t)

    if profile.get("strip_hashtag_symbol", True):
        t = _HASHTAG.sub(r"\1", t)

    t = _maybe_demojize(t, profile.get("keep_emoji", True), profile.get("demojize", True))

    if profile.get("lowercase", False):
        t = t.lower()

    if profile.get("strip_punct_noise", False):
        t = _PUNCT_NOISE.sub(" ", t)

    t = _WS.sub(" ", t).strip()
    return t


def clean_series(texts, profile: dict):
    import pandas as pd

    return pd.Series(texts, dtype="object").map(lambda x: clean_text(x, profile))


def get_profile(data_cfg: dict, name: str) -> dict:
    return data_cfg["clean"]["profiles"][name]
