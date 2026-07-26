"""Fase 5 — language-detection front-end.

Primary detector: `lingua` (self-contained, strong on short/noisy social text). Built
with a handful of languages so it can also say "other" (route such text to the
multilingual model instead of forcing en/pt). fastText lid.176 is a drop-in alternative
if a single-file model is preferred later.

The detector is evaluated against the source-asserted language (ground truth) with a
per-source breakdown, since memes/tweets (code-switching, slang, proper nouns) are the
hard cases. Its predictions are stored as columns so downstream analysis can separate
"model error" from "routed-to-wrong-model error".
"""

from __future__ import annotations

import pandas as pd

from hsc.config import data_config, resolve
from hsc.utils import ensure_dir, get_logger, read_parquet, write_parquet

log = get_logger("hsc.langid")

# languages the detector can distinguish (extra ones let it flag non-en/pt as "other")
_LANG_SET = ["ENGLISH", "PORTUGUESE", "SPANISH", "FRENCH", "ITALIAN", "GERMAN", "ARABIC"]
_TO_CODE = {"ENGLISH": "en", "PORTUGUESE": "pt"}

_DETECTOR = None


def _detector():
    global _DETECTOR
    if _DETECTOR is None:
        from lingua import Language, LanguageDetectorBuilder

        langs = [getattr(Language, name) for name in _LANG_SET]
        _DETECTOR = LanguageDetectorBuilder.from_languages(*langs).build()
    return _DETECTOR


def detect(text: str) -> tuple[str, float]:
    """Return (code, confidence) where code in {en, pt, other}."""
    text = (text or "").strip()
    if not text:
        return "other", 0.0
    det = _detector()
    conf_vals = det.compute_language_confidence_values(text)
    if not conf_vals:
        return "other", 0.0
    top = conf_vals[0]
    code = _TO_CODE.get(top.language.name, "other")
    return code, float(top.value)


def route(code: str, conf: float, threshold: float = 0.5) -> str:
    """Which model handles this text: a per-language model or the multilingual one."""
    if conf < threshold or code == "other":
        return "multilingual"
    return code


def detect_series(texts) -> pd.DataFrame:
    codes, confs = [], []
    for t in texts:
        c, p = detect(t)
        codes.append(c)
        confs.append(p)
    return pd.DataFrame({"lang_pred": codes, "lang_conf": confs})


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy of lang_pred vs source-asserted language, overall and per source."""
    d = df.copy()
    d["correct"] = d["lang_pred"] == d["language"]
    overall = pd.DataFrame(
        {
            "source_dataset": ["ALL"],
            "n": [len(d)],
            "accuracy": [round(float(d["correct"].mean()), 4)],
            "pct_other": [round(float((d["lang_pred"] == "other").mean()), 4)],
        }
    )
    by_src = (
        d.groupby("source_dataset")
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "accuracy": round(float(g["correct"].mean()), 4),
                    "pct_other": round(float((g["lang_pred"] == "other").mean()), 4),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return pd.concat([overall, by_src], ignore_index=True)


def run_langid(policy: str = "strict", threshold: float = 0.5) -> pd.DataFrame:
    data_cfg = data_config()
    corpus_path = resolve(data_cfg["paths"]["processed"]) / f"corpus_{policy}.parquet"
    df = read_parquet(corpus_path)

    preds = detect_series(df["text_clean"].values)
    df["lang_pred"] = preds["lang_pred"].values
    df["lang_conf"] = preds["lang_conf"].values
    df["route"] = [route(c, p, threshold) for c, p in zip(df["lang_pred"], df["lang_conf"])]
    write_parquet(df, corpus_path)

    report = evaluate(df)
    tables = ensure_dir(resolve("reports/tables"))
    report.to_csv(tables / f"langid_eval_{policy}.csv", index=False)
    log.info("langid [%s] eval:\n%s", policy, report.to_string(index=False))
    return report
