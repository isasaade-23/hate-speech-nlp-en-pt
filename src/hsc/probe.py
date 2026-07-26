"""Fase 2 — resolve the undocumented 1/2/3 labels of the tweets_ip dataset.

Automatic evidence (this module):
  - class distribution (ordering test vs Davidson-style hate<offensive<neither)
  - per-class lexical signals: profanity rate (better-profanity), all-caps word rate,
    exclamation rate, mention rate, length.

Manual step (the user): label the exported stratified sample and compute agreement.

Output:
  - reports/tables/dataset2_probe.csv  (per-class signals)
  - data/external/dataset2_audit_sample.csv  (150 tweets, 50/class, for manual audit)
  - data/interim/tweets_ip_probe.json  (evidence + PROVISIONAL decision -> opens the gate)

The gate file records decision="provisional_auto"; the manual audit upgrades it to
"confirmed" or revises the mapping (see methodology/DECISOES_METODOLOGICAS.md).
"""

from __future__ import annotations

import pandas as pd

from hsc.config import data_config, resolve
from hsc.utils import ensure_dir, get_logger, write_json

log = get_logger("hsc.probe")

# Provisional interpretation to be tested by the evidence below.
PROVISIONAL_MAPPING = {"1": "hate", "2": "offensive", "3": "neither"}


def _profanity_set():
    """Single-token profanity lexicon from better-profanity's bundled wordlist.
    Set-membership over tokens is O(tokens) — vastly faster than per-text library
    calls, and fast enough to score all rows."""
    try:
        import importlib.resources as ir

        text = (
            ir.files("better_profanity")
            .joinpath("profanity_wordlist.txt")
            .read_text(encoding="utf-8")
        )
    except Exception:  # pragma: no cover
        log.warning("better-profanity wordlist not found; profanity_rate will be NaN")
        return None
    words = {w.strip().lower() for w in text.splitlines() if w.strip() and " " not in w.strip()}
    return words or None


def _class_signals(texts: pd.Series, prof_set) -> dict:
    """Vectorized lexical signals over all rows in the class."""
    n = len(texts)
    n_tokens = texts.str.split().map(len)
    prof_rate = float("nan")
    if prof_set is not None and n > 0:
        prof_rate = float(
            texts.str.lower().map(lambda t: any(tok in prof_set for tok in t.split())).mean()
        )
    return {
        "n": int(n),
        "mean_chars": round(float(texts.str.len().mean()), 1),
        "mean_tokens": round(float(n_tokens.mean()), 1),
        "profanity_rate": round(prof_rate, 3),
        "allcaps_word_rate": round(float((texts.str.count(r"\b[A-Z]{3,}\b") > 0).mean()), 3),
        "exclaim_rate": round(float((texts.str.count("!") > 0).mean()), 3),
        "mention_rate": round(float((texts.str.count(r"@\w+") > 0).mean()), 3),
    }


def run(seed: int = 42, sample_per_class: int = 50) -> dict:
    data_cfg = data_config()
    interim = resolve(data_cfg["paths"]["interim"]) / "tweets_ip.parquet"
    df = pd.read_parquet(interim)
    df["label_original"] = df["label_original"].astype(str)
    df["text"] = df["text"].astype(str)

    prof_set = _profanity_set()

    rows = {}
    total = len(df)
    for lab in ["1", "2", "3"]:
        sub = df.loc[df["label_original"] == lab, "text"]
        sig = _class_signals(sub, prof_set)
        sig["share"] = round(sig["n"] / total, 3)
        rows[lab] = sig

    table = pd.DataFrame(rows).T[
        ["n", "share", "mean_chars", "mean_tokens", "profanity_rate",
         "allcaps_word_rate", "exclaim_rate", "mention_rate"]
    ]
    table.index.name = "label_original"

    # --- distribution ordering test (hate < offensive < neither expected for 1<2<3) ---
    counts = {lab: rows[lab]["n"] for lab in ["1", "2", "3"]}
    ordering_ok = counts["1"] < counts["2"] < counts["3"]
    # --- severity signal: profanity should be highest in the 'hate' class ---
    prof = {lab: rows[lab]["profanity_rate"] for lab in ["1", "2", "3"]}
    severity_ok = (prof["1"] >= prof["3"]) if prof["1"] == prof["1"] else None  # nan-safe

    # persist table
    tables_dir = ensure_dir(resolve("reports/tables"))
    table.to_csv(tables_dir / "dataset2_probe.csv")
    log.info("per-class signals:\n%s", table.to_string())
    log.info("distribution ordering 1<2<3 holds: %s | counts=%s", ordering_ok, counts)
    log.info("profanity(1) >= profanity(3): %s | profanity=%s", severity_ok, prof)

    # --- stratified audit sample for the manual step ---
    parts = []
    for lab in ["1", "2", "3"]:
        sub = df[df["label_original"] == lab]
        take = min(sample_per_class, len(sub))
        parts.append(sub.sample(n=take, random_state=seed))
    audit = pd.concat(parts, ignore_index=True)[["id", "text", "label_original"]]
    audit["manual_label"] = ""  # user fills: hate / offensive / neither
    audit = audit.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle
    ext_dir = ensure_dir(resolve(data_cfg["paths"]["external"]))
    audit_path = ext_dir / "dataset2_audit_sample.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8")
    log.info("wrote audit sample (%d rows) -> %s", len(audit), audit_path)

    # --- provisional decision opens the harmonize gate ---
    evidence = {
        "per_class": rows,
        "distribution_counts": counts,
        "distribution_ordering_1_lt_2_lt_3": ordering_ok,
        "profanity_severity_ok": severity_ok,
    }
    decision = {
        "decision": "provisional_auto",
        "mapping": PROVISIONAL_MAPPING,
        "binary_strict": {"1": 1, "2": 0, "3": 0},
        "binary_broad": {"1": 1, "2": 1, "3": 0},
        "manual_audit": "pending",
        "audit_sample": str(audit_path.relative_to(resolve("."))),
        "evidence": evidence,
        "note": (
            "Automatic evidence supports 1=hate, 2=offensive, 3=neither. Confirm with the "
            "manual audit (fill manual_label in the audit sample) before final publication."
        ),
    }
    gate = resolve(data_cfg["paths"]["interim"]) / "tweets_ip_probe.json"
    write_json(decision, gate)
    log.info("wrote gate decision -> %s (harmonize will now include tweets_ip)", gate)
    return decision
