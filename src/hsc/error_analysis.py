"""Fase 9 qualitative error analysis.

Turns a model's per-example test predictions into an interpretable error picture: where
it fails (false negatives = missed hate, false positives = over-flagging), and *why*,
using cheap heuristic categories that map to the failure modes the paper discusses —
implicit hate (hateful but no profanity token), slur-triggered over-flagging (non-hate
with a profanity token predicted hate), very short text, and code-switching / language
routing errors (langid disagrees with the source language).

Operates on saved predictions joined to the frozen corpus text, so it never re-runs a
model. The profanity signal uses better-profanity's English wordlist, so it is meaningful
for the EN sources (tweets, memes); for PT rows only the length / routing signals apply
(documented as a limitation).

Curated examples are written to a file, not printed, and carry a content warning: they
are offensive by nature and exist only to support the paper's error discussion.
"""

from __future__ import annotations

import pandas as pd

from hsc.config import resolve
from hsc.predictions import load_predictions
from hsc.probe import _profanity_set
from hsc.utils import ensure_dir, get_logger, read_json

log = get_logger("hsc.error_analysis")


def _best_model_for_policy(policy: str) -> str:
    reg = read_json(resolve("models") / "registry.json")
    cand = {m: e for m, e in reg.items() if e.get("policy") == policy}
    if not cand:
        raise RuntimeError(f"no models registered for policy {policy}")
    return max(cand, key=lambda m: cand[m].get("test_macro_f1", 0.0))


def _has_profanity(texts: pd.Series, prof_set) -> pd.Series:
    if prof_set is None:
        return pd.Series(False, index=texts.index)
    return texts.str.lower().map(lambda t: any(tok in prof_set for tok in str(t).split()))


def run_error_analysis(model_id: str | None = None, policy: str = "strict", top_k: int = 20) -> pd.DataFrame:
    model_id = model_id or _best_model_for_policy(policy)
    preds = load_predictions(model_id, "test").set_index("id")
    corpus = pd.read_parquet(resolve("data/processed") / f"corpus_{policy}.parquet")
    corpus = corpus.set_index("id")

    df = preds.join(corpus[["text_clean", "label_confidence", "lang_pred", "route"]], how="left")
    prof = _profanity_set()
    df["n_tokens"] = df["text_clean"].str.split().map(len)
    df["has_profanity"] = _has_profanity(df["text_clean"], prof)
    df["is_short"] = df["n_tokens"] < 4
    df["langid_mismatch"] = df["lang_pred"] != df["language"]

    df["error"] = df["y_pred"] != df["y_true"]
    df["kind"] = "correct"
    df.loc[(df["y_true"] == 1) & (df["y_pred"] == 0), "kind"] = "false_negative"
    df.loc[(df["y_true"] == 0) & (df["y_pred"] == 1), "kind"] = "false_positive"

    # Named failure modes for the paper's discussion.
    df["implicit_hate"] = (df["kind"] == "false_negative") & (~df["has_profanity"])
    df["slur_overflag"] = (df["kind"] == "false_positive") & (df["has_profanity"])

    tables_dir = ensure_dir(resolve("reports/tables"))

    # (1) error rate by source and language
    by_src = _rate_table(df, ["source_dataset"])
    by_lang = _rate_table(df, ["language"])
    rates = pd.concat([by_src, by_lang], ignore_index=True)
    rates.insert(0, "model_id", model_id)
    rates.to_csv(tables_dir / f"error_rates_{policy}.csv", index=False)

    # (2) failure-mode counts among errors
    err = df[df["error"]]
    modes = pd.DataFrame(
        {
            "model_id": [model_id],
            "policy": [policy],
            "n_errors": [int(len(err))],
            "false_negatives": [int((df["kind"] == "false_negative").sum())],
            "false_positives": [int((df["kind"] == "false_positive").sum())],
            "implicit_hate_FN": [int(df["implicit_hate"].sum())],
            "slur_overflag_FP": [int(df["slur_overflag"].sum())],
            "short_text_errors": [int((err["is_short"]).sum())],
            "langid_mismatch_errors": [int((err["langid_mismatch"]).sum())],
        }
    )
    modes.to_csv(tables_dir / f"error_modes_{policy}.csv", index=False)

    # (3) curated worst-confident examples -> file (content warning inside)
    _write_examples(df, model_id, policy, top_k)

    log.info(
        "error analysis %s [%s]: %d errors (FN=%d FP=%d; implicit_hate=%d slur_overflag=%d)",
        model_id, policy, len(err),
        int((df["kind"] == "false_negative").sum()), int((df["kind"] == "false_positive").sum()),
        int(df["implicit_hate"].sum()), int(df["slur_overflag"].sum()),
    )
    print("\n===== ERROR MODES =====")
    print(modes.to_string(index=False))
    print("\n===== ERROR RATE BY SOURCE / LANGUAGE =====")
    print(rates.to_string(index=False))
    return df


def _rate_table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by[0]):
        n = len(g)
        rows.append(
            {
                "slice_kind": by[0],
                "slice": key,
                "n": int(n),
                "error_rate": round(float(g["error"].mean()), 4),
                "fn_rate": round(float((g["kind"] == "false_negative").mean()), 4),
                "fp_rate": round(float((g["kind"] == "false_positive").mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def _write_examples(df: pd.DataFrame, model_id: str, policy: str, top_k: int) -> None:
    fn = df[df["kind"] == "false_negative"].nsmallest(top_k, "y_score")  # confidently missed hate
    fp = df[df["kind"] == "false_positive"].nlargest(top_k, "y_score")  # confidently over-flagged
    out = ensure_dir(resolve("reports/tables")) / f"error_examples_{policy}.md"
    lines = [
        f"# Curated error examples — {model_id} ({policy})",
        "",
        "> CONTENT WARNING: these texts include hate speech and slurs. They are reproduced",
        "> verbatim only to support the paper's qualitative error analysis.",
        "",
        f"## Confidently missed hate (false negatives, lowest score) — top {len(fn)}",
        "",
        "| score | source | lang | profanity | text |",
        "|------:|--------|------|-----------|------|",
    ]
    for _, r in fn.iterrows():
        lines.append(_ex_row(r))
    lines += [
        "",
        f"## Confidently over-flagged (false positives, highest score) — top {len(fp)}",
        "",
        "| score | source | lang | profanity | text |",
        "|------:|--------|------|-----------|------|",
    ]
    for _, r in fp.iterrows():
        lines.append(_ex_row(r))
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s (%d FN + %d FP examples)", out, len(fn), len(fp))


def _ex_row(r) -> str:
    txt = str(r["text_clean"]).replace("|", "\\|").replace("\n", " ")[:200]
    return f"| {r['y_score']:.3f} | {r['source_dataset']} | {r['language']} | {'yes' if r['has_profanity'] else 'no'} | {txt} |"
