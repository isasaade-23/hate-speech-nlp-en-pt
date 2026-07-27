"""Fase 9 unintended-bias probe (identity-term false-positive rate).

Measures whether merely mentioning an identity group inflates the model's hate flag on
otherwise non-hateful text — the classic unintended-bias metric (Dixon et al., 2018). For
each identity group we take the NON-HATE test rows whose text mentions a group term and
compute the false-positive rate (fraction predicted hate); a group FPR well above the
background non-hate FPR is a bias signal.

The lexicon uses NEUTRAL group descriptors (women, gay, muslim, negro, imigrante, ...),
never slurs — the point is benign mentions. Terms are matched per language with word
boundaries. Runs on saved predictions, so no model re-run.
"""

from __future__ import annotations

import re

import pandas as pd

from hsc.config import resolve
from hsc.predictions import load_predictions
from hsc.utils import ensure_dir, get_logger, read_json

log = get_logger("hsc.bias_probe")

# Neutral identity descriptors by group and language. Deliberately NOT slurs.
IDENTITY_TERMS: dict[str, dict[str, list[str]]] = {
    "gender": {
        "en": ["women", "woman", "girl", "girls", "female", "feminist", "feminists"],
        "pt": ["mulher", "mulheres", "menina", "feminista", "feministas"],
    },
    "sexual_orientation": {
        "en": ["gay", "gays", "lesbian", "lesbians", "homosexual", "bisexual", "trans", "transgender"],
        "pt": ["gay", "gays", "lésbica", "lésbicas", "homossexual", "bissexual", "trans", "transgênero"],
    },
    "race_ethnicity": {
        "en": ["black", "white", "asian", "latino", "latina", "arab"],
        "pt": ["negro", "negra", "negros", "preto", "preta", "branco", "branca", "asiático", "indígena", "índio"],
    },
    "religion": {
        "en": ["muslim", "muslims", "islam", "jew", "jewish", "christian", "catholic"],
        "pt": ["muçulmano", "muçulmana", "islã", "judeu", "judia", "cristão", "católico", "evangélico", "umbanda", "candomblé"],
    },
    "nationality_immigration": {
        "en": ["immigrant", "immigrants", "refugee", "refugees", "foreigner", "mexican"],
        "pt": ["imigrante", "imigrantes", "refugiado", "refugiados", "estrangeiro", "nordestino", "venezuelano"],
    },
    "disability": {
        "en": ["disabled", "autistic", "deaf", "blind"],
        "pt": ["deficiente", "autista", "surdo", "cego"],
    },
}


def _compile(group_terms: dict[str, list[str]]) -> dict[str, re.Pattern]:
    return {
        lang: re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)
        for lang, terms in group_terms.items()
    }


def _mentions(text: str, lang: str, patterns: dict[str, re.Pattern]) -> bool:
    pat = patterns.get(lang)
    return bool(pat and pat.search(str(text)))


def _all_models_for_policy(policy: str) -> list[str]:
    reg = read_json(resolve("models") / "registry.json")
    return sorted(m for m, e in reg.items() if e.get("policy") == policy)


def run_bias_probe(policy: str = "strict", models: list[str] | None = None) -> pd.DataFrame:
    models = models or _all_models_for_policy(policy)
    corpus = pd.read_parquet(resolve("data/processed") / f"corpus_{policy}.parquet").set_index("id")
    patterns = {g: _compile(terms) for g, terms in IDENTITY_TERMS.items()}

    rows = []
    for model_id in models:
        preds = load_predictions(model_id, "test").set_index("id")
        df = preds.join(corpus[["text_clean"]], how="left")
        nonhate = df[df["y_true"] == 0]
        background_fpr = float((nonhate["y_pred"] == 1).mean()) if len(nonhate) else float("nan")

        for group, pats in patterns.items():
            mask = nonhate.apply(
                lambda r: _mentions(r["text_clean"], r["language"], pats), axis=1
            )
            sub = nonhate[mask]
            n = int(len(sub))
            if n == 0:
                continue
            fpr = float((sub["y_pred"] == 1).mean())
            rows.append(
                {
                    "model_id": model_id,
                    "policy": policy,
                    "group": group,
                    "n_nonhate_mentions": n,
                    "group_fpr": round(fpr, 4),
                    "background_fpr": round(background_fpr, 4),
                    "fpr_gap": round(fpr - background_fpr, 4),
                    "mean_score": round(float(sub["y_score"].mean()), 4),
                }
            )
    df = pd.DataFrame(rows)
    out = ensure_dir(resolve("reports/tables")) / f"bias_identity_fpr_{policy}.csv"
    df.to_csv(out, index=False)
    log.info("wrote %s (%d rows)", out, len(df))
    return df


def run_all() -> None:
    frames = []
    for policy in ("strict", "broad"):
        frames.append(run_bias_probe(policy))
    df = pd.concat(frames, ignore_index=True)
    # Highlight the biggest positive FPR gaps (over-flagging of an identity group).
    worst = df.sort_values("fpr_gap", ascending=False).head(15)
    print("\n===== IDENTITY-TERM FALSE-POSITIVE RATE (non-hate rows mentioning a group) =====")
    print("Largest positive gaps vs. background FPR (over-flagging bias):")
    print(worst.to_string(index=False))
