"""Etapa 2.3: robustez a ofuscacao ortografica e dependencia de termos de identidade.

Todas as variantes alteram so o texto do split de teste congelado (treino e val
intocados), avaliado com vetorizador, estimador e limiar do model.joblib.

Variantes (vocabulario fixo, ver methodology/DECISOES_METODOLOGICAS.md):
  limpo                    texto original
  ofuscacao_aleatoria      cada palavra sofre ofuscacao com prob. --rate (0.15)
  ofuscacao_direcionada    so as palavras-alvo (lemas HurtLex conservativos do idioma
                           + termos de identidade do bias_probe) sofrem ofuscacao, com
                           prob. --target-rate (0.7); mesma regra para odio e nao-odio
  identidade_removida      termos de identidade trocados por [ID]
("mascarado" fica reservado para odio disfarçado e nao e usado aqui.)

Alem das metricas gerais, reporta a taxa de falsos positivos (FPR) nos textos SEM odio
que citam identidade, em cada variante, contra o FPR de fundo (nao-odio sem identidade).

Uso: python scripts/perturbation_robustness.py [model_id ...] [--rate 0.15] [--target-rate 0.7]
"""
from __future__ import annotations

import csv
import random
import re
import sys
import unicodedata
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hsc.bias_probe import IDENTITY_TERMS, _compile  # noqa: E402
from hsc.config import resolve  # noqa: E402
from hsc.evaluate import breakdown, classification_metrics  # noqa: E402
from hsc.predictions import predict_scores  # noqa: E402
from hsc.utils import ensure_dir  # noqa: E402

LEET = {"o": "0", "a": "4", "e": "3", "i": "1", "s": "5"}
PUNCT = re.compile(r"^\W+|\W+$")


def strip_accents(w: str, rng: random.Random) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", w) if unicodedata.category(c) != "Mn")


def typo(w: str, rng: random.Random) -> str:
    if len(w) < 4:
        return w
    i = rng.randrange(1, len(w) - 1)
    if rng.random() < 0.5:
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    return w[:i] + w[i + 1:]


def leet(w: str, rng: random.Random) -> str:
    idx = [i for i, c in enumerate(w.lower()) if c in LEET]
    if not idx:
        return w
    i = rng.choice(idx)
    return w[:i] + LEET[w[i].lower()] + w[i + 1:]


def censor(w: str, rng: random.Random) -> str:
    if len(w) < 4:
        return w
    i = rng.randrange(1, len(w) - 1)
    return w[:i] + "*" + w[i + 1:]


def spacing(w: str, rng: random.Random) -> str:
    if len(w) < 4:
        return w
    i = rng.randrange(1, len(w))
    return w[:i] + " " + w[i:]


OPS = [strip_accents, typo, leet, censor, spacing]


def obfuscate(w: str, rng: random.Random) -> str:
    """Aplica uma operacao sorteada entre as que de fato alteram a palavra."""
    cands = [o(w, rng) for o in rng.sample(OPS, len(OPS))]
    for c in cands:
        if c != w:
            return c
    return w


def perturb_random(text: str, rate: float, rng: random.Random) -> str:
    out = []
    for w in str(text).split():
        out.append(obfuscate(w, rng) if rng.random() < rate else w)
    return " ".join(out)


def load_hurtlex_conservative(root: Path) -> dict[str, set[str]]:
    lex: dict[str, set[str]] = {"en": set(), "pt": set()}
    for lang in ("en", "pt"):
        with open(root / f"hurtlex_{lang.upper()}.tsv", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                lemma = row["lemma"].strip().lower()
                if row["level"] == "conservative" and lemma and " " not in lemma:
                    lex[lang].add(lemma)
    return lex


def identity_sets() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {"en": set(), "pt": set()}
    for terms in IDENTITY_TERMS.values():
        for lang in out:
            out[lang].update(t.lower() for t in terms.get(lang, []) if " " not in t)
    return out


def perturb_targeted(text: str, lang: str, targets: dict[str, set[str]], rate: float,
                     rng: random.Random) -> str:
    tset = targets.get(lang, set())
    out = []
    for w in str(text).split():
        core = PUNCT.sub("", w).lower()
        out.append(obfuscate(w, rng) if core in tset and rng.random() < rate else w)
    return " ".join(out)


def remove_identity(df: pd.DataFrame) -> pd.Series:
    pats = [_compile(t) for t in IDENTITY_TERMS.values()]

    def f(r):
        t = str(r["text_clean"])
        for p in pats:
            pat = p.get(r["language"])
            if pat:
                t = pat.sub("[ID]", t)
        return t

    return df.apply(f, axis=1)


def mentions_identity(df: pd.DataFrame) -> pd.Series:
    pats = [_compile(t) for t in IDENTITY_TERMS.values()]

    def f(r):
        for p in pats:
            pat = p.get(r["language"])
            if pat and pat.search(str(r["text_clean"])):
                return True
        return False

    return df.apply(f, axis=1)


def evaluate(bundle: dict, texts: pd.Series, base: pd.DataFrame, id_mask: pd.Series,
             variant: str) -> dict:
    vec, est, thr = bundle["vectorizer"], bundle["estimator"], bundle["threshold"]
    score = pd.Series(predict_scores(est, vec.transform(texts.values)), index=base.index)
    d = base[["language", "y_true"]].copy()
    d["y_score"] = score
    d["y_pred"] = (score >= thr).astype(int)
    m = classification_metrics(d["y_true"], d["y_pred"], d["y_score"])
    bl = breakdown(d, "y_true", "y_pred", "language").set_index("language")
    nh = d["y_true"] == 0
    row = {"variant": variant, "roc_auc": round(m["roc_auc"], 4), "macro_f1": round(m["macro_f1"], 4),
           "recall_hate": round(m["recall_hate"], 4)}
    for lang in ("en", "pt"):
        if lang in bl.index:
            row[f"recall_{lang}"] = bl.loc[lang, "recall_hate"]
    row["fpr_fundo"] = round(float(d.loc[nh & ~id_mask, "y_pred"].mean()), 4)
    row["fpr_identidade"] = round(float(d.loc[nh & id_mask, "y_pred"].mean()), 4)
    row["n_nao_odio_identidade"] = int((nh & id_mask).sum())
    return row


def main() -> None:
    args = sys.argv[1:]
    opts = {"--rate": 0.15, "--target-rate": 0.7}
    for k in list(opts):
        if k in args:
            i = args.index(k)
            opts[k] = float(args[i + 1])
            args = args[:i] + args[i + 2:]
    rate, trate = opts["--rate"], opts["--target-rate"]
    models = args or ["tfidf_logreg_strict_s42"]

    lex = load_hurtlex_conservative(resolve("data/external/hurtlex"))
    ids = identity_sets()
    targets = {lang: lex[lang] | ids[lang] for lang in lex}

    rows = []
    for model_id in models:
        policy = "broad" if "_broad_" in model_id else "strict"
        preds = pd.read_parquet(resolve("reports/predictions") / f"{model_id}_test.parquet")
        corpus = pd.read_parquet(resolve("data/processed") / f"corpus_{policy}.parquet")
        base = preds.merge(corpus[["id", "text_clean"]], on="id", how="left")
        id_mask = mentions_identity(base)
        bundle = joblib.load(resolve("models") / model_id / "model.joblib")
        rng = random.Random(42)
        rng_t = random.Random(42)
        targeted = base.apply(
            lambda r: perturb_targeted(r["text_clean"], r["language"], targets, trate, rng_t), axis=1)
        variants = {
            "limpo": base["text_clean"],
            f"ofuscacao_aleatoria_{rate}": base["text_clean"].map(lambda t: perturb_random(t, rate, rng)),
            f"ofuscacao_direcionada_{trate}": targeted,
            "identidade_removida": remove_identity(base),
        }
        for name, texts in variants.items():
            rows.append({"model_id": model_id, **evaluate(bundle, texts, base, id_mask, name)})
    out = pd.DataFrame(rows)
    ensure_dir(resolve("reports"))
    out.to_csv(resolve("reports") / "perturbation_robustness.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
