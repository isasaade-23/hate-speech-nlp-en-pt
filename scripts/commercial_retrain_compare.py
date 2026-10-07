"""Etapa 2.2: retreina o baseline classico (tfidf_logreg) no corpus comercial e
compara com o modelo servido (stack v5) e com o mesmo baseline treinado no corpus
completo, no MESMO teste congelado.

O split NAO e refeito. Val e teste continuam exatamente os de sempre (todas as
fontes); so o TREINO e filtrado, mantendo apenas linhas de fontes commercial_ok
(configs/labels.yaml: commercial_whitelist = multioff, vidgen, hatexplain).
ToLD-BR (CC BY-SA 4.0, licenca ainda pendente) entra numa segunda variante
separada, nunca na whitelist por padrao.

Candidatos comparados, por policy (strict e broad):
  - stack_<policy>_s42            modelo servido hoje (predicoes ja existem)
  - tfidf_logreg_<policy>_s42      mesmo baseline, treinado no corpus COMPLETO
  - tfidf_logreg_commercial_<policy>_s42         treinado so na whitelist
  - tfidf_logreg_commercial_toldbr_<policy>_s42  whitelist + ToLD-BR

Uso: python scripts/commercial_retrain_compare.py
Saida:
  data/processed/corpus_<policy>_commercial[_toldbr].parquet  (treino filtrado)
  reports/tables/commercial_compare_<policy>.csv   (AUC, macro-F1, recall por
                                                     idioma/fonte, memotion externo)
  reports/tables/commercial_identity_gate_<policy>.csv  (sonda de identidade PT)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hsc.bias_probe import IDENTITY_TERMS, _compile, _mentions  # noqa: E402
from hsc.config import data_config, labels_config, resolve  # noqa: E402
from hsc.evaluate import breakdown, classification_metrics  # noqa: E402
from hsc.harmonize import harmonize_source  # noqa: E402
from hsc.predictions import predict_scores  # noqa: E402
from hsc.train import train_from_config  # noqa: E402
from hsc.utils import ensure_dir, write_parquet  # noqa: E402

POLICIES = ["strict", "broad"]
# (file suffix, model_id suffix, include_toldbr)
VARIANTS = [("_commercial", False), ("_commercial_toldbr", True)]
CONFIG = "configs/classical/tfidf_logreg.yaml"


def build_variant(policy: str, include_toldbr: bool) -> pd.DataFrame:
    """Filtra o corpus JA SPLITADO: mantem val/teste intactos, restringe o treino
    as fontes commercial_ok (+ ToLD-BR se pedido). Nao recalcula dedup nem split."""
    labels_cfg = labels_config()
    whitelist = set(labels_cfg.get("commercial_whitelist", []))
    if include_toldbr:
        whitelist.add("toldbr")

    full = pd.read_parquet(resolve("data/processed") / f"corpus_{policy}.parquet")
    is_train = full["split"] == "train"
    keep = (~is_train) | full["source_dataset"].isin(whitelist)
    return full.loc[keep].reset_index(drop=True)


def report_pt_counts(policy: str, suffix: str, df: pd.DataFrame) -> None:
    train = df[df["split"] == "train"]
    pt = train[train["language"] == "pt"]
    print(
        f"[{policy}{suffix}] treino total={len(train)}  treino PT={len(pt)}  "
        f"(odio PT={int(pt['label'].sum())} de {len(pt)})"
    )


def load_candidate_predictions(model_id: str, policy: str) -> pd.DataFrame:
    """Predicoes de teste + texto/idioma, para qualquer model_id ja treinado
    (servido ou novo) — todos avaliados no MESMO teste congelado."""
    preds = pd.read_parquet(resolve("reports/predictions") / f"{model_id}_test.parquet")
    corpus = pd.read_parquet(resolve("data/processed") / f"corpus_{policy}.parquet")
    return preds.merge(corpus[["id", "text_clean"]], on="id", how="left")


def memotion_metrics(model_id: str, policy: str) -> dict:
    import joblib

    bundle = joblib.load(resolve("models") / model_id / "model.joblib")
    vec, est, threshold = bundle["vectorizer"], bundle["estimator"], bundle["threshold"]

    labels_cfg, data_cfg = labels_config(), data_config()
    df, _ = harmonize_source("memotion", policy, labels_cfg, data_cfg, "light")
    X = vec.transform(df["text_clean"].values)
    y_score = predict_scores(est, X)
    y_pred = (pd.Series(y_score).values >= threshold).astype(int)
    m = classification_metrics(df["label"].values, y_pred, y_score)
    return {"memotion_n": m["n"], "memotion_auc": round(m.get("roc_auc", float("nan")), 4),
            "memotion_recall_hate": round(m["recall_hate"], 4)}


def identity_gate_row(model_id: str, policy: str) -> dict:
    """Mesma logica do scripts/bias_gate_v5.py: entre as linhas PT do teste que NAO
    sao odio e mencionam um termo neutro de identidade, taxa de falso positivo
    menos a taxa de fundo do proprio modelo."""
    df = load_candidate_predictions(model_id, policy)
    pt = df[df["language"] == "pt"]
    nao_odio = pt[pt["y_true"] == 0]
    if not len(nao_odio):
        return {"model_id": model_id, "fp_fundo_pt": None, "fp_com_identidade": None,
                "excesso": None, "n_linhas": 0}
    fundo = float((nao_odio["y_pred"] == 1).mean())
    pats = {g: _compile(t) for g, t in IDENTITY_TERMS.items()}
    mask = nao_odio.apply(
        lambda r: any(_mentions(r["text_clean"], "pt", p) for p in pats.values()), axis=1
    )
    sub = nao_odio[mask]
    fp_id = float((sub["y_pred"] == 1).mean()) if len(sub) else float("nan")
    return {
        "model_id": model_id,
        "fp_fundo_pt": round(fundo, 4),
        "fp_com_identidade": round(fp_id, 4) if len(sub) else None,
        "excesso": round(fp_id - fundo, 4) if len(sub) else None,
        "n_linhas": int(len(sub)),
    }


def compare_row(model_id: str, policy: str) -> dict:
    df = load_candidate_predictions(model_id, policy)
    overall = classification_metrics(df["y_true"], df["y_pred"], df["y_score"])
    by_lang = breakdown(df, "y_true", "y_pred", "language").set_index("language")
    by_src = breakdown(df, "y_true", "y_pred", "source_dataset").set_index("source_dataset")

    row = {
        "model_id": model_id,
        "policy": policy,
        "n_test": overall["n"],
        "roc_auc": round(overall.get("roc_auc", float("nan")), 4),
        "macro_f1": round(overall["macro_f1"], 4),
        "recall_hate": round(overall["recall_hate"], 4),
    }
    for lang in ("en", "pt"):
        if lang in by_lang.index:
            row[f"recall_{lang}"] = by_lang.loc[lang, "recall_hate"]
            row[f"n_{lang}"] = int(by_lang.loc[lang, "n"])
    for src in by_src.index:
        row[f"recall_src_{src}"] = by_src.loc[src, "recall_hate"]
    return row


def main() -> None:
    all_rows, gate_rows = [], []

    for policy in POLICIES:
        print(f"\n===== policy={policy} =====")
        variant_ids = []
        for suffix, include_toldbr in VARIANTS:
            df = build_variant(policy, include_toldbr)
            report_pt_counts(policy, suffix, df)
            out = resolve("data/processed") / f"corpus_{policy}{suffix}.parquet"
            write_parquet(df, out)
            variant_ids.append(suffix)

        print(f"\n-- treinando tfidf_logreg nas variantes comerciais [{policy}] --")
        for suffix in variant_ids:
            result = train_from_config(CONFIG, policy_override=policy, corpus_suffix=suffix)
            print(f"  {result['model_id']}: n_train={result['n_train']}  "
                  f"test macro-F1={result['splits']['test']['macro_f1']:.4f}")

        candidates = (
            [f"stack_{policy}_s42", f"tfidf_logreg_{policy}_s42"]
            + [f"tfidf_logreg{suffix}_{policy}_s42" for suffix in variant_ids]
        )
        for mid in candidates:
            row = compare_row(mid, policy)
            try:
                row.update(memotion_metrics(mid, policy))
            except (FileNotFoundError, KeyError):
                pass  # stack is an ensemble bundle (no single vectorizer/estimator) -> skip
            all_rows.append(row)
            gate_rows.append(identity_gate_row(mid, policy))

        cmp_df = pd.DataFrame([r for r in all_rows if r["policy"] == policy])
        cmp_out = ensure_dir(resolve("reports/tables")) / f"commercial_compare_{policy}.csv"
        cmp_df.to_csv(cmp_out, index=False)
        print(f"\n{cmp_df.to_string(index=False)}\n-> {cmp_out}")

        gate_df = pd.DataFrame([g for g in gate_rows if g["model_id"] in candidates])
        gate_out = ensure_dir(resolve("reports/tables")) / f"commercial_identity_gate_{policy}.csv"
        gate_df.to_csv(gate_out, index=False)
        print(f"\n{gate_df.to_string(index=False)}\n-> {gate_out}")


if __name__ == "__main__":
    main()
