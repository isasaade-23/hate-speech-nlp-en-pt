"""Combina o stack servido com o modelo dedicado ao PT, na fatia PT.

Motivo: o modelo PT sozinho dobra o recall de ódio mas TRIPLICA o falso
positivo em frases neutras com termo de identidade (9/20 contra 3/20), marcando
pessoas LGBT, negras e muçulmanas que falam de si mesmas. Isso é inaceitável
numa ferramenta antiódio, então o modelo PT sozinho não vai ao ar.

Hipótese: combinar os dois recupera parte do recall sem o dano, porque o stack
pontua baixo justamente nessas frases (0,09 a 0,32) e puxa a média para baixo.

Protocolo: pesos e limiar ajustados na VALIDAÇÃO PT; teste tocado uma vez por
configuração; a sonda de identidade roda em todas.

Uso: python scripts/pt_ensemble.py
Saída: reports/tables/pt_ensemble_v5.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsc.clean import clean_text  # noqa: E402
from hsc.config import data_config  # noqa: E402
from hsc.inference import HateClassifier  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from pt_identity_probe import CONTROLES, FRASES  # noqa: E402


def best_threshold(y_true, y_score) -> float:
    y_true, y_score = np.asarray(y_true), np.asarray(y_score, dtype=float)
    cands = np.unique(np.quantile(y_score, np.linspace(0.02, 0.98, 97)))
    best_t, best_f = 0.5, -1.0
    for t in cands:
        f = f1_score(y_true, (y_score >= t).astype(int), average="macro", zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


def row(nome, y, s, thr, fp_ident, tp_ctrl):
    yp = (np.asarray(s) >= thr).astype(int)
    return {"config": nome, "limiar": round(float(thr), 4),
            "macro_f1": round(f1_score(y, yp, average="macro", zero_division=0), 4),
            "recall_hate": round(recall_score(y, yp, zero_division=0), 4),
            "precision_hate": round(precision_score(y, yp, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y, s), 4),
            "fp_identidade_20": fp_ident, "acerto_controle_3": tp_ctrl}


def main() -> None:
    corpus = pd.read_parquet(ROOT / "data/processed/corpus_strict.parquet")
    pt = corpus[corpus.language == "pt"]
    va, te = pt[pt.split == "val"], pt[pt.split == "test"]

    stack = HateClassifier("stack_strict_s42")
    bundle = joblib.load(ROOT / "models/pt_logreg_strict_s42/model.joblib")
    vec, est = bundle["vectorizer"], bundle["estimator"]
    profile = data_config()["clean"]["profiles"]["light"]

    def s_pt(texts):
        X = vec.transform([clean_text(t, profile) for t in texts])
        return est.predict_proba(X)[:, 1]

    def s_stack(texts):
        return np.array([r["score"] for r in stack.predict_batch(list(texts))])

    print("escorando validação e teste PT nos dois modelos...")
    va_s, va_p = s_stack(va.text_clean), s_pt(va.text_clean)
    te_s, te_p = s_stack(te.text_clean), s_pt(te.text_clean)
    yv, yt = va.label.values, te.label.values

    # sonda de identidade: frases neutras (esperado 0) + controles de ódio
    id_s, id_p = s_stack(FRASES), s_pt(FRASES)
    ct_s, ct_p = s_stack(CONTROLES), s_pt(CONTROLES)

    configs = {}
    configs["stack servido"] = (te_s, stack.threshold, id_s, ct_s)

    thr_ptonly = best_threshold(yv, va_p)
    configs["modelo PT sozinho"] = (te_p, thr_ptonly, id_p, ct_p)

    for w in (0.3, 0.5, 0.7):
        nome = f"media ponderada (PT {w:.0%})"
        v = (1 - w) * va_s + w * va_p
        t_ = (1 - w) * te_s + w * te_p
        configs[nome] = (t_, best_threshold(yv, v),
                         (1 - w) * id_s + w * id_p, (1 - w) * ct_s + w * ct_p)

    # meta-logreg sobre os dois scores, ajustada na validação PT
    meta = LogisticRegression(C=1.0, max_iter=1000).fit(np.c_[va_s, va_p], yv)
    mv = meta.predict_proba(np.c_[va_s, va_p])[:, 1]
    configs["meta-logreg (val PT)"] = (
        meta.predict_proba(np.c_[te_s, te_p])[:, 1], best_threshold(yv, mv),
        meta.predict_proba(np.c_[id_s, id_p])[:, 1],
        meta.predict_proba(np.c_[ct_s, ct_p])[:, 1])

    # exige que OS DOIS concordem (regra conservadora, sem score combinado)
    both_v = np.minimum(va_s / stack.threshold, va_p / thr_ptonly)
    both_t = np.minimum(te_s / stack.threshold, te_p / thr_ptonly)
    configs["ambos concordam"] = (
        both_t, best_threshold(yv, both_v),
        np.minimum(id_s / stack.threshold, id_p / thr_ptonly),
        np.minimum(ct_s / stack.threshold, ct_p / thr_ptonly))

    rows = []
    for nome, (s_te, thr, s_id, s_ct) in configs.items():
        rows.append(row(nome, yt, s_te, thr,
                        int((np.asarray(s_id) >= thr).sum()),
                        int((np.asarray(s_ct) >= thr).sum())))

    out = pd.DataFrame(rows)
    dest = ROOT / "reports/tables/pt_ensemble_v5.csv"
    out.to_csv(dest, index=False)
    print("\n=== fatia PT do teste v5 (4.748) + sonda de identidade ===")
    print("fp_identidade_20: falsos positivos em 20 frases neutras (menor é melhor)")
    print("acerto_controle_3: ódio real detectado em 3 frases (maior é melhor)\n")
    print(out.to_string(index=False))
    print("\n->", dest)


if __name__ == "__main__":
    main()
