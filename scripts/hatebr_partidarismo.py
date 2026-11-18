"""Partidarismo do HateBR na política strict v2 (decisão de 20/10).

O código `hate_speech` 5 do HateBR ("Partyism") sozinho cobre ataque a grupo político e a
partido ou político individual. Na política v2, filiação ou ideologia política é protegida:
ataque a eleitores ou militantes continua ódio; ataque a partido, governo ou político, não.
O Groq com o prompt v2 de `llm_label.py` separa os casos.

Uso:
  python scripts/hatebr_partidarismo.py ids      # data/processed/hatebr_cod5_ids.csv
  python scripts/llm_label.py rotular --provedor groq --ids data/processed/hatebr_cod5_ids.csv --tag hatebr_cod5_v21 --lote 5
  python scripts/hatebr_partidarismo.py corpus   # data/processed/corpus_strict_v2.parquet
  python scripts/hatebr_partidarismo.py comparar # linear PT no strict atual e no v2

O corpus v2 é o strict atual com `label` trocado de 1 para 0 nos ids em que o Groq v2 deu
rotulo 0 com status ok. `split` não muda: refazer o split reestratifica pelo rótulo.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/hatebr/HateBR.csv"
CORPUS = ROOT / "data/processed/corpus_strict.parquet"
CORPUS_V2 = ROOT / "data/processed/corpus_strict_v2.parquet"
IDS = ROOT / "data/processed/hatebr_cod5_ids.csv"
GROQ = ROOT / "data/processed/llm_label_hatebr_cod5_v21_groq.jsonl"
GEMINI = ROOT / "data/processed/llm_label_hatebr_cod5_v21_gemini.jsonl"  # ids em que o Groq falha no JSON
FLIP = ROOT / "reports/tables/hatebr_partidarismo_flip.csv"


def hash_split(df: pd.DataFrame) -> str:
    return hashlib.sha256("\n".join(df.id + ":" + df.split).encode()).hexdigest()


def cmd_ids() -> None:
    raw = pd.read_csv(RAW)
    cod = raw.hate_speech.astype(str).str.strip()
    ids = pd.Series([f"hatebr_{i}" for i in raw.index[cod == "5"]])  # id de hsc/ingest/base.py
    corpus = pd.read_parquet(CORPUS, columns=["id", "label"]).set_index("id")
    dentro = ids[ids.isin(corpus.index)]
    assert (corpus.loc[dentro, "label"] == 1).all(), "código 5 fora do positivo strict"
    dentro.to_frame("id").to_csv(IDS, index=False)
    print(f"código 5 sozinho: {len(ids)} no HateBR, {len(dentro)} no corpus strict -> {IDS.name}")


def cmd_corpus() -> None:
    ids = pd.read_csv(IDS).id
    g = pd.read_json(GROQ, lines=True).drop_duplicates("id", keep="last").set_index("id").reindex(ids)
    if GEMINI.exists():  # completa com o Gemini (mesmo prompt) o que o Groq não respondeu
        gm = pd.read_json(GEMINI, lines=True).drop_duplicates("id", keep="last").set_index("id")
        usar = g.index[(g.status != "ok") & g.index.isin(gm.index[gm.status == "ok"])]
        g.loc[usar] = gm.loc[usar, g.columns]
        print(f"Gemini completa {len(usar)} ids")
    assert (g.prompt.dropna() == "v2.1").all(), "saída de LLM que não é do prompt v2.1"
    ok = g.status == "ok"
    print("por provedor (ok):", g[ok].provedor.value_counts().to_dict())
    vira0 = g.index[ok & (g.rotulo == 0)]
    print(f"{len(ids)} ids: {ok.sum()} ok, {(~ok).sum()} sem resposta ok (ficam 1); "
          f"{len(vira0)} viram 0, {(ok & (g.rotulo == 1)).sum()} continuam 1")

    c = pd.read_parquet(CORPUS)
    antes = hash_split(c)
    c.loc[c.id.isin(vira0), "label"] = 0
    assert hash_split(c) == antes
    c.to_parquet(CORPUS_V2, index=False)
    pd.Series(sorted(vira0), name="id").to_csv(FLIP, index=False)

    h = c[c.source_dataset == "hatebr"]
    print("HateBR por split (n, positivos v2):")
    print(h.groupby("split").label.agg(["size", "sum"]).to_string())
    print(f"-> {CORPUS_V2.name}, {FLIP.relative_to(ROOT)}")


FRASES = ["PT", "PL", "Fora PT", "Fora PL", "Votei no PT a vida toda", "Votei no PL a vida toda",
          "O candidato do PT falou bem", "O candidato do PL falou bem", "Petista", "bolsonarista",
          "Petista e bolsonarista, tudo gado", "petista é tudo vagabundo", "o PT virou uma quadrilha"]


def cmd_comparar() -> None:
    """Linear PT (receita de pt_boost.py) treinado no strict atual e no v2, avaliado nos dois rótulos."""
    import numpy as np
    import yaml
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    from pt_boost import SEED, best_threshold, build_vectorizer

    from hsc.clean import clean_text, get_profile

    v1 = pd.read_parquet(CORPUS)
    v2 = pd.read_parquet(CORPUS_V2).set_index("id").label
    pt = v1[v1.language == "pt"].copy()
    pt["label_v1"], pt["label_v2"] = pt.label, v2.reindex(pt.id).values
    tr, va, te = (pt[pt.split == s] for s in ("train", "val", "test"))
    prof = get_profile(yaml.safe_load(open(ROOT / "configs/data.yaml", encoding="utf-8")), "light")
    frases = [clean_text(f, prof) for f in FRASES]

    rows, esc = [], {"frase": FRASES}
    for treino in ("v1", "v2"):
        vec = build_vectorizer()
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                                 solver="liblinear", random_state=SEED)
        clf.fit(vec.fit_transform(tr.text_clean), tr[f"label_{treino}"].values)
        thr = best_threshold(va[f"label_{treino}"].values, clf.predict_proba(vec.transform(va.text_clean))[:, 1])
        s = clf.predict_proba(vec.transform(te.text_clean))[:, 1]
        esc[f"modelo {treino}"] = np.round(clf.predict_proba(vec.transform(frases))[:, 1], 3)
        for rot in ("v1", "v2"):
            for fonte, m in [("todas", np.ones(len(te), bool))] + [
                    (f, (te.source_dataset == f).values) for f in ("hatebr", "pt_fortuna", "toldbr")]:
                y, sc = te[f"label_{rot}"].values[m], s[m]
                yp = sc >= thr
                rows.append({"treino": treino, "rotulo_test": rot, "fonte": fonte, "n": int(m.sum()),
                             "positivos": int(y.sum()), "limiar": round(thr, 3),
                             "pr_auc": round(average_precision_score(y, sc), 3),
                             "recall": round(float(yp[y == 1].mean()), 3),
                             "precisao": round(float(y[yp].mean()), 3) if yp.any() else np.nan,
                             "taxa_fp": round(float(yp[y == 0].mean()), 3)})
    tab = pd.DataFrame(rows)
    dest = ROOT / "reports/tables/hatebr_partidarismo_antes_depois.csv"
    tab.to_csv(dest, index=False)
    print(tab.to_string(index=False))
    print()
    print(pd.DataFrame(esc).to_string(index=False))
    print("->", dest.relative_to(ROOT))


if __name__ == "__main__":
    {"ids": cmd_ids, "corpus": cmd_corpus, "comparar": cmd_comparar}[sys.argv[1]]()
