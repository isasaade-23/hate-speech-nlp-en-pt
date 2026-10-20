"""Fase 0 do plano PT: auditoria do pipeline em português (itens 1 a 3, sem API).

1. Integridade: repetidos e quase repetidos entre treino/val/test, rótulo strict
   contra o rótulo original, efeito da limpeza light.
2. Curva de aprendizado do linear PT (receita de pt_boost.py) com 25/50/75/100%
   do treino PT, 3 sementes de subamostra, limiar ajustado na validação.
3. Erros por fonte: linear da extensão, linear PT e BERTimbau, separados por
   ToLD-BR, HateBR e pt_fortuna (e pelo rótulo original dentro de cada fonte).

O item 4 (LLM como juiz) depende da Fase 1 e fica pendente no relatório.

Uso: python scripts/auditoria_pt.py
Saída: reports/auditoria_pt.md (só agregados) e reports/tables/auditoria_pt_dups.csv (só ids)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import FeatureUnion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsc.clean import clean_text, get_profile  # noqa: E402
from hsc.splits import near_dup_cluster_ids, normalize_for_dedup  # noqa: E402

SEED = 42
FRACS = (0.25, 0.5, 0.75, 1.0)
SUB_SEEDS = (0, 1, 2)
# Platt do bundle tfidf_logreg_strict_s42 (o linear da extensão)
PLATT_COEF, PLATT_INT = 5.891850407761439, -3.715252235801602
NIVEIS = {"sensível 0,424": 0.42417971021544754, "conservador 0,70": 0.70}
SPLITS = ("train", "val", "test")
N_BOOT = 2000


def md(df: pd.DataFrame) -> str:
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
    linhas = ["| " + " | ".join(map(str, df.columns)) + " |",
              "|" + "---|" * len(df.columns)]
    linhas += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(linhas)


def best_threshold(y_true, y_score) -> float:
    """Mesmo protocolo de pt_boost.py: quantis, maximiza macro-F1 na validação."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score, dtype=float)
    cands = np.unique(np.quantile(y_score, np.linspace(0.02, 0.98, 97)))
    best_t, best_f = 0.5, -1.0
    for t in cands:
        f = f1_score(y_true, (y_score >= t).astype(int), average="macro", zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


def build_vectorizer() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3,
                                 max_features=150000, sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 max_features=150000, sublinear_tf=True, lowercase=True)),
    ])


# ---------------------------------------------------------------- 1. integridade

def integridade(pt: pd.DataFrame, data_cfg: dict) -> tuple[str, pd.DataFrame]:
    out = ["## 1. Integridade\n"]
    dup_rows = []

    # 1a. repetidos exatos (mesma normalização do build)
    key = pt.text_clean.map(normalize_for_dedup)
    g = pt.assign(_k=key).groupby("_k")
    n_exact = int((g.size() > 1).sum())
    out.append(f"**Repetidos exatos** (texto limpo normalizado): {n_exact} grupos com mais de um texto.\n")

    # 1b. quase repetidos, no limiar do build (0,90) e num limiar mais frouxo (0,70)
    out.append("**Quase repetidos** (MinHash 5-gramas de caractere, só PT):\n")
    rows = []
    for thr in (0.90, 0.70):
        cid = near_dup_cluster_ids(pt.text_clean.values, threshold=thr)
        d = pt.assign(cid=cid)
        multi = d.groupby("cid").filter(lambda x: len(x) > 1)
        nspl = multi.groupby("cid").split.nunique()
        cross = nspl[nspl > 1].index
        cross_rows = multi[multi.cid.isin(cross)]
        train_c = set(d.loc[d.split == "train", "cid"])
        test_vaz = d[(d.split == "test") & d.cid.isin(train_c)]
        lab = multi.groupby("cid").label.nunique()
        src = multi.groupby("cid").source_dataset.nunique()
        rows.append({"limiar_jaccard": thr,
                     "clusters_com_2+": int(multi.cid.nunique()),
                     "textos_em_clusters": len(multi),
                     "clusters_entre_splits": len(cross),
                     "test_com_par_no_treino": len(test_vaz),
                     "test_com_par_no_treino_odio": int(test_vaz.label.sum()),
                     "clusters_rotulo_conflitante": int((lab > 1).sum()),
                     "clusters_entre_fontes": int((src > 1).sum())})
        for _, r in cross_rows.iterrows():
            dup_rows.append({"limiar": thr, "cluster": int(r.cid), "id": r.id,
                             "split": r.split, "source_dataset": r.source_dataset,
                             "label": int(r.label)})
    out.append(md(pd.DataFrame(rows)) + "\n")
    out.append("O build agrupa por cluster a 0,90 antes do split, então a linha de 0,90 deve dar "
               "zero entre splits. A de 0,70 mede o que escapa a esse limiar. Ids em "
               "`reports/tables/auditoria_pt_dups.csv`.\n")

    # 1c. rótulo strict contra o original
    out.append("**Rótulo strict contra o rótulo original:**\n")
    ct = (pt.groupby(["source_dataset", "label_original", "label"]).size()
          .rename("n").reset_index())
    out.append(md(ct) + "\n")
    esperado = {"hate": 1, "neither": 0, "offensive_nothate": 0, "1": 1, "0": 0}
    exp = pt.label_original.astype(str).map(esperado)
    n_bad = int((exp != pt.label).sum())
    out.append(f"Linhas fora do mapeamento esperado (hate e 1 → 1; neither, offensive_nothate e 0 → 0): "
               f"{n_bad}. `label_policy`: {pt.label_policy.value_counts().to_dict()}. "
               f"`label_confidence`: {pt.label_confidence.value_counts(dropna=False).to_dict()}.\n")
    tx = (pt.groupby(["source_dataset", "split"]).label.agg(["size", "mean"])
          .rename(columns={"size": "n", "mean": "taxa_odio"}).reset_index())
    out.append("Taxa de ódio por fonte e split (estratificação):\n")
    out.append(md(tx) + "\n")

    # 1d. limpeza light
    prof = get_profile(data_cfg, "light")
    recl = pt.text.map(lambda t: clean_text(t, prof))
    raw_len = pt.text.astype(str).str.len()
    cl_len = pt.text_clean.astype(str).str.len()
    toks = pt.text_clean.str.split()
    so_marc = toks.map(lambda ts: len(ts) > 0 and all(t in ("<user>", "<url>") for t in ts))
    d = pd.DataFrame({
        "fonte": pt.source_dataset, "label": pt.label,
        "vazio": cl_len == 0,
        "ate_2_tokens": toks.map(len) <= 2,
        "so_user_url": so_marc,
        "perdeu_>50%_chars": (cl_len < 0.5 * raw_len),
        "tem_<user>": pt.text_clean.str.contains("<user>", regex=False),
        "tem_<url>": pt.text_clean.str.contains("<url>", regex=False),
        "tem_emoji_:nome:": pt.text_clean.str.contains(r":[a-z_]+:", regex=True),
        "text_clean_difere_da_limpeza_atual": recl != pt.text_clean,
    })
    cols = [c for c in d.columns if c not in ("fonte", "label")]
    lim = d.groupby("fonte")[cols].mean().reset_index()
    tot = d[cols].mean().to_frame().T.assign(fonte="total")
    lim = pd.concat([lim, tot])[["fonte"] + cols]
    out.append("**Limpeza light** (proporção de textos):\n")
    out.append(md(lim) + "\n")
    out.append(f"Comprimento mediano em caracteres: bruto {raw_len.median():.0f}, limpo {cl_len.median():.0f}. "
               f"Taxa de ódio entre textos com até 2 tokens: {d.loc[d.ate_2_tokens, 'label'].mean():.4f} "
               f"(n = {int(d.ate_2_tokens.sum())}).\n")
    return "\n".join(out), pd.DataFrame(dup_rows)


# ---------------------------------------------------------- 2. curva de aprendizado

def curva(pt: pd.DataFrame) -> str:
    tr, va, te = (pt[pt.split == s] for s in SPLITS)
    rows = []
    for frac in FRACS:
        for s in (SUB_SEEDS if frac < 1 else (0,)):
            sub = tr if frac == 1 else (tr.groupby(["source_dataset", "label"], group_keys=False)
                                        .sample(frac=frac, random_state=s))
            vec = build_vectorizer()
            Xtr = vec.fit_transform(sub.text_clean)
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                                     solver="liblinear", random_state=SEED)
            clf.fit(Xtr, sub.label.values)
            s_va = clf.predict_proba(vec.transform(va.text_clean))[:, 1]
            s_te = clf.predict_proba(vec.transform(te.text_clean))[:, 1]
            thr = best_threshold(va.label.values, s_va)
            yp = (s_te >= thr).astype(int)
            r = {"frac": frac, "seed": s, "n_treino": len(sub), "odio_treino": int(sub.label.sum()),
                 "macro_f1": f1_score(te.label, yp, average="macro"),
                 "recall_odio": recall_score(te.label, yp),
                 "precisao_odio": precision_score(te.label, yp, zero_division=0),
                 "pr_auc": average_precision_score(te.label, s_te)}
            for src in sorted(te.source_dataset.unique()):
                m = (te.source_dataset == src).values
                r[f"pr_auc_{src}"] = average_precision_score(te.label.values[m], s_te[m])
            rows.append(r)
            print(f"  curva frac={frac} seed={s} macro_f1={r['macro_f1']:.4f}", flush=True)
    d = pd.DataFrame(rows)
    agg = d.drop(columns="seed").groupby("frac").agg(["mean", "std"])
    agg.columns = [f"{a}" if b == "mean" else f"{a}_dp" for a, b in agg.columns]
    keep = ["n_treino", "odio_treino", "macro_f1", "macro_f1_dp", "recall_odio", "precisao_odio",
            "pr_auc", "pr_auc_dp"] + [c for c in agg.columns if c.startswith("pr_auc_") and not c.endswith("_dp")]
    agg = agg[keep].reset_index()
    agg[["n_treino", "odio_treino"]] = agg[["n_treino", "odio_treino"]].round().astype(int)
    return ("## 2. Curva de aprendizado do linear PT\n\n"
            "Receita de `scripts/pt_boost.py` (TF-IDF word 1-2 + char_wb 3-5, LogReg balanceada), "
            "treinada só em PT. Subamostra estratificada por fonte e rótulo, 3 sementes nas frações "
            "menores (dp = desvio-padrão entre sementes). Limiar ajustado na validação PT completa; "
            f"test PT com {len(te)} textos.\n\n" + md(agg) + "\n")


# ---------------------------------------------------------------- 3. erros por fonte

def platt(s):
    return 1 / (1 + np.exp(-(PLATT_COEF * np.asarray(s) + PLATT_INT)))


def bloco(y, yp, sc) -> dict:
    y, yp = np.asarray(y), np.asarray(yp)
    neg = (y == 0).sum()
    r = {"n": len(y), "n_odio": int(y.sum()),
         "recall": recall_score(y, yp, zero_division=0),
         "precisao": precision_score(y, yp, zero_division=0),
         "taxa_fp": float(((yp == 1) & (y == 0)).sum() / neg) if neg else np.nan,
         "fp": int(((yp == 1) & (y == 0)).sum()), "fn": int(((yp == 0) & (y == 1)).sum())}
    if 0 < y.sum() < len(y):
        r["pr_auc"] = average_precision_score(y, sc)
        r["roc_auc"] = roc_auc_score(y, sc)
    return r


def boot_indices(fonte, y) -> dict[str, np.ndarray]:
    """Reamostras (N_BOOT x n) estratificadas por fonte e classe, posições no frame alinhado."""
    rng = np.random.default_rng(SEED)
    out = {}
    for src in sorted(set(fonte)):
        partes = []
        for c in (0, 1):
            pos = np.flatnonzero((fonte == src) & (y == c))
            partes.append(pos[rng.integers(0, len(pos), size=(N_BOOT, len(pos)))])
        out[src] = np.hstack(partes)
    out["todas"] = np.hstack([out[s] for s in sorted(set(fonte))])
    return out


def boot_ic(y, yp, sc, idx) -> tuple[dict, np.ndarray]:
    """IC 95% por percentil; devolve também a série de PR-AUC para o teste pareado."""
    m = {"recall": [], "precisao": [], "taxa_fp": [], "pr_auc": []}
    for b in idx:
        yb, pb = y[b], yp[b]
        tp, fp = ((pb == 1) & (yb == 1)).sum(), ((pb == 1) & (yb == 0)).sum()
        m["recall"].append(tp / yb.sum())
        m["precisao"].append(tp / (tp + fp) if tp + fp else 0.0)
        m["taxa_fp"].append(fp / (yb == 0).sum())
        m["pr_auc"].append(average_precision_score(yb, sc[b]))
    ic = {f"{k}_ic": "{:.3f}–{:.3f}".format(*np.percentile(v, [2.5, 97.5])) for k, v in m.items()}
    return ic, np.asarray(m["pr_auc"])


def erros(pt: pd.DataFrame) -> str:
    meta = pt.set_index("id")[["label_original"]]
    modelos = {}
    p = pd.read_parquet(ROOT / "reports/predictions/tfidf_logreg_strict_s42_test.parquet")
    p = p[p.language == "pt"].copy()
    p["y_score"] = platt(p.y_score)
    for nome, thr in NIVEIS.items():
        modelos[f"linear extensão ({nome})"] = p.assign(y_pred=(p.y_score >= thr).astype(int))
    for arq, nome in (("pt_logreg_strict_s42", "linear PT (limiar val)"),
                      ("bertimbau_pt_strict_s42", "BERTimbau (limiar val)")):
        q = pd.read_parquet(ROOT / f"reports/predictions/{arq}_test.parquet")
        modelos[nome] = q[q.language == "pt"]

    # todas as predições alinhadas na mesma ordem de ids, para o bootstrap pareado
    ids = np.sort(modelos["BERTimbau (limiar val)"].id.values)
    modelos = {k: v.set_index("id").loc[ids].reset_index() for k, v in modelos.items()}
    ref = modelos["BERTimbau (limiar val)"]
    idx = boot_indices(ref.source_dataset.values, ref.y_true.values)
    pr_boot = {}

    rows, rows_orig = [], []
    for nome, d in modelos.items():
        d = d.join(meta, on="id")
        assert d.label_original.notna().all(), nome
        for src in ["todas"] + sorted(d.source_dataset.unique()):
            g = d if src == "todas" else d[d.source_dataset == src]
            r = {"modelo": nome, "fonte": src, **bloco(g.y_true, g.y_pred, g.y_score)}
            ic, pr_boot[(nome, src)] = boot_ic(d.y_true.values, d.y_pred.values,
                                               d.y_score.values, idx[src])
            rows.append({**r, **ic})
        for src, g in d.groupby("source_dataset"):
            for lo, h in g.groupby("label_original"):
                rows_orig.append({"modelo": nome, "fonte": src, "rotulo_original": lo, "n": len(h),
                                  "taxa_predito_odio": h.y_pred.mean()})
    tab = pd.DataFrame(rows)
    dif = []
    for src in ["todas"] + sorted(ref.source_dataset.unique()):
        dd = pr_boot[("BERTimbau (limiar val)", src)] - pr_boot[("linear PT (limiar val)", src)]
        pto = tab.set_index(["modelo", "fonte"]).pr_auc
        dif.append({"fonte": src,
                    "dif_pr_auc": pto[("BERTimbau (limiar val)", src)] - pto[("linear PT (limiar val)", src)],
                    "ic95": "{:.3f}–{:.3f}".format(*np.percentile(dd, [2.5, 97.5])),
                    "prop_reamostras_bertimbau_melhor": float((dd > 0).mean())})
    orig =(pd.DataFrame(rows_orig)
            .pivot_table(index=["fonte", "rotulo_original", "n"], columns="modelo",
                         values="taxa_predito_odio").reset_index())
    orig.columns.name = None
    return ("## 3. Erros por fonte (test PT)\n\n"
            "Linear da extensão = `tfidf_logreg_strict_s42` (bilíngue) com o Platt do bundle, nos "
            "níveis sensível e conservador. Linear PT = `pt_logreg_strict_s42`. Taxa de FP = FP "
            "sobre os não-ódio da fonte. PR-AUC e ROC-AUC não dependem do limiar. Colunas `_ic`: "
            f"IC 95% por bootstrap ({N_BOOT} reamostras estratificadas por fonte e classe, "
            "percentil).\n\n"
            + md(tab) + "\n\n"
            "**BERTimbau menos linear PT, PR-AUC** (bootstrap pareado, mesmas reamostras):\n\n"
            + md(pd.DataFrame(dif)) + "\n\n"
            "**Taxa de predito ódio por rótulo original.** Nas linhas `hate`/`1` é o recall; nas "
            "outras, a taxa de FP. Separa o FP sobre ofensa sem ódio do FP sobre texto neutro.\n\n"
            + md(orig) + "\n")


def amostra_juiz() -> pd.DataFrame:
    """Item 4: erros e acertos COMUNS ao BERTimbau e ao linear PT no test PT. Grava só ids."""
    a = pd.read_parquet(ROOT / "reports/predictions/bertimbau_pt_strict_s42_test.parquet")
    b = pd.read_parquet(ROOT / "reports/predictions/pt_logreg_strict_s42_test.parquet")
    d = a[a.language == "pt"].merge(b[["id", "y_pred"]], on="id", suffixes=("", "_lin"))
    d["grupo"] = None
    d.loc[(d.y_true == 1) & (d.y_pred == 0) & (d.y_pred_lin == 0), "grupo"] = "FN_comum"
    d.loc[(d.y_true == 0) & (d.y_pred == 1) & (d.y_pred_lin == 1), "grupo"] = "FP_comum"
    d.loc[(d.y_pred == d.y_true) & (d.y_pred_lin == d.y_true), "grupo"] = "acerto_comum"
    fn = d[d.grupo == "FN_comum"]
    fp_all = d[d.grupo == "FP_comum"]
    # 100 FP, proporcional à fonte (maior resto para fechar a conta)
    cota = fp_all.source_dataset.value_counts() * 100 / len(fp_all)
    n_fp = cota.astype(int)
    n_fp[(cota - n_fp).sort_values(ascending=False).index[:100 - n_fp.sum()]] += 1
    fp = pd.concat([g.sample(n_fp[s], random_state=SEED) for s, g in fp_all.groupby("source_dataset")])
    # 300 acertos: 100 por fonte, metade ódio quando houver
    ac = []
    for s, g in d[d.grupo == "acerto_comum"].groupby("source_dataset"):
        odio = g[g.y_true == 1]
        n1 = min(50, len(odio))
        ac += [odio.sample(n1, random_state=SEED),
               g[g.y_true == 0].sample(100 - n1, random_state=SEED)]
    out = pd.concat([fn, fp] + ac).sample(frac=1, random_state=SEED)
    out = out.rename(columns={"source_dataset": "fonte", "y_true": "label"})[["id", "grupo", "fonte", "label"]]
    out.to_csv(ROOT / "data/processed/auditoria_pt_amostra.csv", index=False)
    return out


def main() -> None:
    if "--amostra" in sys.argv:
        a = amostra_juiz()
        print(a.groupby(["grupo", "fonte", "label"]).size().to_string())
        print(f"{len(a)} ids -> data/processed/auditoria_pt_amostra.csv")
        return
    data_cfg = yaml.safe_load(open(ROOT / "configs/data.yaml", encoding="utf-8"))
    corpus = pd.read_parquet(ROOT / "data/processed/corpus_strict.parquet")
    pt = corpus[corpus.language == "pt"].reset_index(drop=True)
    print(f"PT: {len(pt)} textos", flush=True)

    sec1, dups = integridade(pt, data_cfg)
    print("1 ok", flush=True)
    sec3 = erros(pt)
    print("3 ok", flush=True)
    sec2 = curva(pt)

    head = ("# Auditoria do pipeline em português (Fase 0, itens 1 a 3)\n\n"
            "Gerado por `scripts/auditoria_pt.py`. Corpus `corpus_strict.parquet`, fatia PT "
            f"({len(pt)} textos: {pt.split.value_counts().to_dict()}). Só agregados; nenhum texto.\n\n"
            "O item 4 (LLM como juiz do rótulo) espera o pipeline de API da Fase 1.\n")
    dest = ROOT / "reports/auditoria_pt.md"
    dest.write_text("\n".join([head, sec1, sec2, sec3]), encoding="utf-8")
    dups.to_csv(ROOT / "reports/tables/auditoria_pt_dups.csv", index=False)
    print("->", dest)


if __name__ == "__main__":
    main()
