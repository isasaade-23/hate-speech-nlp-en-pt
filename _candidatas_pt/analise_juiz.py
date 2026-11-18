"""Juiz Gemini+Groq (prompt v2.1) sobre a amostra de OLID-BR e TuPy: sobrevivência do ódio e concordância."""
import json
from pathlib import Path

import pandas as pd

raiz = Path(__file__).parent
am = pd.read_csv(raiz / "amostra_juiz.csv").set_index("id")


def ler(p):
    d = {}
    for l in (raiz.parent / f"data/processed/llm_label_juiz_bases_livres_{p}.jsonl").read_text(encoding="utf-8").splitlines():
        j = json.loads(l)
        d[j["id"]] = j
    return pd.DataFrame(d).T


g, q = ler("gemini"), ler("groq")
am["gem"] = g.loc[am.index].apply(lambda r: r["rotulo"] if r["status"] == "ok" else None, axis=1)
am["grq"] = q.loc[am.index].apply(lambda r: r["rotulo"] if r["status"] == "ok" else None, axis=1)
print(am[["gem", "grq"]].apply(lambda s: s.value_counts(dropna=False).to_dict()))
for c in ("gem", "grq"):
    am[c] = pd.to_numeric(am[c], errors="coerce")
am["ambos"] = (am.gem == 1) & (am.grq == 1)
am["algum"] = (am.gem == 1) | (am.grq == 1)
pos = am[am.ref == 1]
print("\nSobrevivência entre os positivos de referência (n por base):")
print(pos.groupby("fonte").agg(n=("ref", "size"), gemini=("gem", "mean"), groq=("grq", "mean"), ambos=("ambos", "mean"), algum=("algum", "mean")).round(3))
neg = am[am.ref == 0]
print("\nDe referência negativa, rotulados como ódio:")
print(neg.groupby("fonte").agg(n=("ref", "size"), gemini=("gem", "mean"), groq=("grq", "mean"), ambos=("ambos", "mean")).round(3))
ok = am.dropna(subset=["gem", "grq"])
print("\nConcordância Gemini x Groq (n=%d):" % len(ok))
print(ok.groupby("fonte").apply(lambda d: pd.Series({"n": len(d), "acordo": (d.gem == d.grq).mean()})).round(3))
