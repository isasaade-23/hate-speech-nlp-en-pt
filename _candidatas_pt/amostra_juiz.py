"""Amostra de OLID-BR e TuPy para o juiz Gemini+Groq (prompt v2.1): 150 com ódio e 150 sem, por base.
Ódio de referência: OLID-BR = algum tipo de ódio marcado; TuPy = voto majoritário. Só leitura das bases."""
from pathlib import Path

import pandas as pd

raiz = Path(__file__).parent
TIPOS = ["health", "ideology", "lgbtqphobia", "other_lifestyle", "physical_aspects",
         "racism", "religious_intolerance", "sexism", "xenophobia"]

o = pd.read_parquet(next((raiz / "olid-br/data").glob("train-*.parquet")))
o = pd.DataFrame({"id": "olidbr_" + o["id"], "texto": o["text"], "ref": o[TIPOS].any(axis=1).astype(int)})
t = pd.read_csv(raiz / "TuPy-Dataset/binary/tupy_binary_vote.csv")
t = pd.DataFrame({"id": "tupy_" + t.index.astype(str), "texto": t["text"], "ref": t["hate"].astype(int)})

partes = []
for nome, df in [("olidbr", o), ("tupy", t)]:
    for ref in (1, 0):
        partes.append(df[df.ref == ref].sample(150, random_state=42).assign(fonte=nome))
am = pd.concat(partes, ignore_index=True)
am.to_csv(raiz / "amostra_juiz.csv", index=False, encoding="utf-8")
print(am.groupby(["fonte", "ref"]).size())
