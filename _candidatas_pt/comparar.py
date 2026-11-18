"""Compara OLID-BR e TuPy com o corpus PT atual: sobreposição exata e proporção de ódio. Só leitura."""
import re
from pathlib import Path

import pandas as pd

raiz = Path(__file__).parent
corpus = pd.read_parquet(raiz.parent / "data" / "processed" / "corpus_strict_v2.parquet")
pt = corpus[corpus["language"] == "pt"].copy()


def norm(s):
    s = re.sub(r"https?://\S+|@\w+", " ", str(s).lower())
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


pt["k"] = pt["text"].map(norm)
print("Corpus PT (strict v2):")
print(pt.groupby("source_dataset")["label"].agg(n="size", odio="sum"))

olid = pd.concat(pd.read_parquet(f) for f in sorted((raiz / "olid-br" / "data").glob("*.parquet")))
tupy = pd.read_csv(raiz / "TuPy-Dataset" / "binary" / "tupy_binary_vote.csv")
tipos = ["health", "ideology", "lgbtqphobia", "other_lifestyle", "physical_aspects",
         "racism", "religious_intolerance", "sexism", "xenophobia"]
olid["odio_tipo"] = olid[tipos].any(axis=1)  # exclui insult e profanity_obscene
olid["k"] = olid["text"].map(norm)
tupy["k"] = tupy["text"].map(norm)

print("\nOLID-BR:", len(olid), "linhas")
print(" OFF:", (olid["is_offensive"] == "OFF").sum(), "| com tipo de ódio (sem insult/profanity):", olid["odio_tipo"].sum())
print(olid[tipos].sum().to_string())
print("\nTuPy:", len(tupy), "linhas | hate=1:", int(tupy["hate"].sum()))

for nome, df in [("OLID-BR", olid), ("TuPy", tupy)]:
    print(f"\nSobreposição exata de {nome} com o corpus PT (texto normalizado):")
    print(" duplicatas internas:", int(df["k"].duplicated().sum()))
    for fonte, g in pt.groupby("source_dataset"):
        comuns = set(df["k"]) & set(g["k"])
        print(f"  {fonte}: {len(comuns)} textos em comum ({len(comuns) / len(df):.1%} da base)")
print("\nOLID-BR x TuPy:", len(set(olid["k"]) & set(tupy["k"])), "em comum")
