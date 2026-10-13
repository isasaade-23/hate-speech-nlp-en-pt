"""Etapa 2.5: dados sinteticos em PT-BR gerados por LLM local (Ollama, Qwen2.5-7B).

Sementes: SOMENTE linhas de treino das fontes permissivas (vidgen, hatexplain) do
corpus strict. O texto da semente orienta tema, grupo-alvo e tamanho; o modelo e
instruido a escrever um texto NOVO, nao uma traducao. Fontes restritas nao sao usadas.

Saida em JSONL, uma linha por exemplo, com checkpoint a cada item (retoma de onde
parou). Os sinteticos ficam na fila de validacao humana; nao entram no treino.

Uso:
  python scripts/generate_synthetic.py gen --n 200 --out data/processed/synthetic_pilot.jsonl
  python scripts/generate_synthetic.py check --out data/processed/synthetic_pilot.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen2.5:7b-instruct"
SEED_SOURCES = ["vidgen", "hatexplain"]
SEED = 42

REFUSAL_RE = re.compile(
    r"\bI (cannot|can't|won't)\b|\bI'm (sorry|unable)\b|as an ai\b|\bnão posso\b|"
    r"\bnão consigo\b|\bpolítica de conteúdo\b|\bI apologize\b|\bdesculpe\b|"
    r"\bsinto muito\b|\bnão é apropriado\b|\bnão devo\b|\bnão vou\b|\bnão gero\b",
    re.IGNORECASE,
)

PROMPT_HATE = (
    "Escreva UM post novo de rede social brasileira (informal, 1 a 2 frases) que ataque, "
    "ofenda ou humilhe um GRUPO de pessoas (por etnia, religião, nacionalidade, região, "
    "gênero, orientação sexual, deficiência ou idade), citando o grupo de forma explícita "
    "no texto. Use o mesmo tipo de grupo e de tema do exemplo abaixo, mas com outras "
    "palavras e outra situação; não traduza o exemplo. Não suavize, não explique, não "
    "adicione avisos. Responda SOMENTE com o texto do post.\n\nExemplo (em inglês): {seed}"
)
PROMPT_CLEAN = (
    "Escreva UM post novo de rede social brasileira (informal, 1 a 2 frases) sobre o mesmo "
    "tema do exemplo abaixo, citando o mesmo tipo de grupo ou assunto se o exemplo citar. "
    "O post é civil: uma opinião, uma notícia, uma pergunta ou uma experiência pessoal, sem "
    "nenhuma ofensa, ataque ou estereótipo. Fale como uma pessoa comum falaria, sem "
    "comentar sobre o próprio post, sobre pesquisa, dados ou projetos. Não traduza o "
    "exemplo. Responda SOMENTE com o texto do post.\n\nExemplo (em inglês): {seed}"
)

# Filtros da v2 (aplicados depois da geracao; o motivo do descarte fica registrado)
ECO_PROMPT_RE = re.compile(
    r"discurso[s]? de [oó]dio|dados? de treino|treinamento|detec[cç][aã]o|classificador|"
    r"projeto|pesquisa|acad[eê]mic|dataset|conjunto de dados|modelo de linguagem|"
    r"\bpost\b|\bexemplo\b|\bIA\b|intelig[eê]ncia artificial|aqui est[aá]|#discurso",
    re.IGNORECASE,
)
# radicais de grupos-alvo (etnia, religiao, nacionalidade, regiao, genero, orientacao,
# deficiencia, idade, classe); casa o inicio da palavra
GRUPO_RE = re.compile(
    r"\b(negr|preto|pret[ao]s?\b|branc|asi[aá]tic|chin[eê]s|japon[eê]s|coreano|[aá]rabe|"
    r"latino|ind[ií]gena|[ií]ndio|cigan|judeu|jud[aá]ic|mu[cç]ulman|isl[aâ]m|xiita|sunita|"
    r"hindu|crist[aã]|evang[eé]lic|cat[oó]lic|ateu|umbandist|macumbeir|imigrante|refugiad|"
    r"estrangeir|gringo|nordest|nortist|\bnorte\b|bahia|sert[aã]o|sertanej|baiano|paraiba|carioca|paulista|ga[uú]ch|mineir|favel|gr[aá]vid|gestante|"
    r"favelad|pobre|mendig|morador de rua|mulher|feminist|menina|homem|gay\b|gays\b|"
    r"l[eé]sbica|lgbt|homossexual|bissexual|trans\b|transg[eê]nero|travesti|bicha|"
    r"viado|sapat[aã]o|deficiente|cadeirante|autista|retardad|aleijad|idos[oa]|velh[oa]s?\b|"
    r"gord[oa]|obes[oa]|anao|an[aã]o)",
    re.IGNORECASE,
)


def descarte(text: str, label: int) -> list[str]:
    """Motivos de descarte (lista vazia = aprovado)."""
    from hsc.langid import detect

    motivos = []
    if ECO_PROMPT_RE.search(text):
        motivos.append("eco_prompt")
    if label == 1 and not GRUPO_RE.search(text):
        motivos.append("sem_grupo_alvo")
    code, conf = detect(text)
    if code != "pt":
        motivos.append("nao_portugues")
    if len(text) < 20:
        motivos.append("curto")
    return motivos


def load_seeds(n: int, offset: int = 0, exclude: list[str] | None = None,
               n1: int | None = None, n0: int | None = None) -> pd.DataFrame:
    """n1 sementes de odio e n0 de nao odio (padrao: n//2 de cada), sem repetir as de
    nenhum dos jsonl em `exclude`; classes intercaladas na proporcao."""
    c = pd.read_parquet(ROOT / "data/processed/corpus_strict.parquet")
    c = c[c["source_dataset"].isin(SEED_SOURCES) & (c["split"] == "train")]
    c = c[c["text_clean"].str.len().between(40, 300)]
    for arq in exclude or []:
        if Path(arq).exists():
            c = c[~c["id"].isin(set(pd.read_json(arq, lines=True)["seed_id"]))]
    sizes = {1: n // 2 if n1 is None else n1, 0: n // 2 if n0 is None else n0}
    parts = []
    for lab in (1, 0):
        s = c[c["label"] == lab].sample(sizes[lab], random_state=SEED + offset).reset_index(drop=True)
        s["_k"] = (s.index + 0.5) / max(len(s), 1)
        parts.append(s)
    # intercala as classes para que um corte antecipado fique balanceado
    out = pd.concat(parts).sort_values("_k", kind="stable").drop(columns="_k")
    return out.reset_index(drop=True)


def generate(seed_text: str, label: int) -> str:
    prompt = (PROMPT_HATE if label == 1 else PROMPT_CLEAN).format(seed=seed_text)
    r = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.8, "num_predict": 120}},
        timeout=180,
    )
    r.raise_for_status()
    text = r.json()["response"].strip()
    head, sep, rest = text.partition("\n\n")
    if sep and head.rstrip().endswith(":"):  # remove preambulo tipo "Aqui esta o post:"
        text = rest.strip()
    return text.strip('"')


def cmd_gen(args) -> None:
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["seed_id"] for l in out.read_text(encoding="utf-8").splitlines() if l}
    seeds = load_seeds(args.n, args.offset, args.exclude, args.n_odio, args.n_nao)
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        for i, row in seeds.iterrows():
            if row["id"] in done:
                continue
            text = generate(row["text_clean"], int(row["label"]))
            rec = {"seed_id": row["id"], "seed_source": row["source_dataset"],
                   "label": int(row["label"]), "text_pt": text,
                   "flag_refusal": bool(REFUSAL_RE.search(text)) or len(text) < 10}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(seeds)}  {time.time() - t0:.0f}s", flush=True)


def cmd_check(args) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    syn = pd.read_json(args.out, lines=True)
    ok = syn[~syn["flag_refusal"]].reset_index(drop=True)
    corpus = pd.read_parquet(ROOT / "data/processed/corpus_strict.parquet")
    real = corpus[(corpus["split"] == "train") & (corpus["language"] == "pt")]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    R = vec.fit_transform(real["text_clean"].values)
    S = vec.transform(ok["text_pt"].values)
    sim_real = linear_kernel(S, R).max(axis=1)
    SS = linear_kernel(S, S)
    for i in range(len(ok)):
        SS[i, i] = 0
    sim_syn = SS.max(axis=1)
    print(f"n={len(syn)}  recusas={int(syn['flag_refusal'].sum())}  "
          f"(odio {int(syn[syn.label == 1]['flag_refusal'].sum())}/{int((syn.label == 1).sum())}, "
          f"nao-odio {int(syn[syn.label == 0]['flag_refusal'].sum())}/{int((syn.label == 0).sum())})")
    print(f"max sim vs treino PT real: mediana={pd.Series(sim_real).median():.3f}  "
          f">0.8: {int((sim_real > 0.8).sum())}")
    print(f"max sim entre sinteticos:  mediana={pd.Series(sim_syn).median():.3f}  "
          f">0.8: {int((sim_syn > 0.8).sum())}")


def cmd_filter(args) -> None:
    """Aplica os filtros da v2 e grava <out>.aprovados.jsonl; imprime a taxa de descarte."""
    syn = pd.read_json(args.out, lines=True)
    syn["motivos"] = [
        ["recusa"] if r.flag_refusal else descarte(r.text_pt, int(r.label))
        for r in syn.itertuples()
    ]
    syn["descartado"] = syn["motivos"].map(bool)
    for lab, nome in ((1, "ódio"), (0, "não ódio")):
        s = syn[syn.label == lab]
        print(f"{nome}: {int(s.descartado.sum())}/{len(s)} descartados ({s.descartado.mean():.0%})")
    print(f"total: {int(syn.descartado.sum())}/{len(syn)} descartados ({syn.descartado.mean():.0%})")
    from collections import Counter
    c = Counter(m for ms in syn["motivos"] for m in ms)
    print("motivos (um texto pode ter mais de um):", dict(c))
    out = Path(args.out).with_suffix(".aprovados.jsonl")
    syn[~syn.descartado].drop(columns=["motivos", "descartado"]).to_json(
        out, orient="records", lines=True, force_ascii=False)
    print("aprovados em", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "check", "filter"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0, help="soma na semente do sorteio de sementes")
    ap.add_argument("--exclude", nargs="*", help="jsonl cujas sementes (seed_id) nao devem ser reusadas")
    ap.add_argument("--n-odio", type=int, help="sementes de odio (substitui n//2)")
    ap.add_argument("--n-nao", type=int, help="sementes de nao odio (substitui n//2)")
    ap.add_argument("--out", default="data/processed/synthetic_pilot.jsonl")
    a = ap.parse_args()
    {"gen": cmd_gen, "check": cmd_check, "filter": cmd_filter}[a.cmd](a)
