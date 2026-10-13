"""Cria os projetos de validacao no doccano do servidor e sobe os textos.

- Na tela do anotador aparece SO o texto (sem rotulo da LLM, sem origem, sem original em EN).
- O mapeamento id_doccano -> {rotulo da LLM, origem, modelo, ...} fica FORA do doccano e
  FORA do repositorio, em MAP_DIR.
- Senhas por variavel de ambiente (ver _acessos.py); nada e impresso.

Uso: python deploy_projects.py traducao|sintetico
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests

from _acessos import senha

BASE = os.environ.get("DOCCANO_URL", "http://127.0.0.1:8010")
# mapeamento (rotulo da LLM, origem) fica FORA do repositorio
MAP_DIR = Path(os.environ.get("LUCIOLA_MAP_DIR", Path.home() / "luciola_mapeamentos"))
HSP = Path(__file__).resolve().parents[2]

COMUM = (
    "\n\nORDEM: a sequência dos itens é fixa. Isa e Vini anotam os 100 PRIMEIROS itens "
    "(os mesmos para os dois, na ordem em que aparecem) antes de qualquer outro."
)
AVISO = (
    "AVISO DE CONTEÚDO SENSÍVEL: os textos deste projeto incluem discurso de ódio, "
    "insultos e estereótipos contra grupos de pessoas, reproduzidos para fins de pesquisa. "
    "Faça pausas (sugestão: a cada 20 itens) e pare se sentir desconforto.\n\n"
)
DIRETRIZES = {
    "traducao": AVISO + (
        "Cada texto é uma tradução automática para o português do Brasil de um post em ingles.\n"
        "1. Marque EXATAMENTE UM entre 'Ódio' e 'Não ódio', lendo o texto em português.\n"
        "2. Marque '⚑ Tradução estranha' (opcional) se a tradução estiver incoerente, suavizada, "
        "inventada ou não soar como português."
    ),
    "sintetico": AVISO + (
        "Cada texto foi gerado por um modelo de linguagem, imitando um post de rede social.\n"
        "1. Marque EXATAMENTE UM entre 'Ódio' e 'Não ódio', lendo o texto.\n"
        "2. Marque '⚑ Texto artificial' (opcional) se o texto soar gerado por máquina, "
        "pouco natural ou fora do jeito de falar de uma rede social brasileira."
    ),
}
LABELS = {
    "traducao": ["Ódio", "Não ódio", "⚑ Tradução estranha"],
    "sintetico": ["Ódio", "Não ódio", "⚑ Texto artificial"],
}
NOMES = {"traducao": "Piloto tradução PT", "sintetico": "Piloto sintético PT"}
CORES = ["#d9534f", "#5cb85c", "#f0ad4e"]
MEMBROS = {"isa": "project_admin", "vini": "project_admin",
           "anotador1": "annotator", "anotador2": "annotator"}


def login(user, pw):
    s = requests.Session()
    s.post(f"{BASE}/v1/auth/login/", json={"username": user, "password": pw}).raise_for_status()
    s.headers.update({"X-CSRFToken": s.cookies.get("csrftoken", ""), "Referer": BASE + "/"})
    return s


def ordenar(df, chaves, n_comum=100, seed=42):
    """Os n_comum primeiros itens sao uma amostra estratificada (mesmas proporcoes das
    chaves); o resto vem embaralhado. A ordem e fixa (random_order=False), entao todos
    os membros veem os mesmos 100 primeiros itens, na mesma sequencia."""
    partes = [g.sample(max(1, round(len(g) * n_comum / len(df))), random_state=seed)
              for _, g in df.groupby(chaves)]
    comum = pd.concat(partes).sample(frac=1, random_state=seed).head(n_comum)
    resto = df.drop(comum.index).sample(frac=1, random_state=seed)
    return pd.concat([comum, resto])


def itens(kind):
    """(texto, mapeamento) por item."""
    if kind == "traducao":
        df = pd.read_parquet(HSP / "data/processed/translation_pilot_500_out.parquet")
        assert df["text_pt"].notna().all()
        df = ordenar(df, ["engine", "label"])
        for _, r in df.iterrows():
            yield r["text_pt"], {
                "id_corpus": r["id"], "rotulo_llm": int(r["label"]),
                "rotulo_llm_nota": "rotulo original da fonte, nao de LLM",
                "origem": r["source_dataset"], "modelo": r["engine"], "texto_en": r["text"],
                "flag_refusal": bool(r["flag_refusal"]), "flag_short": bool(r["flag_short"])}
    else:
        df = pd.read_json(HSP / "data/processed/synthetic_pilot.jsonl", lines=True)
        df = df[~df["flag_refusal"]]
        df = ordenar(df, ["label", "seed_source"])
        for _, r in df.iterrows():
            yield r["text_pt"], {
                "id_corpus": r["seed_id"], "rotulo_llm": int(r["label"]),
                "rotulo_llm_nota": "classe pedida ao gerador",
                "origem": r["seed_source"], "modelo": "qwen2.5:7b-instruct"}


def main(kind):
    s = login("adm", senha("adm"))
    p = s.post(f"{BASE}/v1/projects", json={
        "name": NOMES[kind], "description": "Validação humana (Luciola)",
        "resourcetype": "TextClassificationProject", "project_type": "DocumentClassification", "guideline": DIRETRIZES[kind] + COMUM,
        "random_order": False, "collaborative_annotation": False,
        "single_class_classification": False}).json()
    if "id" not in p:
        sys.exit(f"falha ao criar projeto: {p}")
    pid = p["id"]
    assert p["collaborative_annotation"] is False
    for text, color in zip(LABELS[kind], CORES):
        s.post(f"{BASE}/v1/projects/{pid}/category-types",
               json={"text": text, "background_color": color, "text_color": "#ffffff"}).raise_for_status()
    roles = {r["name"]: r["id"] for r in s.get(f"{BASE}/v1/roles").json()}
    users = {u["username"]: u["id"] for u in s.get(f"{BASE}/v1/users").json()}
    for u, role in MEMBROS.items():
        s.post(f"{BASE}/v1/projects/{pid}/members",
               json={"user": users[u], "role": roles[role], "project": pid}).raise_for_status()
    mapa, falhas = {}, 0
    for text, m in itens(kind):
        r = s.post(f"{BASE}/v1/projects/{pid}/examples", json={"text": text})
        if r.status_code == 201:
            mapa[r.json()["id"]] = m
        else:
            falhas += 1
    MAP_DIR.mkdir(exist_ok=True)
    (MAP_DIR / f"{kind}_projeto{pid}.json").write_text(
        json.dumps({"project_id": pid, "kind": kind, "itens": mapa}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"projeto {pid} ({kind}): {len(mapa)} enviados, {falhas} falhas; mapeamento em {MAP_DIR}")


if __name__ == "__main__":
    main(sys.argv[1])
