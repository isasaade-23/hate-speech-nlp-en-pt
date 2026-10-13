"""Renomeia o projeto de traducao v1 no doccano e o marca como pausado; renomeia o mapeamento.

Uso: python pausar_v1.py <project_id>   (senhas e URL por variavel de ambiente; ver _acessos.py)
"""
import json
import sys

import deploy_projects as d
from _acessos import senha

NOME = "Tradução v1 (MarianMT+Qwen) – pausado"
DESCRICAO = ("PAUSADO. Primeira tradução do piloto (248 textos pelo MarianMT e 252 pelo Qwen2.5-7B), "
             "considerada insuficiente. Substituída pelo projeto Tradução v2 (Gemini). Não anotar aqui; "
             "o que já foi anotado fica preservado.")

pid = int(sys.argv[1])
s = d.login("adm", senha("adm"))
r = s.patch(f"{d.BASE}/v1/projects/{pid}", json={
    "resourcetype": "TextClassificationProject", "name": NOME, "description": DESCRICAO})
print("projeto", pid, r.status_code)

antigo = d.MAP_DIR / f"traducao_projeto{pid}.json"
if antigo.exists():
    m = json.loads(antigo.read_text(encoding="utf-8"))
    m["kind"] = "traducao_v1"
    novo = d.MAP_DIR / f"traducao_v1_projeto{pid}.json"
    novo.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    antigo.unlink()
    print("mapeamento:", novo.name)
