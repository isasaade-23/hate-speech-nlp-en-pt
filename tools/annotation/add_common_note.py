"""Acrescenta a nota da amostra comum (100 primeiros itens de Isa e Vini) as diretrizes
dos projetos ja criados, sem apagar nada. Idempotente."""
import deploy_projects as d
from _acessos import senha

s = d.login("adm", senha("adm"))
for pid in (1, 2):
    p = s.get(f"{d.BASE}/v1/projects/{pid}").json()
    if d.COMUM.strip() in p["guideline"]:
        print(pid, "ja tem a nota")
        continue
    r = s.patch(f"{d.BASE}/v1/projects/{pid}", json={
        "resourcetype": "TextClassificationProject", "guideline": p["guideline"] + d.COMUM})
    print(pid, r.status_code, "random_order =", s.get(f"{d.BASE}/v1/projects/{pid}").json()["random_order"])
