"""Teste de anotacao cega no servidor, em projeto descartavel (criado e apagado aqui).
Usa anotador1/anotador2 reais; credenciais do arquivo fora do repositorio."""
import os

import requests

from _acessos import senha

URL = os.environ["DOCCANO_URL"]
BASE = URL + "/v1"


def login(u):
    s = requests.Session()
    s.post(f"{BASE}/auth/login/", json={"username": u, "password": senha(u)}).raise_for_status()
    s.headers.update({"X-CSRFToken": s.cookies.get("csrftoken", ""), "Referer": URL + "/"})
    return s


adm, a, b = login("adm"), login("anotador1"), login("anotador2")
p = adm.post(f"{BASE}/projects", json={
    "name": "tmp_blind", "description": "descartavel", "resourcetype": "TextClassificationProject",
    "project_type": "DocumentClassification", "guideline": "", "random_order": False,
    "collaborative_annotation": False, "single_class_classification": False}).json()
pid = p["id"]
try:
    lab = adm.post(f"{BASE}/projects/{pid}/category-types", json={"text": "X", "background_color": "#ff0000", "text_color": "#ffffff"}).json()
    roles = {r["name"]: r["id"] for r in adm.get(f"{BASE}/roles").json()}
    users = {u["username"]: u["id"] for u in adm.get(f"{BASE}/users").json()}
    for u in ("anotador1", "anotador2"):
        adm.post(f"{BASE}/projects/{pid}/members", json={"user": users[u], "role": roles["annotator"], "project": pid}).raise_for_status()
    adm.post(f"{BASE}/projects/{pid}/examples", json={"text": "texto de teste"}).raise_for_status()
    ex = a.get(f"{BASE}/projects/{pid}/examples").json()["results"][0]["id"]
    a.post(f"{BASE}/projects/{pid}/examples/{ex}/categories", json={"label": lab["id"]}).raise_for_status()
    va = len(a.get(f"{BASE}/projects/{pid}/examples/{ex}/categories").json())
    vb = len(b.get(f"{BASE}/projects/{pid}/examples/{ex}/categories").json())
    print(f"collaborative={p['collaborative_annotation']}  anotador1 ve {va}, anotador2 ve {vb} ->",
          "CEGO OK" if (va, vb) == (1, 0) else "FALHOU")
finally:
    adm.delete(f"{BASE}/projects/{pid}")
