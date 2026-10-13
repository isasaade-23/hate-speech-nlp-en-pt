"""Teste de anotacao cega: com collaborative_annotation=false, o anotador B nao ve o
rotulo posto pelo anotador A no mesmo item. Usa projeto descartavel e usuarios tmp_*.
Senha do admin via env DOCCANO_ADMIN_PW. Uso: python test_blind.py"""
import os
import secrets
import subprocess
import sys

import requests

BASE = "http://localhost:8010/v1"
ADM_PW = os.environ["DOCCANO_ADMIN_PW"]
DOCCANO = os.path.join(os.path.dirname(sys.executable), "doccano.exe")


def login(user, pw):
    s = requests.Session()
    r = s.post(f"{BASE}/auth/login/", json={"username": user, "password": pw})
    r.raise_for_status()
    return s


def hdr(s):
    return {"X-CSRFToken": s.cookies.get("csrftoken", "")}


pw = secrets.token_urlsafe(12) + "!A1"
for u in ("tmp_a", "tmp_b"):
    subprocess.run([DOCCANO, "createuser", "--username", u, "--password", pw,
                    "--email", f"{u}@x.local"], check=False, capture_output=True)

adm = login("admin", ADM_PW)
p = adm.post(f"{BASE}/projects", headers=hdr(adm), json={
    "name": "tmp_blind_test", "description": "descartavel", "project_type": "DocumentClassification",
    "guideline": "", "resourcetype": "TextClassificationProject", "random_order": False, "collaborative_annotation": False,
    "single_class_classification": False}).json()
print(p) if "id" not in p else None
pid = p["id"]
print("projeto", pid, "collaborative_annotation =", p["collaborative_annotation"])
lab = adm.post(f"{BASE}/projects/{pid}/category-types", headers=hdr(adm),
               json={"text": "Odio", "background_color": "#ff0000", "text_color": "#ffffff"}).json()
roles = {r["name"]: r["id"] for r in adm.get(f"{BASE}/roles", headers=hdr(adm)).json()}
users = {u["username"]: u["id"] for u in adm.get(f"{BASE}/users", headers=hdr(adm)).json()}
for u in ("tmp_a", "tmp_b"):
    adm.post(f"{BASE}/projects/{pid}/members", headers=hdr(adm),
             json={"user": users[u], "role": roles["annotator"], "project": pid}).raise_for_status()
adm.post(f"{BASE}/projects/{pid}/examples", headers=hdr(adm), json={"text": "texto de teste"}).raise_for_status()

a, b = login("tmp_a", pw), login("tmp_b", pw)
ex = a.get(f"{BASE}/projects/{pid}/examples", headers=hdr(a)).json()["results"][0]["id"]
a.post(f"{BASE}/projects/{pid}/examples/{ex}/categories", headers=hdr(a), json={"label": lab["id"]}).raise_for_status()
va = a.get(f"{BASE}/projects/{pid}/examples/{ex}/categories", headers=hdr(a)).json()
vb = b.get(f"{BASE}/projects/{pid}/examples/{ex}/categories", headers=hdr(b)).json()
print("A ve:", len(va), "anotacao(oes) | B ve:", len(vb))
adm.delete(f"{BASE}/projects/{pid}", headers=hdr(adm))
print("RESULTADO:", "CEGO OK" if len(va) == 1 and len(vb) == 0 else "FALHOU")
