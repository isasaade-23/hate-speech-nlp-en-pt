"""Configura o projeto doccano via API: labels, membros anotadores, e sobe os
itens de teste/piloto. Roda contra o servidor local ja no ar (porta 8010).
"""
import os
import sys
import requests

BASE = "http://127.0.0.1:8010"
ADMIN = ("admin", os.environ["DOCCANO_ADMIN_PW"])
PROJECT_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def login(user, pw):
    s = requests.Session()
    s.post(f"{BASE}/v1/auth/login/", json={"username": user, "password": pw})
    s.csrf = s.cookies.get("csrftoken")
    return s


def h(s):
    return {"X-CSRFToken": s.csrf, "Referer": BASE + "/"}


def main():
    s = login(*ADMIN)

    # 1. labels: Odio/Nao odio + LLM acertou/errou, cores diferentes pros dois grupos
    labels = [
        {"text": "Ódio", "background_color": "#d9534f"},
        {"text": "Não ódio", "background_color": "#5cb85c"},
        {"text": "LLM acertou", "background_color": "#5bc0de"},
        {"text": "LLM errou", "background_color": "#f0ad4e"},
    ]
    label_ids = {}
    for lab in labels:
        r = s.post(f"{BASE}/v1/projects/{PROJECT_ID}/category-types", headers=h(s), json=lab)
        if r.status_code == 201:
            label_ids[lab["text"]] = r.json()["id"]
            print("label criado:", lab["text"], r.json()["id"])
        else:
            print("label FALHOU:", lab["text"], r.status_code, r.text[:200])

    # 2. roles disponiveis
    r = s.get(f"{BASE}/v1/roles", headers=h(s))
    roles = {x["name"]: x["id"] for x in r.json()}
    print("roles:", roles)

    # 3. membros: anotador1 e anotador2 como 'annotator'
    for username in ("anotador1", "anotador2"):
        # precisa do user id -- users endpoint
        ru = s.get(f"{BASE}/v1/users", headers=h(s))
        users = {u["username"]: u["id"] for u in ru.json()}
        payload = {"user": users[username], "role": roles["annotator"]}
        r = s.post(f"{BASE}/v1/projects/{PROJECT_ID}/members", headers=h(s), json=payload)
        print("membro:", username, r.status_code, r.text[:200])

    print("\nOK. label_ids:", label_ids)


if __name__ == "__main__":
    main()
