"""Sobe os 500 itens do piloto de traducao (data/processed/translation_pilot_500_out.parquet,
no repo hate-speech-project) para o projeto doccano, com metadado visivel:
  - origem: fonte do corpus (vidgen/hatexplain/multioff)
  - engine: motor de traducao (marianmt / qwen2.5-7b-instruct)
  - rotulo_original: odio/nao odio, segundo o corpus fonte (label=1/0)
  - texto_en: o original em ingles, para comparacao lado a lado

So roda depois que translate_pilot.py tiver terminado os 500. Idempotente? Nao --
roda uma vez so (ou limpe o projeto antes de rodar de novo).

Uso: python upload_pilot.py [project_id]
"""
import os
import sys
import pandas as pd
import requests

BASE = "http://127.0.0.1:8010"
PROJECT_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 4
PARQUET = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed",
                       "translation_pilot_500_out.parquet")


def login(user, pw):
    s = requests.Session()
    s.post(f"{BASE}/v1/auth/login/", json={"username": user, "password": pw})
    s.csrf = s.cookies.get("csrftoken")
    return s


def h(s):
    return {"X-CSRFToken": s.csrf, "Referer": BASE + "/"}


def main():
    df = pd.read_parquet(PARQUET)
    missing = df["text_pt"].isna().sum()
    if missing:
        print(f"AVISO: {missing} itens ainda sem traducao. Rode translate_pilot.py ate 500/500 antes.")
        return

    s = login("admin", os.environ["DOCCANO_ADMIN_PW"])
    ok, fail = 0, 0
    for _, row in df.iterrows():
        meta = {
            "origem": row["source_dataset"],
            "engine": row["engine"],
            "rotulo_original": "ódio" if row["label"] == 1 else "não ódio",
            "texto_en": row["text"],
            "flag_refusal": bool(row["flag_refusal"]),
            "flag_short": bool(row["flag_short"]),
            "id_corpus": row["id"],
        }
        payload = {"text": row["text_pt"], "meta": meta}
        r = s.post(f"{BASE}/v1/projects/{PROJECT_ID}/examples", headers=h(s), json=payload)
        if r.status_code == 201:
            ok += 1
        else:
            fail += 1
            print("FALHOU:", row["id"], r.status_code, r.text[:200])
        if ok % 50 == 0 and ok:
            print(f"  {ok}/{len(df)} enviados")

    print(f"\nOK: {ok} enviados, {fail} falharam")


if __name__ == "__main__":
    main()
