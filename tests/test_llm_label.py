"""Lógica do scripts/llm_label.py com a API simulada: nenhuma chamada real, nenhuma chave real."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import llm_label as ll  # noqa: E402

CHAVE_M = "mistralFAKE0123456789abcdefXYZ"
CHAVE_G = "AIza" + "F" * 35


class Resp:
    def __init__(self, status=200, corpo=None):
        self.status_code, self._c, self.text = status, corpo or {}, json.dumps(corpo or {})

    def json(self):
        return self._c


def item(i, rot=1, prob=0.9, cat="religiao"):
    return {"id": i, "rotulo": rot, "prob": prob, "categoria": cat, "justificativa_curta": "ataque"}


def resp_gemini(itens, finish="STOP"):
    return Resp(200, {"modelVersion": "gemini-3-flash", "candidates": [{"finishReason": finish,
            "content": {"parts": [{"text": json.dumps({"itens": itens})}]}}]})


def resp_mistral(itens):
    return Resp(200, {"model": "mistral-large-2511", "choices": [{"message": {
        "content": json.dumps({"itens": itens})}}]})


@pytest.fixture(autouse=True)
def sem_espera(monkeypatch):
    monkeypatch.setattr(ll.time, "sleep", lambda s: None)
    monkeypatch.setattr(ll.tg.time, "sleep", lambda s: None)
    for v in ("MISTRAL_API_KEY", "MISTRAL_KEYS_FILE", "GEMINI_API_KEY", "GEMINI_KEYS_FILE"):
        monkeypatch.delenv(v, raising=False)


def test_chave_mistral_na_linha_abaixo(monkeypatch, tmp_path):
    arq = tmp_path / "keys.txt"
    arq.write_text(f"gemini\n{CHAVE_G}\n\nMistral (conta pessoal)\n\n  {CHAVE_M}  \noutra\nxyz\n",
                   encoding="utf-8")
    monkeypatch.setenv("MISTRAL_KEYS_FILE", str(arq))
    assert ll.chave_de_arquivo("mistral") == CHAVE_M


def test_chave_mistral_ausente_nao_vaza(monkeypatch, tmp_path, capsys):
    arq = tmp_path / "keys.txt"
    arq.write_text(f"mistral\ncurta\n{CHAVE_G}\n", encoding="utf-8")
    monkeypatch.setenv("MISTRAL_KEYS_FILE", str(arq))
    with pytest.raises(SystemExit) as e:
        ll.chave_de_arquivo("mistral")
    assert "curta" not in str(e.value) and CHAVE_G not in str(e.value)


def test_esquema_validado_item_a_item():
    dados = {"itens": [item("a"), item("b", rot=2), item("c", cat="inventada"),
                       {"id": "d", "rotulo": 0}, item("x")]}
    res = ll.validar(dados, ["a", "b", "c", "d", "e"])
    assert res["a"][1] == "ok" and res["a"][0]["prob"] == 0.9
    assert [res[i][1] for i in "bcde"] == ["erro_json"] * 4
    assert "x" not in res
    assert ll.validar({"outra": 1}, ["a"])["a"][1] == "erro_json"


def test_gemini_bloqueado(monkeypatch):
    monkeypatch.setattr(ll.tg.requests, "request", lambda *a, **k: resp_gemini([], finish="SAFETY"))
    pool = ll.tg.Pool([CHAVE_G], rotacionar=False)
    res, _ = ll.chamar("gemini", pool, "gemini-3-flash", [{"id": "a", "texto": "t"}])
    assert res["a"][1] == "bloqueado"


def test_mistral_corpo_temperatura_zero_e_esquema(monkeypatch):
    visto = {}

    def fake(metodo, url, json=None, headers=None, timeout=None):
        visto.update(url=url, corpo=json, headers=headers)
        return resp_mistral([item("a", rot=0, prob=0.1, cat="nenhuma")])

    monkeypatch.setattr(ll.requests, "request", fake)
    res, modelo = ll.chamar("mistral", ll.OpenAICompat("mistral", CHAVE_M), "mistral-large-latest",
                            [{"id": "a", "texto": "t"}])
    assert res["a"] == ({"rotulo": 0, "prob": 0.1, "categoria": "nenhuma",
                         "justificativa_curta": "ataque"}, "ok")
    assert modelo == "mistral-large-2511"
    assert visto["corpo"]["temperature"] == 0
    assert visto["corpo"]["response_format"]["type"] == "json_schema"
    assert visto["url"].endswith("/chat/completions")


def test_mistral_tenta_de_novo_em_429(monkeypatch):
    respostas = iter([Resp(429), Resp(503), resp_mistral([item("a")])])
    monkeypatch.setattr(ll.requests, "request", lambda *a, **k: next(respostas))
    res, _ = ll.chamar("mistral", ll.OpenAICompat("mistral", CHAVE_M), "m", [{"id": "a", "texto": "t"}])
    assert res["a"][1] == "ok"


def preparar(monkeypatch, tmp_path):
    corpus = tmp_path / "corpus.parquet"
    pd.DataFrame({"id": [f"i{k}" for k in range(25)], "text_clean": [f"texto {k}" for k in range(25)]}
                 ).to_parquet(corpus)
    ids = tmp_path / "ids.csv"
    pd.DataFrame({"id": [f"i{k}" for k in range(25)]}).to_csv(ids, index=False)
    monkeypatch.setattr(ll, "CORPUS", corpus)
    monkeypatch.setattr(ll, "saida", lambda tag, p: tmp_path / f"out_{tag}_{p}.jsonl")
    monkeypatch.setenv("MISTRAL_API_KEY", CHAVE_M)
    return ids


def test_retomada_nao_refaz_ok_e_nao_vaza_chave(monkeypatch, tmp_path, capsys):
    ids = preparar(monkeypatch, tmp_path)
    chamadas = []

    def fake(metodo, url, json=None, headers=None, timeout=None):
        entrada = __import__("json").loads(json["messages"][1]["content"])
        chamadas.append([e["id"] for e in entrada])
        # i3 volta sempre sem esquema válido
        return resp_mistral([item(e["id"]) for e in entrada if e["id"] != "i3"])

    monkeypatch.setattr(ll.requests, "request", fake)
    args = ["rotular", "--provedor", "mistral", "--ids", str(ids), "--modelo", "mistral-large-latest"]
    ll.main(args + ["--n", "12"])
    feitos = ll.carregar_feitos(tmp_path / "out_juiz_pt_mistral.jsonl")
    assert len(feitos) == 12 and feitos["i3"]["status"] == "erro_json"
    assert feitos["i0"]["modelo"] == "mistral-large-2511" and feitos["i0"]["prompt"] == ll.PROMPT_VERSAO

    chamadas.clear()
    ll.main(args)  # completo: refaz só i3 e faz os 13 novos
    pedidos = {i for c in chamadas for i in c}
    assert pedidos == {"i3"} | {f"i{k}" for k in range(12, 25)}

    with pytest.raises(SystemExit):
        ll.main(["rotular", "--provedor", "mistral", "--ids", str(ids), "--modelo", "outro"])

    saida = capsys.readouterr().out + (tmp_path / "out_juiz_pt_mistral.jsonl").read_text(encoding="utf-8")
    assert CHAVE_M not in saida


def test_groq_mesma_interface(monkeypatch, tmp_path):
    arq = tmp_path / "keys.txt"
    arq.write_text(f"mistral\n{CHAVE_M}\nGroq\ngsk_FAKE0123456789abcdefghij\n", encoding="utf-8")
    monkeypatch.setenv("GROQ_KEYS_FILE", str(arq))
    assert ll.chave_de_arquivo("groq") == "gsk_FAKE0123456789abcdefghij"
    visto = {}

    def fake(metodo, url, json=None, headers=None, timeout=None):
        visto["url"] = url
        return resp_mistral([item("a")])

    monkeypatch.setattr(ll.requests, "request", fake)
    res, _ = ll.chamar("groq", ll.cliente("groq"), "openai/gpt-oss-120b", [{"id": "a", "texto": "t"}])
    assert res["a"][1] == "ok" and visto["url"].startswith("https://api.groq.com/openai/v1/")
