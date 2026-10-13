"""Logica do scripts/translate_gemini.py com a API simulada: nenhuma chamada real, nenhuma chave."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import translate_gemini as tg  # noqa: E402


class Resp:
    def __init__(self, status=200, corpo=None):
        self.status_code, self._c, self.text = status, corpo or {}, json.dumps(corpo or {})

    def json(self):
        return self._c


def ok_json(itens_traduzidos):
    return Resp(200, {"candidates": [{"finishReason": "STOP", "content": {"parts": [
        {"text": json.dumps(itens_traduzidos)}]}}]})


@pytest.fixture(autouse=True)
def sem_espera(monkeypatch):
    monkeypatch.setattr(tg.time, "sleep", lambda s: None)


def test_escolher_modelo_nunca_o_proibido():
    assert tg.escolher_modelo(["gemini-2.5-flash", "gemini-3.5-flash", "gemini-3-flash"]) == "gemini-3.5-flash"
    nomes = ["gemini-2.5-flash", "gemini-3-flash", "gemini-3-flash-lite", "gemini-3.1-flash-preview", "gemini-2.5-pro"]
    assert tg.escolher_modelo(nomes) == "gemini-3-flash"
    with pytest.raises(SystemExit):
        tg.escolher_modelo(["gemini-2.5-flash", "gemini-2.0-flash"])


def test_chaves_extraidas_do_arquivo_sem_vazar(monkeypatch, tmp_path, capsys):
    k1, k2 = "AIza" + "a" * 35, "AIza" + "b" * 35
    arq = tmp_path / "k.txt"
    arq.write_text(f"minha chave: {k1}\nnome=x {k2}\n{k1}\n", encoding="utf-8")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_KEYS_FILE", str(arq))
    assert tg.carregar_chaves() == [k1, k2]
    assert k1 not in capsys.readouterr().out
    monkeypatch.delenv("GEMINI_KEYS_FILE")
    with pytest.raises(SystemExit):
        tg.carregar_chaves()


def test_status_ok_vazio_bloqueado_json_ruim(monkeypatch):
    itens = [{"id": "a", "texto": "x"}, {"id": "b", "texto": "y"}, {"id": "c", "texto": "z"}]
    pool = tg.Pool(["k1"], False)
    monkeypatch.setattr(tg.requests, "request", lambda *a, **k: ok_json(
        [{"id": "a", "traducao": "A"}, {"id": "b", "traducao": ""}]))
    r = tg.chamar(pool, "m", itens)
    assert r == {"a": ("A", "ok"), "b": ("", "vazio"), "c": ("", "erro_json")}
    monkeypatch.setattr(tg.requests, "request", lambda *a, **k: Resp(200, {"candidates": [
        {"finishReason": "SAFETY"}]}))
    assert set(s for _, s in tg.chamar(pool, "m", itens).values()) == {"bloqueado"}
    monkeypatch.setattr(tg.requests, "request", lambda *a, **k: Resp(200, {"promptFeedback": {"blockReason": "OTHER"}}))
    assert set(s for _, s in tg.chamar(pool, "m", itens).values()) == {"bloqueado"}
    monkeypatch.setattr(tg.requests, "request", lambda *a, **k: Resp(200, {"candidates": [
        {"content": {"parts": [{"text": "isto nao e json"}]}}]}))
    assert set(s for _, s in tg.chamar(pool, "m", itens).values()) == {"erro_json"}


def test_corpo_tem_filtros_permissivos_e_temperatura_baixa(monkeypatch):
    visto = {}

    def falso(metodo, url, headers=None, json=None, timeout=None):
        visto.update(json=json, headers=headers)
        return ok_json([{"id": "a", "traducao": "A"}])
    monkeypatch.setattr(tg.requests, "request", falso)
    tg.chamar(tg.Pool(["k1"], False), "m", [{"id": "a", "texto": "x"}])
    assert all(s["threshold"] == "BLOCK_NONE" for s in visto["json"]["safetySettings"])
    assert visto["json"]["generationConfig"]["temperature"] <= 0.3
    assert visto["json"]["generationConfig"]["responseMimeType"] == "application/json"
    assert visto["headers"]["x-goog-api-key"] == "k1"  # chave so no cabecalho, nao na URL


def test_rotacao_so_quando_pedida(monkeypatch):
    usadas = []

    def falso(metodo, url, headers=None, json=None, timeout=None):
        usadas.append(headers["x-goog-api-key"])
        return Resp(429) if headers["x-goog-api-key"] == "k1" else ok_json([{"id": "a", "traducao": "A"}])
    monkeypatch.setattr(tg.requests, "request", falso)
    with pytest.raises(tg.SemCota):
        tg.Pool(["k1", "k2"], False).pedir("POST", "x", {})
    assert set(usadas) == {"k1"}
    usadas.clear()
    p = tg.Pool(["k1", "k2"], True)
    assert p.pedir("POST", "x", {}) and usadas[-1] == "k2" and p.trocou and p.slot == 2
    with pytest.raises(tg.SemCota):  # todas esgotadas
        monkeypatch.setattr(tg.requests, "request", lambda *a, **k: Resp(429))
        tg.Pool(["k1", "k2"], True).pedir("POST", "x", {})


def _ambiente(monkeypatch, tmp_path, n=120):
    df = pd.DataFrame({"id": [f"id{i}" for i in range(n)], "text": [f"text {i}" for i in range(n)],
                       "text_clean": "", "label": [i % 2 for i in range(n)], "source_dataset": "vidgen",
                       "engine": ["marianmt" if i % 3 else "qwen2.5-7b-instruct" for i in range(n)]})
    df.to_parquet(tmp_path / "piloto.parquet")
    monkeypatch.setattr(tg, "PILOTO", tmp_path / "piloto.parquet")
    monkeypatch.setattr(tg, "SAIDA", tmp_path / "saida.jsonl")
    monkeypatch.setattr(tg, "CHECKPOINTS", tmp_path / "ckpt.jsonl")
    monkeypatch.setenv("GEMINI_API_KEY", "k1")
    monkeypatch.delenv("GEMINI_KEYS_FILE", raising=False)

    def falso(metodo, url, headers=None, json=None, timeout=None):
        if metodo == "GET":
            return Resp(200, {"models": [{"name": "models/gemini-3.5-flash",
                                          "supportedGenerationMethods": ["generateContent"]}]})
        itens = __import__("json").loads(json["contents"][0]["parts"][0]["text"])
        return ok_json([{"id": i["id"], "traducao": "PT " + i["texto"]} for i in itens])
    monkeypatch.setattr(tg.requests, "request", falso)


def test_completo_checkpoint_a_cada_50_e_retomada(monkeypatch, tmp_path):
    _ambiente(monkeypatch, tmp_path)
    args = SimpleNamespace(piloto=50, completo=True, modelo=None, rotacionar=False)
    tg.cmd_traduzir(args)
    feitos = tg.carregar_feitos()
    assert len(feitos) == 120 and {f["modelo"] for f in feitos.values()} == {"gemini-3.5-flash"}
    motivos = [json.loads(l)["motivo"] for l in (tmp_path / "ckpt.jsonl").read_text().splitlines()]
    assert motivos.count("50 traducoes") == 2 and motivos[-1] == "fim"
    antes = (tmp_path / "saida.jsonl").read_text()
    tg.cmd_traduzir(args)  # retomada: nada pendente, nada reescrito
    assert (tmp_path / "saida.jsonl").read_text() == antes


def test_piloto_estratificado_e_ids_do_piloto(monkeypatch, tmp_path):
    _ambiente(monkeypatch, tmp_path)
    tg.cmd_traduzir(SimpleNamespace(piloto=50, completo=False, modelo=None, rotacionar=False))
    ids = set(tg.carregar_feitos())
    base = set(pd.read_parquet(tmp_path / "piloto.parquet")["id"])
    assert len(ids) == 50 and ids <= base
