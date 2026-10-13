"""Contrato de seguranca da API: chave, limite de uso, erro padronizado e log sem texto."""
import hashlib
import logging

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.security import zerar_uso

CHAVE = "chave-de-teste"
SEGREDO = "um texto muito particular que nao pode aparecer no log"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("HSC_API_KEYS", "teste:" + hashlib.sha256(CHAVE.encode()).hexdigest())
    monkeypatch.delenv("HSC_API_AUTH", raising=False)
    monkeypatch.setenv("HSC_RATE_PER_MIN", "5")
    zerar_uso()
    with TestClient(app) as c:
        yield c


def _post(client, rota, corpo, chave=CHAVE):
    return client.post(rota, json=corpo, headers={"X-API-Key": chave} if chave else {})


def test_health_aberto_com_aviso(client):
    r = client.get("/health")
    assert r.status_code == 200 and "notice" in r.json()


def test_sem_chave_e_chave_errada(client):
    assert _post(client, "/v1/predict", {"text": "oi"}, chave=None).status_code == 401
    r = _post(client, "/v1/predict", {"text": "oi"}, chave="errada")
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"


def test_sem_nenhuma_chave_configurada(client, monkeypatch):
    monkeypatch.delenv("HSC_API_KEYS")
    assert _post(client, "/v1/predict", {"text": "oi"}).status_code == 503


def test_predict_com_chave(client):
    r = _post(client, "/v1/predict", {"text": "bom dia a todos"})
    assert r.status_code == 200 and r.json()["label"] in ("hate", "not_hate", "non_hate")


def test_limite_por_minuto_com_retry_after(client):
    for _ in range(5):
        assert _post(client, "/v1/predict", {"text": "oi"}).status_code == 200
    r = _post(client, "/v1/predict", {"text": "oi"})
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    assert r.json()["error"]["code"] == "rate_limit_exceeded"


def test_lote_gasta_uma_unidade_por_texto(client):
    assert _post(client, "/v1/predict_batch", {"texts": ["a b"] * 4}).status_code == 200
    assert _post(client, "/v1/predict_batch", {"texts": ["a b"] * 2}).status_code == 429


def test_validacao_nao_devolve_a_entrada(client):
    r = _post(client, "/v1/predict_batch", {"texts": [SEGREDO * 100]})
    assert r.status_code == 422 and SEGREDO not in r.text


def test_log_nao_tem_o_texto(client, caplog):
    with caplog.at_level(logging.INFO, logger="hsc.api"):
        _post(client, "/v1/predict", {"text": SEGREDO})
        _post(client, "/v1/predict_batch", {"texts": [SEGREDO]})
    assert "predict" in caplog.text and SEGREDO not in caplog.text
