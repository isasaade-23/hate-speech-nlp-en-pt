"""Chave de acesso e limite de uso da API.

Chaves: variavel de ambiente HSC_API_KEYS, "nome:sha256hex" separados por virgula. O servidor
guarda so o hash; a chave em claro vai para quem a recebe. Gerar uma: python -m api.security NOME.
Sem nenhuma chave configurada a API recusa tudo (503), a menos que HSC_API_AUTH=off (dev/testes).

Limite: unidades por minuto por chave (HSC_RATE_PER_MIN, padrao 600); /predict gasta 1 unidade,
/predict_batch gasta len(texts). Janela deslizante em memoria (um processo).
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sys
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException

_JANELA_S = 60.0
_uso: dict[str, deque] = defaultdict(deque)


def _hash(chave: str) -> str:
    return hashlib.sha256(chave.encode("utf-8")).hexdigest()


def _chaves() -> dict[str, str]:
    """hash -> nome."""
    out = {}
    for item in os.environ.get("HSC_API_KEYS", "").split(","):
        if ":" in item:
            nome, h = item.strip().split(":", 1)
            out[h.lower()] = nome
    return out


def autenticar(x_api_key: str | None = Header(default=None)) -> str:
    """Dependencia FastAPI: devolve o nome do cliente ou levanta 401/503."""
    if os.environ.get("HSC_API_AUTH", "on").lower() == "off":
        return "dev"
    chaves = _chaves()
    if not chaves:
        raise HTTPException(503, "api_key_not_configured")
    if not x_api_key:
        raise HTTPException(401, "missing_api_key")
    h = _hash(x_api_key)
    for conhecido, nome in chaves.items():
        if hmac.compare_digest(h, conhecido):
            return nome
    raise HTTPException(401, "invalid_api_key")


def gastar(cliente: str, unidades: int) -> None:
    """Registra o uso; levanta 429 (com Retry-After) se passar do limite por minuto."""
    limite = int(os.environ.get("HSC_RATE_PER_MIN", "600"))
    agora = time.monotonic()
    fila = _uso[cliente]
    while fila and agora - fila[0][0] > _JANELA_S:
        fila.popleft()
    usado = sum(u for _, u in fila)
    if usado + unidades > limite:
        espera = int(_JANELA_S - (agora - fila[0][0])) + 1 if fila else 1
        raise HTTPException(429, "rate_limit_exceeded", headers={"Retry-After": str(espera)})
    fila.append((agora, unidades))


def zerar_uso() -> None:
    _uso.clear()


if __name__ == "__main__":
    nome = sys.argv[1] if len(sys.argv) > 1 else "cliente"
    chave = secrets.token_urlsafe(32)
    print(f"Chave (entregue ao cliente, nao fica no servidor): {chave}")
    print(f"HSC_API_KEYS (acrescente ao servidor): {nome}:{_hash(chave)}")
