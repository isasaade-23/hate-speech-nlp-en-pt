"""Credenciais por variavel de ambiente (nunca no repositorio).

    DOCCANO_PW_ADM, DOCCANO_PW_ISA, DOCCANO_PW_VINI, DOCCANO_PW_ANOTADOR1, DOCCANO_PW_ANOTADOR2

Se uma delas faltar e LUCIOLA_ACESSOS apontar para um arquivo FORA do repositorio com linhas
"usuario<TAB>senha<TAB>...", a senha e lida de la. Nada e impresso.
"""
import os
from pathlib import Path


def senha(user: str) -> str:
    v = os.environ.get(f"DOCCANO_PW_{user.upper()}")
    if v:
        return v
    arq = os.environ.get("LUCIOLA_ACESSOS")
    if arq:
        for line in Path(arq).read_text(encoding="utf-8").splitlines():
            p = line.split("\t")
            if len(p) >= 2 and p[0] == user:
                return p[1]
    raise SystemExit(f"senha de {user!r} nao encontrada: defina DOCCANO_PW_{user.upper()}")
