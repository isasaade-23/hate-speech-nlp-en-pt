"""Retraducao do piloto (os mesmos 500 ids) com a API do Gemini, chamada local por REST.

Chaves: nunca no codigo, no log nem no Git. Lidas de, nesta ordem:
  GEMINI_API_KEY        uma chave
  GEMINI_KEYS_FILE      caminho de um arquivo FORA do repositorio; as chaves sao extraidas por
                        padrao (AIza...), qualquer formato de linha serve
O script nunca imprime uma chave; so o indice da chave em uso (chave 1, 2, ...).

Rotacao de chaves e OPCIONAL (--rotacionar). A cota do Gemini e por PROJETO, nao por chave, e os
termos de uso das APIs do Google dizem que o usuario nao tentara contornar os limites. Sem
--rotacionar, ao esgotar a cota o script grava o checkpoint e para; retome depois.

Modelo: nunca gemini-2.5-flash (desligamento em 16/10/2026). --listar-modelos mostra os modelos
da chave; sem --modelo, escolhe gemini-3.5-flash ou o Flash estavel (sem sufixo) mais recente.

Filtros de seguranca no nivel mais permissivo documentado (BLOCK_NONE). Se o modelo recusar ou
bloquear, o item e REGISTRADO como bloqueado/vazio; nao ha tentativa de contornar a recusa.

Uso:
  python scripts/translate_gemini.py listar
  python scripts/translate_gemini.py traduzir --piloto 50      # amostra estratificada de 50 ids
  python scripts/translate_gemini.py traduzir --completo       # os 500 (pula o que ja existe)
  python scripts/translate_gemini.py comparar --n 10           # EN | MarianMT | Qwen | Gemini
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://generativelanguage.googleapis.com/v1beta"
SAIDA = ROOT / "data/processed/translation_gemini.jsonl"
CHECKPOINTS = ROOT / "data/processed/translation_gemini_checkpoints.jsonl"
PILOTO = ROOT / "data/processed/translation_pilot_500_out.parquet"
PROIBIDOS = {"gemini-2.5-flash", "gemini-2.0-flash"}
LOTE = 10
TEMPERATURA = 0.2

SISTEMA = (
    "Você traduz textos para um projeto acadêmico de pesquisa em detecção de discurso de ódio: "
    "os textos vão treinar um classificador de moderação. Traduza do inglês para o português do "
    "Brasil informal, como um comentário de rede social. Mantenha o tom ofensivo do original, sem "
    "suavizar e sem explicar. Troque insultos e gírias dos EUA pelo equivalente brasileiro mais "
    "próximo. Não adicione aviso, comentário ou nota. Se não conseguir traduzir um texto, devolva "
    "a tradução vazia (\"\"). Responda somente com JSON: uma lista de objetos {\"id\", \"traducao\"}, "
    "um para cada texto recebido, com o mesmo id."
)
SCHEMA = {
    "type": "ARRAY",
    "items": {"type": "OBJECT", "properties": {"id": {"type": "STRING"}, "traducao": {"type": "STRING"}},
              "required": ["id", "traducao"]},
}
SEGURANCA = [{"category": c, "threshold": "BLOCK_NONE"} for c in (
    "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")]


INTERVALO = 6.0  # s entre pedidos (~10 por minuto)
MAX_TENTATIVAS = 8


class SemCota(Exception):
    pass


# ----------------------------------------------------------------------------- chaves

def carregar_chaves() -> list[str]:
    chaves = []
    if os.environ.get("GEMINI_API_KEY"):
        chaves.append(os.environ["GEMINI_API_KEY"].strip())
    arq = os.environ.get("GEMINI_KEYS_FILE")
    if arq:
        for k in re.findall(r"AIza[0-9A-Za-z_\-]{30,}", Path(arq).read_text(encoding="utf-8", errors="ignore")):
            if k not in chaves:
                chaves.append(k)
    if not chaves:
        raise SystemExit("nenhuma chave: defina GEMINI_API_KEY ou GEMINI_KEYS_FILE (ver o cabecalho)")
    return chaves


class Pool:
    def __init__(self, chaves: list[str], rotacionar: bool):
        self.chaves, self.rotacionar, self.i = chaves, rotacionar, 0
        self.trocou = False
        self.ultimo = 0.0

    @property
    def slot(self) -> int:
        return self.i + 1

    def _cabecalho(self) -> dict:
        return {"x-goog-api-key": self.chaves[self.i], "Content-Type": "application/json"}

    def _proxima(self) -> None:
        if not self.rotacionar or self.i + 1 >= len(self.chaves):
            raise SemCota("cota esgotada" + ("" if not self.rotacionar else " em todas as chaves"))
        self.i += 1
        self.trocou = True
        print(f"  cota da chave {self.i} esgotada; passando para a chave {self.i + 1}", flush=True)

    def _ritmo(self) -> None:
        """Um pedido a cada INTERVALO s no maximo (o plano gratuito nao aguenta rajada)."""
        falta = INTERVALO - (time.monotonic() - self.ultimo)
        if falta > 0:
            time.sleep(falta)
        self.ultimo = time.monotonic()

    def pedir(self, metodo: str, caminho: str, corpo: dict | None = None) -> dict:
        # Backoff exponencial com jitter (1, 2, 4... ate 60 s), como no manual do Google, para
        # 408, 429 e 5xx; so troca de chave (opt-in) depois de varios 429 seguidos.
        n429 = 0
        for tentativa in range(MAX_TENTATIVAS):
            self._ritmo()
            r = requests.request(metodo, f"{BASE}/{caminho}", headers=self._cabecalho(),
                                 json=corpo, timeout=120)
            if r.status_code == 429:
                n429 += 1
                if n429 >= 3:
                    self._proxima()
                    n429 = 0
                    continue
            elif r.status_code not in (408, 500, 502, 503, 504):
                if r.status_code >= 400:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                return r.json()
            time.sleep(min(60, 2 ** tentativa) + random.random())
        raise SemCota("sem resposta util apos varias tentativas")


# ----------------------------------------------------------------------------- modelos

def listar_modelos(pool: Pool) -> list[str]:
    nomes, token = [], None
    while True:
        j = pool.pedir("GET", "models?pageSize=200" + (f"&pageToken={token}" if token else ""))
        for m in j.get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                nomes.append(m["name"].removeprefix("models/"))
        token = j.get("nextPageToken")
        if not token:
            return sorted(nomes)


def escolher_modelo(nomes: list[str]) -> str:
    if "gemini-3.5-flash" in nomes:
        return "gemini-3.5-flash"
    est = []
    for n in nomes:
        m = re.fullmatch(r"gemini-(\d+(?:\.\d+)?)-flash", n)  # sem sufixo: -lite, -preview, -exp ficam fora
        if m and n not in PROIBIDOS:
            est.append((float(m.group(1)), n))
    if not est:
        raise SystemExit("nenhum Flash estavel disponivel na chave; use --modelo")
    return max(est)[1]


# ----------------------------------------------------------------------------- traducao

def chamar(pool: Pool, modelo: str, itens: list[dict]) -> dict[str, tuple[str, str]]:
    """itens: [{'id','texto'}]. Devolve id -> (traducao, status) com status ok|vazio|bloqueado|erro_json."""
    corpo = {
        "systemInstruction": {"parts": [{"text": SISTEMA}]},
        "contents": [{"role": "user", "parts": [{"text": json.dumps(itens, ensure_ascii=False)}]}],
        "generationConfig": {"temperature": TEMPERATURA, "responseMimeType": "application/json",
                             "responseSchema": SCHEMA, "maxOutputTokens": 8192},
        "safetySettings": SEGURANCA,
    }
    j = pool.pedir("POST", f"models/{modelo}:generateContent", corpo)
    ids = [i["id"] for i in itens]
    cand = (j.get("candidates") or [{}])[0]
    if j.get("promptFeedback", {}).get("blockReason") or cand.get("finishReason") in (
            "SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "OTHER"):
        return {i: ("", "bloqueado") for i in ids}
    texto = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
    try:
        dados = json.loads(texto)
        mapa = {str(d["id"]): str(d["traducao"]).strip() for d in dados}
    except (ValueError, KeyError, TypeError):
        return {i: ("", "erro_json") for i in ids}
    return {i: ((mapa[i], "ok" if mapa[i] else "vazio") if i in mapa else ("", "erro_json")) for i in ids}


def traduzir_lote(pool: Pool, modelo: str, itens: list[dict]) -> dict[str, tuple[str, str]]:
    res = chamar(pool, modelo, itens)
    if len(itens) > 1:  # o que voltou bloqueado ou com erro e refeito item a item, uma vez
        for it in itens:
            if res[it["id"]][1] in ("bloqueado", "erro_json"):
                res[it["id"]] = chamar(pool, modelo, [it])[it["id"]]
    return res


def carregar_feitos() -> dict[str, dict]:
    if not SAIDA.exists():
        return {}
    return {j["id"]: j for j in (json.loads(l) for l in SAIDA.read_text(encoding="utf-8").splitlines() if l)}


def checkpoint(pool: Pool, modelo: str, motivo: str) -> None:
    feitos = carregar_feitos()
    cont = pd.Series([f["status"] for f in feitos.values()]).value_counts().to_dict()
    with CHECKPOINTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "motivo": motivo, "modelo": modelo,
                            "chave_em_uso": pool.slot, "feitos": len(feitos), "status": cont},
                           ensure_ascii=False) + "\n")
    print(f"  checkpoint ({motivo}): {len(feitos)} itens, {cont}", flush=True)


def selecionar(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    if n is None or n >= len(df):
        return df
    partes = [g.sample(max(1, round(len(g) * n / len(df))), random_state=42)
              for _, g in df.groupby(["engine", "label"])]
    return pd.concat(partes).sample(frac=1, random_state=42).head(n)


def cmd_traduzir(args) -> None:
    df = pd.read_parquet(PILOTO)[["id", "text", "label", "source_dataset", "engine"]]
    alvo = selecionar(df, args.piloto if not args.completo else None)
    pool = Pool(carregar_chaves(), args.rotacionar)
    modelo = args.modelo or escolher_modelo(listar_modelos(pool))
    if modelo in PROIBIDOS:
        raise SystemExit(f"{modelo} sera desligado; escolha outro modelo")
    print(f"modelo: {modelo}; chaves carregadas: {len(pool.chaves)}; rotacao: {pool.rotacionar}", flush=True)
    feitos = carregar_feitos()
    pend = alvo[~alvo["id"].isin(feitos)]
    print(f"{len(alvo)} ids no escopo, {len(pend)} pendentes", flush=True)
    ultimo_ckpt = len(feitos)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    try:
        for k in range(0, len(pend), LOTE):
            lote = pend.iloc[k:k + LOTE]
            res = traduzir_lote(pool, modelo, [{"id": r.id, "texto": r.text} for r in lote.itertuples()])
            with SAIDA.open("a", encoding="utf-8") as f:
                for r in lote.itertuples():
                    t, s = res[r.id]
                    f.write(json.dumps({"id": r.id, "text_pt": t, "status": s, "modelo": modelo,
                                        "chave": pool.slot, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")},
                                       ensure_ascii=False) + "\n")
            n_feitos = len(carregar_feitos())
            if pool.trocou:
                checkpoint(pool, modelo, "troca de chave")
                pool.trocou, ultimo_ckpt = False, n_feitos
            elif n_feitos - ultimo_ckpt >= 50:
                checkpoint(pool, modelo, "50 traducoes")
                ultimo_ckpt = n_feitos
    except SemCota as e:
        checkpoint(pool, modelo, f"parou: {e}")
        print("Parou por cota. Retome depois com o mesmo comando; nada foi perdido.")
        return
    checkpoint(pool, modelo, "fim")
    resumo(alvo)


def resumo(alvo: pd.DataFrame) -> None:
    feitos = carregar_feitos()
    st = pd.Series([feitos[i]["status"] for i in alvo["id"] if i in feitos]).value_counts()
    print("status dos itens do escopo:", st.to_dict())


# ----------------------------------------------------------------------------- comparacao

def _marian(textos: list[str]) -> list[str]:
    from transformers import MarianMTModel, MarianTokenizer
    nome = "Helsinki-NLP/opus-mt-tc-big-en-pt"
    tok, mod = MarianTokenizer.from_pretrained(nome), MarianMTModel.from_pretrained(nome)
    enc = tok(textos, return_tensors="pt", padding=True, truncation=True, max_length=256)
    return tok.batch_decode(mod.generate(**enc, max_length=256), skip_special_tokens=True)


def cmd_comparar(args) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    df = pd.read_parquet(PILOTO)
    feitos = carregar_feitos()
    ok = df[df["id"].map(lambda i: feitos.get(i, {}).get("status") == "ok")]
    amostra = pd.concat([ok[ok.label == 1].head(args.n // 2), ok[ok.label == 0].head(args.n - args.n // 2)])
    # cada id foi traduzido por UM motor no piloto; o outro e calculado agora so para estes itens
    falta_marian = amostra[amostra.engine != "marianmt"]
    marian = dict(zip(falta_marian.id, _marian(falta_marian.text.tolist()))) if len(falta_marian) else {}
    qwen = {}
    try:
        from translate_pilot import translate_qwen_one
        for r in amostra[amostra.engine != "qwen2.5-7b-instruct"].itertuples():
            qwen[r.id] = translate_qwen_one(r.text)
    except Exception as e:  # Ollama fora do ar
        print("Qwen indisponivel agora:", type(e).__name__)
    for r in amostra.itertuples():
        m = r.text_pt if r.engine == "marianmt" else marian.get(r.id, "(indisponivel)")
        q = r.text_pt if r.engine != "marianmt" else qwen.get(r.id, "(indisponivel)")
        print(f"\n[{r.id}] {'ódio' if r.label else 'não ódio'}  (piloto: {r.engine})")
        for nome, txt in (("EN    ", r.text), ("Marian", m), ("Qwen  ", q), ("Gemini", feitos[r.id]["text_pt"])):
            print(f"  {nome}: {' '.join(str(txt).split())[:260]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("listar")
    t = sub.add_parser("traduzir")
    t.add_argument("--piloto", type=int, default=50)
    t.add_argument("--completo", action="store_true")
    t.add_argument("--modelo")
    t.add_argument("--rotacionar", action="store_true", help="passa para a proxima chave quando a cota acaba")
    c = sub.add_parser("comparar")
    c.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    if a.cmd == "listar":
        pool = Pool(carregar_chaves(), False)
        nomes = listar_modelos(pool)
        flash = [n for n in nomes if "flash" in n]
        print("modelos com generateContent:", len(nomes))
        print("Flash:", ", ".join(flash))
        print("escolhido por padrao:", escolher_modelo(nomes))
    elif a.cmd == "traduzir":
        cmd_traduzir(a)
    else:
        cmd_comparar(a)
