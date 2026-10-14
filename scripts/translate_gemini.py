"""Retraducao do piloto (os mesmos 500 ids) com a API do Gemini, chamada local por REST.

Chaves: nunca no codigo, no log nem no Git. Lidas de, nesta ordem:
  GEMINI_API_KEY        uma chave
  GEMINI_KEYS_FILE      caminho de um arquivo FORA do repositorio; as chaves sao extraidas por
                        padrao (AIza...), qualquer formato de linha serve
O script nunca imprime uma chave; so o indice da chave em uso (chave 1, 2, ...).

Rotacao de chaves e OPCIONAL (--rotacionar; ver a classe Pool). A cota do Gemini e por PROJETO,
nao por chave, e os termos de uso das APIs do Google dizem que o usuario nao tentara contornar os
limites; usar varias chaves e decisao de quem roda. Sem --rotacionar, ao esgotar a cota o script
grava o checkpoint e para; retome depois. Com ela, nao para ate terminar (ou Ctrl+C).

Progresso: cada item vai para data/processed/translation_gemini.jsonl assim que o lote volta; a
retomada pula os ids feitos. Checkpoints em translation_gemini_checkpoints.jsonl (a cada 50, a
cada troca de chave e a cada espera longa); chaves sem cota no dia em translation_gemini_estado.json
(so a impressao SHA-256 da chave). Tudo em data/processed/, fora do Git.

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
import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://generativelanguage.googleapis.com/v1beta"
SAIDA = ROOT / "data/processed/translation_gemini.jsonl"
CHECKPOINTS = ROOT / "data/processed/translation_gemini_checkpoints.jsonl"
ESTADO = ROOT / "data/processed/translation_gemini_estado.json"
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


PACIFICO = ZoneInfo("America/Los_Angeles")  # a cota diaria do Gemini reinicia a meia-noite daqui
ESPERA_MAX_VOLTA = 300  # s; teto do backoff quando uma volta inteira pelas chaves falha
TRANSITORIOS = (408, 429, 500, 502, 503, 504)


def proxima_meia_noite_pacifico(agora: datetime | None = None) -> datetime:
    agora = (agora or datetime.now(timezone.utc)).astimezone(PACIFICO)
    return (agora + timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)


def impressao(chave: str) -> str:
    """Identifica a chave no arquivo de estado sem guarda-la: 12 primeiros hex do SHA-256."""
    return hashlib.sha256(chave.encode()).hexdigest()[:12]


class Pool:
    """Chaves em fila circular.

    Com --rotacionar: 503/5xx/408/429 por minuto/timeout pulam para a proxima chave na hora; 429 de
    cota diaria (PerDay) tira a chave da fila ate a meia-noite do Pacifico (estado salvo em disco, so
    com a impressao SHA-256 da chave). Uma volta inteira sem resposta 200 espera com backoff e jitter
    (ate ESPERA_MAX_VOLTA) e recomeca; todas esgotadas no dia, espera a cota reiniciar. Nao desiste.
    Sem --rotacionar: uma chave so, backoff ate MAX_TENTATIVAS, depois SemCota.
    """

    def __init__(self, chaves: list[str], rotacionar: bool, estado: Path | None = None):
        self.chaves, self.rotacionar, self.i = chaves, rotacionar, 0
        self.trocou = False
        self.ultimo = 0.0
        self.estado = estado
        self.esgotadas: dict[str, str] = {}  # impressao -> ISO de quando a cota volta
        self.invalidas: set[int] = set()
        self.ao_esperar = None  # callback(motivo) para checkpoint em espera longa
        if estado and estado.exists():
            self.esgotadas = json.loads(estado.read_text(encoding="utf-8")).get("esgotadas", {})
        self._limpar_vencidas()

    @property
    def slot(self) -> int:
        return self.i + 1

    def _cabecalho(self) -> dict:
        return {"x-goog-api-key": self.chaves[self.i], "Content-Type": "application/json"}

    def _limpar_vencidas(self) -> None:
        agora = datetime.now(timezone.utc)
        self.esgotadas = {h: t for h, t in self.esgotadas.items() if datetime.fromisoformat(t) > agora}

    def _salvar(self) -> None:
        if self.estado:
            self.estado.parent.mkdir(parents=True, exist_ok=True)
            self.estado.write_text(json.dumps({"esgotadas": self.esgotadas}, indent=1), encoding="utf-8")

    def _disponivel(self, i: int) -> bool:
        return i not in self.invalidas and impressao(self.chaves[i]) not in self.esgotadas

    def disponiveis(self) -> int:
        self._limpar_vencidas()
        return sum(self._disponivel(i) for i in range(len(self.chaves)))

    def _avancar(self, motivo: str) -> None:
        """Proxima chave disponivel na fila circular (pode ser a mesma, se for a unica)."""
        n = len(self.chaves)
        for passo in range(1, n + 1):
            j = (self.i + passo) % n
            if self._disponivel(j):
                if j != self.i:
                    print(f"  chave {self.slot}: {motivo}; passando para a chave {j + 1}", flush=True)
                    self.trocou = True
                self.i = j
                return

    def _esgotar(self) -> None:
        self.esgotadas[impressao(self.chaves[self.i])] = proxima_meia_noite_pacifico().isoformat()
        self._salvar()

    def _esperar(self, segundos: float, motivo: str) -> None:
        print(f"  espera de {segundos / 60:.1f} min: {motivo}", flush=True)
        if self.ao_esperar:
            self.ao_esperar(motivo)
        time.sleep(segundos)

    def _ritmo(self) -> None:
        """Um pedido a cada INTERVALO s no maximo (o plano gratuito nao aguenta rajada)."""
        falta = INTERVALO - (time.monotonic() - self.ultimo)
        if falta > 0:
            time.sleep(falta)
        self.ultimo = time.monotonic()

    def _um_pedido(self, metodo: str, caminho: str, corpo: dict | None):
        """(json, None) em sucesso; (None, motivo) em falha transitoria; (None, 'perday') em cota do dia."""
        self._ritmo()
        try:
            r = requests.request(metodo, f"{BASE}/{caminho}", headers=self._cabecalho(), json=corpo, timeout=120)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            return None, type(e).__name__
        if r.status_code == 429 and "PerDay" in r.text:
            return None, "perday"
        if r.status_code in TRANSITORIOS:
            return None, f"HTTP {r.status_code}"
        if r.status_code in (400, 401, 403) and "API key" in r.text:
            return None, "invalida"
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json(), None

    def pedir(self, metodo: str, caminho: str, corpo: dict | None = None) -> dict:
        if not self.rotacionar:
            for tentativa in range(MAX_TENTATIVAS):
                j, motivo = self._um_pedido(metodo, caminho, corpo)
                if j is not None:
                    return j
                if motivo in ("perday", "invalida"):
                    break
                time.sleep(min(60, 2 ** tentativa) + random.random())  # backoff do manual do Google
            raise SemCota("sem resposta util apos varias tentativas")
        voltas = 0
        while True:
            if not self.disponiveis():
                volta = min(datetime.fromisoformat(t) for t in self.esgotadas.values()) if self.esgotadas else None
                if volta is None:
                    raise SemCota("nenhuma chave valida")
                self._esperar(max(60.0, (volta - datetime.now(timezone.utc)).total_seconds()),
                              "todas as chaves sem cota no dia; aguardando a meia-noite do Pacifico")
                self._limpar_vencidas()
                continue
            if not self._disponivel(self.i):
                self._avancar("sem cota no dia")
            falhas = 0
            for _ in range(self.disponiveis()):  # uma volta pelas chaves disponiveis
                j, motivo = self._um_pedido(metodo, caminho, corpo)
                if j is not None:
                    return j
                if motivo == "perday":
                    self._esgotar()
                    motivo = "cota do dia esgotada"
                elif motivo == "invalida":
                    self.invalidas.add(self.i)
                    motivo = "chave recusada pela API"
                falhas += 1
                if not self.disponiveis():
                    break
                self._avancar(motivo)
            if falhas and self.disponiveis():
                voltas += 1
                self._esperar(min(ESPERA_MAX_VOLTA, 2 ** voltas) + random.random() * 5,
                              f"volta {voltas} sem resposta util em nenhuma chave")


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
    pool = Pool(carregar_chaves(), args.rotacionar, ESTADO)
    feitos = carregar_feitos()
    anteriores = {f["modelo"] for f in feitos.values()}
    modelo = args.modelo or (anteriores.pop() if len(anteriores) == 1 else escolher_modelo(listar_modelos(pool)))
    if modelo in PROIBIDOS:
        raise SystemExit(f"{modelo} sera desligado; escolha outro modelo")
    outros = {f["modelo"] for f in feitos.values()} - {modelo}
    if outros and not args.misturar_modelos:  # as 500 nao devem misturar modelos sem decisao explicita
        raise SystemExit(f"a saida ja tem itens de {sorted(outros)}; use o mesmo modelo ou --misturar-modelos")
    pool.ao_esperar = lambda motivo: checkpoint(pool, modelo, f"espera: {motivo}")
    print(f"modelo: {modelo}; chaves carregadas: {len(pool.chaves)} (de {os.environ.get('GEMINI_KEYS_FILE', 'GEMINI_API_KEY')}); "
          f"rotacao: {pool.rotacionar}; sem cota hoje: {len(pool.chaves) - pool.disponiveis()}", flush=True)
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


def cmd_chaves() -> None:
    """Confere de onde vem cada chave sem mostrar nenhuma: slot e 4 ultimos caracteres."""
    arq = os.environ.get("GEMINI_KEYS_FILE")
    print("GEMINI_API_KEY:", "definida" if os.environ.get("GEMINI_API_KEY") else "nao definida")
    print("GEMINI_KEYS_FILE:", arq or "nao definida")
    ocorrencias = len(re.findall(r"AIza[0-9A-Za-z_\-]{30,}", Path(arq).read_text(encoding="utf-8", errors="ignore"))) if arq else 0
    chaves = carregar_chaves()
    print(f"chaves distintas: {len(chaves)}; ocorrencias no arquivo: {ocorrencias}; duplicadas descartadas: {ocorrencias - len(chaves)}")
    for i, k in enumerate(chaves, 1):
        print(f"  slot {i}: ...{k[-4:]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("listar")
    sub.add_parser("chaves")
    t = sub.add_parser("traduzir")
    t.add_argument("--piloto", type=int, default=50)
    t.add_argument("--completo", action="store_true")
    t.add_argument("--modelo")
    t.add_argument("--rotacionar", action="store_true",
                   help="fila circular de chaves: pula em 503/429/timeout, tira do dia a que esgotou a cota")
    t.add_argument("--misturar-modelos", action="store_true", help="aceita continuar com modelo diferente do ja usado")
    c = sub.add_parser("comparar")
    c.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    if a.cmd == "chaves":
        cmd_chaves()
    elif a.cmd == "listar":
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
