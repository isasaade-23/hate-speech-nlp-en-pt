"""Rotulagem de discurso de ódio por LLM (Gemini, Groq ou Mistral), com a mesma interface.

Usos previstos: juiz do rótulo na auditoria PT (Fase 0, item 4), rotulador de bancos sem rótulo
e professor (probabilidade suave). Cada uso grava num arquivo próprio (--tag).

Chaves: nunca no código, no log nem no Git. Nenhuma chave é impressa.
  gemini   GEMINI_API_KEY ou GEMINI_KEYS_FILE (padrão AIza..., ver translate_gemini.py)
  groq     GROQ_API_KEY, ou GROQ_KEYS_FILE: a chave é a primeira linha não vazia abaixo da
           primeira linha que contém "groq" (sem diferenciar maiúscula); sem essa linha, a
           primeira chave no formato gsk_ do arquivo
  mistral  idem, com MISTRAL_API_KEY / MISTRAL_KEYS_FILE e a linha "mistral"

Saída: data/processed/llm_label_<tag>_<provedor>.jsonl, uma linha por id:
  {id, rotulo, prob, categoria, justificativa_curta, status, provedor, modelo, prompt}
status: ok | bloqueado | erro_json. A retomada pula os ids já gravados com status ok; os demais
são refeitos. O script recusa retomar com outro modelo no mesmo arquivo.

Uso:
  python scripts/llm_label.py modelos --provedor mistral
  python scripts/llm_label.py rotular --provedor gemini --ids data/processed/auditoria_pt_amostra.csv --n 20
  python scripts/llm_label.py rotular --provedor mistral --ids ... --tag juiz_pt
  python scripts/llm_label.py custo --ids ...      # tokens estimados, sem chamada
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import translate_gemini as tg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/processed/corpus_strict.parquet"
# provedores com API no formato da OpenAI: (base, segundos entre pedidos)
OPENAI_COMPAT = {"mistral": ("https://api.mistral.ai/v1", 1.0),
                 "groq": ("https://api.groq.com/openai/v1", 2.5)}
PROVEDORES = ["gemini", *OPENAI_COMPAT]
MODELO_PADRAO = {"gemini": ("gemini-3-flash", "gemini-3-flash-preview"),
                 "mistral": ("mistral-large-latest", "mistral-medium-3.5"),  # chave gratuita: sem large
                 "groq": ("openai/gpt-oss-120b",)}
LOTE = 10
MAX_REFAZER = 3  # acima disso, os itens com falha ficam para a próxima execução (poupa cota)
MAX_TENTATIVAS = 8
PROMPT_VERSAO = "v2.1"  # v2 (20/10): filiação política protegida; v2.1 (21/10): político e veículo não são grupo
CATEGORIAS = ["nenhuma", "raca_etnia", "religiao", "genero", "orientacao_sexual",
              "identidade_de_genero", "origem_regional_ou_nacional", "deficiencia", "idade",
              "classe_social", "filiacao_politica", "outro_grupo"]

SISTEMA = (
    "Você classifica comentários de redes sociais em português para um projeto acadêmico de "
    "pesquisa em detecção de discurso de ódio. Para cada texto, decida se é discurso de ódio pela "
    "definição abaixo.\n\n"
    "Discurso de ódio: ataque, desumanização, ameaça ou incitação à violência ou à discriminação "
    "contra uma pessoa ou um grupo POR CAUSA de uma característica de grupo protegido: raça, cor, "
    "etnia, religião, gênero, identidade de gênero, orientação sexual, origem regional ou nacional, "
    "deficiência, idade, classe social ou filiação ou ideologia política. Qualquer ataque a "
    "eleitores, militantes ou seguidores de uma corrente política (petistas, bolsonaristas, "
    "esquerdistas, comunistas, \"gado\" etc.), ou a uma pessoa por causa da filiação política dela, "
    "é discurso de ódio.\n\n"
    "NÃO é discurso de ódio:\n"
    "- palavrão, xingamento ou ofensa sem alvo em grupo protegido;\n"
    "- crítica ou xingamento contra partido, governo, instituição ou político individual pelo que "
    "ele faz, sem ataque a quem vota ou milita. Xingar um político pelo nome ou apelido "
    "(\"Luladrão\", \"Bozo\") ou um veículo de imprensa não é ataque a grupo político;\n"
    "- texto que cita, denuncia ou discute o ódio sem endossá-lo.\n\n"
    "Marcas <user> e <url> substituem menções e links.\n\n"
    "Exemplos (sintéticos):\n"
    "- \"esse deputado é um ladrão safado\" -> rotulo 0, categoria nenhuma\n"
    "- \"o PT virou uma quadrilha\" -> rotulo 0, categoria nenhuma\n"
    "- \"petista é tudo vagabundo\" -> rotulo 1, categoria filiacao_politica\n"
    "- \"Luladrão, canalha\" -> rotulo 0, categoria nenhuma\n"
    "- \"a Globo é lixo\" -> rotulo 0, categoria nenhuma\n"
    "- \"esses minions desgraçados\" -> rotulo 1, categoria filiacao_politica\n"
    "- \"a direita maldita inventa tudo\" -> rotulo 1, categoria filiacao_politica\n"
    "- \"que jogo horrível, time de merda\" -> rotulo 0, categoria nenhuma\n"
    "- \"nordestino é tudo burro, devia voltar pra lá\" -> rotulo 1, categoria origem_regional_ou_nacional\n"
    "- \"gay tinha que apanhar até virar homem\" -> rotulo 1, categoria orientacao_sexual\n"
    "- \"ele chamou a moça de macaca e foi preso, bem feito\" -> rotulo 0, categoria nenhuma\n\n"
    "Responda só com JSON: {\"itens\": [...]}, um objeto por texto recebido, com o mesmo id e os "
    "campos rotulo (0 ou 1), prob (probabilidade de ser ódio, de 0 a 1), categoria (uma de: "
    + ", ".join(CATEGORIAS) + "; nenhuma quando rotulo 0) e justificativa_curta (até 15 palavras)."
)
_ITEM = {"id": "string", "rotulo": "integer", "prob": "number", "categoria": "string",
         "justificativa_curta": "string"}
SCHEMA_GEMINI = {"type": "OBJECT", "properties": {"itens": {"type": "ARRAY", "items": {
    "type": "OBJECT", "properties": {k: {"type": v.upper()} for k, v in _ITEM.items()},
    "required": list(_ITEM)}}}, "required": ["itens"]}
SCHEMA_JSON = {"type": "object", "additionalProperties": False, "required": ["itens"],
                  "properties": {"itens": {"type": "array", "items": {
                      "type": "object", "additionalProperties": False, "required": list(_ITEM),
                      "properties": {k: {"type": v} for k, v in _ITEM.items()}}}}}


# ----------------------------------------------------------------------------- chaves

def chave_de_arquivo(provedor: str) -> str:
    env = provedor.upper()
    if os.environ.get(f"{env}_API_KEY"):
        return os.environ[f"{env}_API_KEY"].strip()
    arq = os.environ.get(f"{env}_KEYS_FILE")
    if not arq:
        raise SystemExit(f"nenhuma chave: defina {env}_API_KEY ou {env}_KEYS_FILE (ver o cabeçalho)")
    linhas = Path(arq).read_text(encoding="utf-8", errors="ignore").splitlines()
    for i, l in enumerate(linhas):
        if provedor in l.lower():
            for prox in linhas[i + 1:]:
                k = prox.strip()
                if k:
                    if " " in k or len(k) < 20:
                        raise SystemExit(f"a linha abaixo de '{provedor}' não parece uma chave")
                    return k
    if provedor == "groq":  # sem a linha "groq": a primeira chave no formato gsk_
        achadas = re.findall(r"gsk_[A-Za-z0-9]{20,}", "\n".join(linhas))
        if achadas:
            return achadas[0]
    raise SystemExit(f"nenhuma linha com '{provedor}' no arquivo de chaves")


class OpenAICompat:
    """Cliente REST mínimo (Mistral, Groq), com ritmo e backoff; a chave só vai no cabeçalho."""

    def __init__(self, provedor: str, chave: str):
        (self.base, self.intervalo), self._chave, self.ultimo = OPENAI_COMPAT[provedor], chave, 0.0

    def pedir(self, metodo: str, caminho: str, corpo: dict | None = None) -> dict:
        for tentativa in range(MAX_TENTATIVAS):
            falta = self.intervalo - (time.monotonic() - self.ultimo)
            if falta > 0:
                time.sleep(falta)
            self.ultimo = time.monotonic()
            try:
                r = requests.request(metodo, f"{self.base}/{caminho}", json=corpo, timeout=120,
                                     headers={"Authorization": f"Bearer {self._chave}",
                                              "Content-Type": "application/json"})
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                r = None
            if r is not None and r.status_code < 400:
                return r.json()
            if r is not None and r.status_code not in tg.TRANSITORIOS:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            time.sleep(min(60, 2 ** tentativa) + random.random())
        raise tg.SemCota("sem resposta útil após várias tentativas")


def cliente(provedor: str, rotacionar: bool = False):
    if provedor == "gemini":  # rotação: troca de chave (projeto) quando a cota do dia acaba
        return tg.Pool(tg.carregar_chaves(), rotacionar=rotacionar,
                       estado=ROOT / "data/processed/llm_label_gemini_estado.json" if rotacionar else None)
    return OpenAICompat(provedor, chave_de_arquivo(provedor))


def listar_modelos(provedor: str, cli) -> list[str]:
    if provedor == "gemini":
        return tg.listar_modelos(cli)
    return sorted(m["id"] for m in cli.pedir("GET", "models").get("data", []))


# ----------------------------------------------------------------------------- chamada

def validar(dados: dict, ids: list[str]) -> dict[str, tuple[dict, str]]:
    """Confere o esquema item a item; o que faltar ou vier fora do esperado vira erro_json."""
    vazio = {"rotulo": None, "prob": None, "categoria": None, "justificativa_curta": None}
    res = {i: (dict(vazio), "erro_json") for i in ids}
    try:
        itens = dados["itens"]
    except (KeyError, TypeError):
        return res
    for d in itens if isinstance(itens, list) else []:
        try:
            i, rot, prob = str(d["id"]), int(d["rotulo"]), float(d["prob"])
            cat, jus = str(d["categoria"]), str(d["justificativa_curta"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if i in res and rot in (0, 1) and 0 <= prob <= 1 and cat in CATEGORIAS:
            res[i] = ({"rotulo": rot, "prob": prob, "categoria": cat, "justificativa_curta": jus}, "ok")
    return res


def chamar(provedor: str, cli, modelo: str, itens: list[dict]) -> tuple[dict, str]:
    """itens: [{'id','texto'}]. Devolve (id -> (campos, status), modelo informado pela API)."""
    ids = [i["id"] for i in itens]
    entrada = json.dumps(itens, ensure_ascii=False)
    if provedor == "gemini":
        j = cli.pedir("POST", f"models/{modelo}:generateContent", {
            "systemInstruction": {"parts": [{"text": SISTEMA}]},
            "contents": [{"role": "user", "parts": [{"text": entrada}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                 "responseSchema": SCHEMA_GEMINI, "maxOutputTokens": 16384},
            "safetySettings": tg.SEGURANCA})
        cand = (j.get("candidates") or [{}])[0]
        modelo_api = j.get("modelVersion", modelo)
        if j.get("promptFeedback", {}).get("blockReason") or cand.get("finishReason") in (
                "SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "OTHER"):
            return {i: (validar({}, [i])[i][0], "bloqueado") for i in ids}, modelo_api
        texto = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
    else:
        j = cli.pedir("POST", "chat/completions", {
            "model": modelo, "temperature": 0,
            "messages": [{"role": "system", "content": SISTEMA}, {"role": "user", "content": entrada}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "rotulos", "schema": SCHEMA_JSON, "strict": True}}})
        modelo_api = j.get("model", modelo)
        texto = ((j.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    try:
        dados = json.loads(texto)
    except ValueError:
        dados = {}
    return validar(dados, ids), modelo_api


def rotular_lote(provedor, cli, modelo, itens) -> tuple[dict, str]:
    res, modelo_api = chamar(provedor, cli, modelo, itens)
    falhas = [it for it in itens if res[it["id"]][1] != "ok"]
    if len(itens) > 1 and len(falhas) <= MAX_REFAZER:  # refeito item a item, uma vez
        for it in itens:
            if res[it["id"]][1] != "ok":
                res[it["id"]] = chamar(provedor, cli, modelo, [it])[0][it["id"]]
    return res, modelo_api


# ----------------------------------------------------------------------------- execução

def saida(tag: str, provedor: str) -> Path:
    return ROOT / f"data/processed/llm_label_{tag}_{provedor}.jsonl"


def carregar_feitos(arq: Path) -> dict[str, dict]:
    if not arq.exists():
        return {}
    feitos = {}
    for l in arq.read_text(encoding="utf-8").splitlines():
        if l:
            j = json.loads(l)
            feitos[j["id"]] = j  # a última linha de cada id vale
    return feitos


def carregar_itens(ids_csv: str, n: int | None) -> pd.DataFrame:
    ids = pd.read_csv(ids_csv, usecols=["id"]).id.astype(str)
    if n is not None:
        ids = ids.head(n)
    corpus = pd.read_parquet(CORPUS, columns=["id", "text_clean"]).set_index("id")
    faltam = ids[~ids.isin(corpus.index)]
    if len(faltam):
        raise SystemExit(f"{len(faltam)} ids fora do corpus")
    return pd.DataFrame({"id": ids.values, "texto": corpus.loc[ids, "text_clean"].values})


def escolher_modelo(provedor: str, nomes: list[str]) -> str:
    for m in MODELO_PADRAO[provedor]:
        if m in nomes:
            return m
    raise SystemExit(f"nenhum de {MODELO_PADRAO[provedor]} na lista; use --modelo")


def cmd_rotular(args) -> None:
    itens = carregar_itens(args.ids, args.n)
    arq = saida(args.tag, args.provedor)
    feitos = carregar_feitos(arq)
    modelos_antes = {f["modelo_pedido"] for f in feitos.values()}
    cli = cliente(args.provedor, args.rotacionar)
    modelo = args.modelo or escolher_modelo(args.provedor, listar_modelos(args.provedor, cli))
    if modelos_antes and modelos_antes != {modelo}:
        raise SystemExit(f"o arquivo já tem {sorted(modelos_antes)}; não misturo com {modelo}")
    fila = itens[~itens.id.map(lambda i: feitos.get(i, {}).get("status") == "ok")]
    print(f"provedor {args.provedor}, modelo {modelo}: {len(itens)} ids, {len(fila)} a fazer -> {arq.name}",
          flush=True)
    arq.parent.mkdir(parents=True, exist_ok=True)
    lote_n = args.lote
    for ini in range(0, len(fila), lote_n):
        lote =fila.iloc[ini:ini + lote_n].to_dict("records")
        res, modelo_api = rotular_lote(args.provedor, cli, modelo, lote)
        with arq.open("a", encoding="utf-8") as f:
            for it in lote:
                campos, st = res[it["id"]]
                f.write(json.dumps({"id": it["id"], **campos, "status": st, "provedor": args.provedor,
                                    "modelo_pedido": modelo, "modelo": modelo_api,
                                    "prompt": PROMPT_VERSAO}, ensure_ascii=False) + "\n")
        print(f"  {min(ini + lote_n, len(fila))}/{len(fila)}", flush=True)
    st = pd.Series([carregar_feitos(arq)[i]["status"] for i in itens.id]).value_counts().to_dict()
    print("status dos ids pedidos:", st)


def cmd_modelos(args) -> None:
    for m in listar_modelos(args.provedor, cliente(args.provedor)):
        print(m)


def cmd_custo(args) -> None:
    """Estimativa grosseira: 1 token ~ 4 caracteres; saída ~ 45 tokens por item."""
    itens = carregar_itens(args.ids, args.n)
    lotes = -(-len(itens) // LOTE)
    entrada = lotes * len(SISTEMA) / 4 + len(json.dumps(itens.to_dict("records"), ensure_ascii=False)) / 4
    print(f"{len(itens)} itens em {lotes} lotes: ~{entrada:,.0f} tokens de entrada e "
          f"~{45 * len(itens):,} de saída por provedor")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("modelos")
    m.add_argument("--provedor", choices=PROVEDORES, required=True)
    for nome in ("rotular", "custo"):
        p = sub.add_parser(nome)
        p.add_argument("--ids", required=True, help="CSV com coluna id")
        p.add_argument("--n", type=int, help="só os n primeiros ids (piloto)")
        if nome == "rotular":
            p.add_argument("--provedor", choices=PROVEDORES, required=True)
            p.add_argument("--modelo")
            p.add_argument("--tag", default="juiz_pt")
            p.add_argument("--lote", type=int, default=LOTE, help="textos por pedido")
            p.add_argument("--rotacionar", action="store_true",
                           help="gemini: passa para a próxima chave quando a cota do dia acaba")
    args = ap.parse_args(argv)
    {"rotular": cmd_rotular, "modelos": cmd_modelos, "custo": cmd_custo}[args.cmd](args)


if __name__ == "__main__":
    main()
