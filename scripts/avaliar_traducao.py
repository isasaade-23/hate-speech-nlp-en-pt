"""Qualidade das traducoes do piloto: MarianMT, Qwen e Gemini nos mesmos ids, sem chamar API.

Cada id do piloto foi traduzido por UM motor da v1 (248 MarianMT, 252 Qwen); o Gemini traduz os 500.
A comparacao e pareada: Gemini x MarianMT nos ids do Marian, Gemini x Qwen nos ids do Qwen.

Defeitos (cada um vira uma coluna; "quebrada" = pelo menos um):
  html        tags, entidades HTML, cerca de markdown ou JSON que nao estavam no original
  codificacao caractere de substituicao, mojibake (Ã©, â€™), caractere de controle
  alfabeto    letra de alfabeto nao latino (arabe, cirilico, CJK...) que nao estava no original
  ingles      palavra comum do ingles, sem uso em portugues, que sobrou na traducao
  vazio       traducao vazia (inclui o que o Gemini bloqueou)
  igual       igual ao original (nao traduziu)
  tamanho     razao de comprimento PT/EN fora de 0,5 a 2 (original com 15+ caracteres)
Contado a parte, NAO e defeito:
  mascara     palavrao ou insulto com letra trocada por numero ou simbolo (vi4d0, p*rra, c@r@lho)

Saidas: reports/avaliacao_traducao.md (so ids, sem texto) e
data/processed/avaliacao_traducao_itens.csv (um item por linha e motor; fora do Git).
Uso: python scripts/avaliar_traducao.py
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PILOTO = ROOT / "data/processed/translation_pilot_500_out.parquet"
GEMINI = ROOT / "data/processed/translation_gemini.jsonl"
SAIDA_MD = ROOT / "reports/avaliacao_traducao.md"
SAIDA_CSV = ROOT / "data/processed/avaliacao_traducao_itens.csv"

DEFEITOS = ["html", "codificacao", "alfabeto", "ingles", "vazio", "igual", "tamanho"]

# Palavras comuns do ingles que nao existem em portugues. Ficam de fora as que colidem com o
# portugues (a, e, o, do, no, me, se, time, come, real, more, so) e emprestimos de uso corrente.
INGLES = set("""
the and of to is are was were this that these those with for from they them their there what which
who would should could have has had been being not but about just like because people really totally
very you your yours he she his him her its we our my it an if when where why how all some any most
other only also than then can will does did don't can't isn't i'm get got go going know think want
make say said see look still even never always every much many good bad fucking fuck shit weird
weirdos hate love women woman girl guy guys kids country world thing things right left back out up
down over into off again here now yes yeah white black ass damn bitch whore slut cunt dick pussy
asshole moron idiot retard retarded nigger nigga niggers faggot fag kike spic chink tranny dyke
raghead hoe thot shawty
""".split())

# Para reconhecer mascara: palavroes e insultos em portugues, sem acento.
PALAVROES = set("""
viado viadinho porra caralho merda puta puto foda fodase foder fodido buceta cu cuzao arrombado
arrombada piranha vagabunda vagabundo vadia macaco macaca bicha sapatao traveco retardado mongol
idiota otario otaria desgracado desgracada cacete bosta crioulo vaca corno babaca escroto escrota
""".split())
LEET = str.maketrans({"4": "a", "@": "a", "0": "o", "1": "i", "!": "i", "3": "e", "5": "s", "$": "s", "7": "t"})

RE_HTML = re.compile(r"<\s*/?\s*[a-zA-Z][^>]*>|&(?:amp|lt|gt|quot|apos|nbsp|#\d+|#x[0-9a-fA-F]+);|```"
                     r"|^\s*[\[{]\s*\"")
RE_MOJIBAKE = re.compile(r"Ã[\u0080-¿]|Ã[¡-ÿ]|â€|Â[ -¿]")
RE_PALAVRA = re.compile(r"[a-zà-ÿ']+", re.IGNORECASE)
RE_TOKEN = re.compile(r"[\w*@$!]+")


def sem_acento(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def tem_html(t: str, original: str = "") -> bool:
    """So marcacao nova: o HateXplain ja traz <user>, <happy> etc. no original."""
    ja = {m.lower() for m in RE_HTML.findall(original)}
    return any(m.lower() not in ja for m in RE_HTML.findall(t))


def codificacao_quebrada(t: str) -> bool:
    return ("�" in t or bool(RE_MOJIBAKE.search(t))
            or any(unicodedata.category(c) == "Cc" and c not in "\n\r\t" for c in t))


def letras_nao_latinas(t: str, original: str = "") -> int:
    ja = set(original)
    return sum(1 for c in t if unicodedata.category(c).startswith("L") and c not in ja
               and not unicodedata.name(c, "").startswith("LATIN"))


def ingles_que_sobrou(t: str, original: str = "") -> list[str]:
    proprios = {w.lower().split("'")[0] for w in RE_PALAVRA.findall(original) if w[:1].isupper()}
    return [w for w in (x.lower() for x in RE_PALAVRA.findall(t)) if w in INGLES and w not in proprios]


def mascaras(t: str) -> list[str]:
    achadas = []
    for tok in RE_TOKEN.findall(t):
        if not (re.search(r"[a-zA-Z]", tok) and re.search(r"[0-9*@$!]", tok)):
            continue
        base = sem_acento(tok.lower().rstrip("!")).translate(LEET)
        if not re.fullmatch(r"[a-z*]+", base):
            continue
        padrao = re.compile(base.replace("*", "[a-z]{1,2}"))
        if any(padrao.fullmatch(p) for p in PALAVROES):
            achadas.append(tok)
    return achadas


def avaliar(t: str, original: str) -> dict:
    t, original = str(t or ""), str(original or "")
    r = {
        "html": tem_html(t, original),
        "codificacao": codificacao_quebrada(t),
        "alfabeto": letras_nao_latinas(t, original) > 0,
        "ingles": bool(ingles_que_sobrou(t, original)),
        "vazio": not t.strip(),
        "igual": bool(t.strip()) and t.strip().lower() == original.strip().lower(),
        "tamanho": len(original) >= 15 and bool(t.strip()) and not 0.5 <= len(t) / len(original) <= 2,
        "mascara": len(mascaras(t)),
        "ingles_palavras": " ".join(ingles_que_sobrou(t, original)),
    }
    r["quebrada"] = any(r[d] for d in DEFEITOS)
    return r


def montar() -> tuple[pd.DataFrame, int]:
    piloto = pd.read_parquet(PILOTO)
    gem = {}
    if GEMINI.exists():
        for l in GEMINI.read_text(encoding="utf-8").splitlines():
            if l:
                j = json.loads(l)
                gem[j["id"]] = j
    linhas = []
    for r in piloto.itertuples():
        if r.id not in gem:
            continue
        motor_v1 = "MarianMT" if r.engine == "marianmt" else "Qwen"
        for motor, texto in ((motor_v1, r.text_pt), ("Gemini", gem[r.id]["text_pt"])):
            linhas.append({"id": r.id, "par": motor_v1, "motor": motor, "label": r.label,
                           "status_gemini": gem[r.id]["status"], **avaliar(texto, r.text)})
    return pd.DataFrame(linhas), len(piloto) - len(gem)


def relatorio(df: pd.DataFrame, faltam: int) -> str:
    cols = DEFEITOS + ["quebrada"]
    L = [f"# Qualidade das traduções do piloto, {date.today().isoformat()}", "",
         "Comparação pareada nos mesmos ids, sem texto, só ids. Defeitos e máscaras definidos em "
         "`scripts/avaliar_traducao.py`. Máscara (palavrão com letra trocada) é contada à parte e não é defeito.", ""]
    if faltam:
        L += [f"**Gemini incompleto:** {faltam} ids ainda sem tradução, fora desta conta.", ""]
    cab = "| par | motor | n | " + " | ".join(cols) + " | máscaras |"
    L += [cab, "|" + "---|" * (len(cols) + 4)]
    blocos = [(p, m, g) for (p, m), g in df.groupby(["par", "motor"], sort=True)]
    blocos.append(("todos", "Gemini", df[df.motor == "Gemini"]))
    for par, motor, g in blocos:
        pct = " | ".join(f"{100 * g[c].mean():.1f}%" for c in cols)
        L.append(f"| {par} | {motor} | {len(g)} | {pct} | {int(g['mascara'].sum())} em {int((g['mascara'] > 0).sum())} itens |")
    L += ["", "Status do Gemini nos itens avaliados: " +
          ", ".join(f"{k} {v}" for k, v in df[df.motor == "Gemini"]["status_gemini"].value_counts().items()), ""]
    L += ["## Exemplos por defeito (até 5 ids)", ""]
    for motor in ("MarianMT", "Qwen", "Gemini"):
        g = df[df.motor == motor]
        partes = [f"{d}: {', '.join(g[g[d]].id.head(5))}" for d in DEFEITOS + ["mascara"]
                  if (g[d] > 0).any()]
        L.append(f"- **{motor}**: " + ("; ".join(partes) if partes else "nenhum"))
    g = df[df.ingles_palavras != ""]
    if len(g):
        L += ["", "## Palavras em inglês que mais sobraram", ""]
        for motor, gm in g.groupby("motor"):
            cont = pd.Series(" ".join(gm.ingles_palavras).split()).value_counts().head(10)
            L.append(f"- **{motor}**: " + ", ".join(f"{w} ({n})" for w, n in cont.items()))
    return "\n".join(L) + "\n"


def main() -> None:
    df, faltam = montar()
    if df.empty:
        raise SystemExit("nenhuma traducao do Gemini ainda; rode scripts/translate_gemini.py antes")
    SAIDA_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAIDA_CSV, index=False, encoding="utf-8")
    SAIDA_MD.write_text(relatorio(df, faltam), encoding="utf-8")
    print(f"{df.id.nunique()} ids avaliados, {faltam} sem Gemini; relatorio: {SAIDA_MD}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
