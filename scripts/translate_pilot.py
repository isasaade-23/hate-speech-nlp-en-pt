"""Etapa 2.4 (piloto): traduz 500 textos de Vidgen/HateXplain/MultiOFF (as 3
fontes comerciais) de EN para PT, com dois motores diferentes, 250 cada:

  - marianmt: Helsinki-NLP/opus-mt-tc-big-en-pt, CPU local (CC-BY-4.0)
    (opus-mt-en-pt original foi descontinuado pela Helsinki-NLP; este é o
    sucessor, mesma licença)
  - qwen2.5-7b-instruct: via Ollama local, http://127.0.0.1:11434 (Apache-2.0)

A amostra (ver data/processed/translation_pilot_500.parquet) preserva a
proporcao real de odio e a proporcao real entre as 3 fontes no treino
comercial -- NAO e 50/50.

Saida NAO vai para o treino. Vai para
data/processed/translation_pilot_500_out.parquet e depois para o doccano
(fila de validacao humana, Etapa 3).

Checkpointa a cada item (Qwen e lento em CPU, roda em varias janelas de
background): rodar de novo pula o que ja foi traduzido.

Uso:
  python scripts/translate_pilot.py --engine marianmt
  python scripts/translate_pilot.py --engine qwen --limit 40
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
QWEN_MODEL = "qwen2.5:7b-instruct"
OUT_PATH = ROOT / "data/processed/translation_pilot_500_out.parquet"

REFUSAL_PATTERNS = [
    r"\bI (cannot|can't|won't)\b", r"\bI'm (sorry|unable)\b", r"as an ai\b",
    r"\bnão posso\b", r"\bnão consigo\b", r"\bpolítica de conteúdo\b",
    r"\bconteúdo ofensivo\b.{0,20}\bremov", r"\[conteúdo removido\]",
    r"\bI apologize\b", r"\bdesculpe, mas\b",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def _log(msg: str) -> None:
    print(msg, flush=True)


def flag_row(original: str, translated: str) -> dict:
    refusal = bool(_REFUSAL_RE.search(translated))
    len_ratio = len(translated) / max(len(original), 1)
    softened = (not refusal) and len_ratio < 0.4
    return {"flag_refusal": refusal, "flag_short": softened, "len_ratio": round(len_ratio, 2)}


def load_state() -> pd.DataFrame:
    sample = pd.read_parquet(ROOT / "data/processed/translation_pilot_500.parquet")
    if OUT_PATH.exists():
        done = pd.read_parquet(OUT_PATH)
        sample = sample.merge(
            done[["id", "text_pt", "flag_refusal", "flag_short", "len_ratio"]],
            on="id", how="left",
        )
    else:
        for c in ("text_pt", "flag_refusal", "flag_short", "len_ratio"):
            sample[c] = None
    return sample


def save_state(df: pd.DataFrame) -> None:
    df.to_parquet(OUT_PATH, index=False)


def run_marianmt(df: pd.DataFrame) -> pd.DataFrame:
    from transformers import MarianMTModel, MarianTokenizer

    mask = (df["engine"] == "marianmt") & df["text_pt"].isna()
    todo = df[mask]
    _log(f"marianmt: {len(todo)} pendentes")
    if not len(todo):
        return df

    name = "Helsinki-NLP/opus-mt-tc-big-en-pt"
    _log(f"carregando {name} (CPU)...")
    tok = MarianTokenizer.from_pretrained(name)
    model = MarianMTModel.from_pretrained(name)

    batch_size = 8
    idx_list = todo.index.tolist()
    for i in range(0, len(idx_list), batch_size):
        chunk_idx = idx_list[i : i + batch_size]
        texts = df.loc[chunk_idx, "text"].tolist()
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        gen = model.generate(**enc, max_length=256)
        out_texts = tok.batch_decode(gen, skip_special_tokens=True)
        for j, oi in enumerate(chunk_idx):
            flags = flag_row(df.loc[oi, "text"], out_texts[j])
            df.loc[oi, "text_pt"] = out_texts[j]
            for k, v in flags.items():
                df.loc[oi, k] = v
        save_state(df)
        _log(f"  marianmt: {min(i + batch_size, len(idx_list))}/{len(idx_list)} (checkpoint salvo)")
    return df


def translate_qwen_one(text: str) -> str:
    prompt = (
        "Você é um tradutor para um projeto de pesquisa acadêmica sobre detecção de "
        "discurso de ódio online. Traduza o texto abaixo do inglês para o português do "
        "Brasil. Preserve EXATAMENTE o tom, a agressividade ou a ofensividade do "
        "original -- isto é um dado de pesquisa, não suavize, não censure, não troque "
        "palavras ofensivas por eufemismos, não adicione avisos. Responda SOMENTE com a "
        f"tradução, sem aspas, sem comentário.\n\nTexto: {text}"
    )
    resp = requests.post(
        OLLAMA_URL,
        json={"model": QWEN_MODEL, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.1}},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def run_qwen(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    mask = (df["engine"] == "qwen2.5-7b-instruct") & df["text_pt"].isna()
    todo_idx = df[mask].index.tolist()
    if limit:
        todo_idx = todo_idx[:limit]
    _log(f"qwen: {len(todo_idx)} pendentes nesta rodada")

    t0 = time.time()
    for n, oi in enumerate(todo_idx, 1):
        text = df.loc[oi, "text"]
        for attempt in range(3):
            try:
                t = translate_qwen_one(text)
                break
            except requests.RequestException as e:
                _log(f"  erro id={df.loc[oi, 'id']} tentativa {attempt + 1}: {e}")
                time.sleep(2)
        else:
            t = ""
        flags = flag_row(text, t)
        df.loc[oi, "text_pt"] = t
        for k, v in flags.items():
            df.loc[oi, k] = v
        save_state(df)
        elapsed = time.time() - t0
        _log(f"  qwen: {n}/{len(todo_idx)} (checkpoint salvo, {elapsed/n:.1f}s/item)")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["marianmt", "qwen", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None, help="max items this run (qwen only)")
    args = ap.parse_args()

    df = load_state()

    if args.engine in ("marianmt", "both"):
        df = run_marianmt(df)
    if args.engine in ("qwen", "both"):
        df = run_qwen(df, args.limit)

    done = df["text_pt"].notna().sum()
    _log(f"\n===== progresso: {done}/{len(df)} traduzidos =====")
    if done:
        summary = df[df["text_pt"].notna()].groupby("engine")[["flag_refusal", "flag_short"]].sum()
        summary["n"] = df[df["text_pt"].notna()].groupby("engine").size()
        _log(str(summary))
    _log(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
