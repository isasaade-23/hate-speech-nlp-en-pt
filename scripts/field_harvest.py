"""Colheita de blocos de texto de páginas reais pros testes de campo.

Replica as regras do content.js da extensão (seletor p/li/blockquote, só
folhas, 8-2000 chars, corte em 1000) sobre o HTML servido. A checagem de
visibilidade (getBoundingClientRect) não existe aqui; a diferença fica
registrada no protocolo.

Uso:
  python scripts/field_harvest.py urls.txt reports/field_tests/pages/
Cada linha de urls.txt: <page_id>\t<url>
Saída: um JSONL por página: {"page_id", "url", "fetched_at", "blocks": [...]}
"""

import json
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

SELECTOR = "p, li, blockquote"
SKIP_ANCESTORS = {"nav", "header", "footer", "script", "style"}
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/139.0 Safari/537.36")


def harvest(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for el in soup.select(SELECTOR):
        if any(p.name in SKIP_ANCESTORS for p in el.parents if p.name):
            continue
        if el.select(SELECTOR):  # só folhas
            continue
        text = el.get_text(" ", strip=True)
        if len(text) < 8 or len(text) > 2000:
            continue
        out.append(text[:1000])
    # dedup preservando ordem (menus repetidos etc.)
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def main() -> None:
    urls_file, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    for line in urls_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        page_id, url = line.split("\t", 1)
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
            r.raise_for_status()
            blocks = harvest(r.text)
            status = "ok"
        except Exception as e:  # noqa: BLE001 - log e segue
            blocks, status = [], f"error: {e}"
        rec = {
            "page_id": page_id,
            "url": url,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": status,
            "n_blocks": len(blocks),
            "blocks": blocks,
        }
        path = out_dir / f"{page_id}.jsonl"
        path.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{page_id}: {status} ({len(blocks)} blocos)")


if __name__ == "__main__":
    main()
