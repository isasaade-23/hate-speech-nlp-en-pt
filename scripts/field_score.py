"""Escore duplo dos blocos colhidos em campo (stack v5 servido + linear v5 da extensão).

Uso (na raiz do repo, .venv ativo):
  python scripts/field_score.py

Lê reports/field_tests/pages/*.jsonl, escora cada bloco com:
  - stack_strict_s42 (o modelo do site), limiar do bundle
  - luciola_linear_v5 (o modelo da extensão, via node), limiar 0.424
Saídas:
  - reports/field_tests/blocks_scored.csv (tudo)
  - reports/field_tests/revisao_isabela.csv (sinalizados por qualquer um dos
    dois + amostra de até 30 não-sinalizados por página, embaralhada com seed)
"""

import csv
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hsc.inference import get_classifier  # noqa: E402

EXT_DIR = Path("C:/Users/Renato/luciola-extension")
LINEAR_THRESHOLD = 0.424
FN_SAMPLE_PER_PAGE = 30
SEED = 42


def score_linear(texts: list[str]) -> list[float]:
    with tempfile.TemporaryDirectory() as td:
        inp, out = Path(td) / "in.json", Path(td) / "out.json"
        inp.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
        subprocess.run(
            ["node", str(EXT_DIR / "test" / "score_file.mjs"), str(inp), str(out)],
            check=True, capture_output=True,
        )
        return [r["prob"] for r in json.loads(out.read_text(encoding="utf-8"))]


def main() -> None:
    pages_dir = Path("reports/field_tests/pages")
    clf = get_classifier("stack_strict_s42")
    print(f"stack threshold: {clf.threshold:.4f} | linear threshold: {LINEAR_THRESHOLD}")

    rows = []
    for f in sorted(pages_dir.glob("*.jsonl")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        blocks = rec["blocks"]
        if not blocks:
            continue
        stack = clf.predict_batch(blocks)
        linear = score_linear(blocks)
        for i, (text, s, lp) in enumerate(zip(blocks, stack, linear)):
            rows.append({
                "page_id": rec["page_id"],
                "block_id": f"{rec['page_id']}_{i:04d}",
                "text": text,
                "stack_prob": round(float(s["score"]), 4),
                "stack_flag": int(s["label"] == "hate"),
                "linear_prob": round(float(lp), 4),
                "linear_flag": int(lp >= LINEAR_THRESHOLD),
            })
        print(f"{rec['page_id']}: {len(blocks)} blocos")

    out_all = Path("reports/field_tests/blocks_scored.csv")
    with out_all.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # planilha de revisão: todos os sinalizados + amostra de não-sinalizados
    rng = random.Random(SEED)
    review = []
    by_page: dict[str, list[dict]] = {}
    for r in rows:
        by_page.setdefault(r["page_id"], []).append(r)
    for page, page_rows in by_page.items():
        flagged = [r for r in page_rows if r["stack_flag"] or r["linear_flag"]]
        unflagged = [r for r in page_rows if not (r["stack_flag"] or r["linear_flag"])]
        sample = rng.sample(unflagged, min(FN_SAMPLE_PER_PAGE, len(unflagged)))
        for r in flagged:
            review.append({**r, "grupo": "sinalizado"})
        for r in sample:
            review.append({**r, "grupo": "amostra_nao_sinalizado"})

    out_rev = Path("reports/field_tests/revisao_isabela.csv")
    fields = list(review[0].keys()) + ["veredito_isabela", "tipo_erro", "obs"]
    with out_rev.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in review:
            w.writerow({**r, "veredito_isabela": "", "tipo_erro": "", "obs": ""})

    n_flag = sum(1 for r in review if r["grupo"] == "sinalizado")
    print(f"\nTotal blocos: {len(rows)} | sinalizados (qualquer modelo): {n_flag} | "
          f"linhas na revisão: {len(review)}")
    print(f"-> {out_all}\n-> {out_rev}")
    print("\nGuia veredito_isabela: TP (é ódio), FP (não é), FN (amostra que É "
          "ódio e passou), OK (amostra corretamente limpa).")
    print("tipo_erro sugerido: identidade_neutra, discurso_reportado, ironia, "
          "implicito_perdido, xingamento_sem_alvo, outro.")


if __name__ == "__main__":
    main()
