"""Gate de release: viés de identidade dos candidatos a modelo servido (PT).

Regra adotada em 24/08/2026 (ver methodology/pt_recall_x_vies.md): nenhum modelo
entra em produção só por métrica agregada. O modelo clássico dedicado ao PT
dobrava o recall e foi barrado por marcar pessoas LGBT, negras e muçulmanas
falando de si mesmas.

Aqui a sonda roda sobre o TESTE REAL, que é mais forte que frases inventadas:
entre as linhas em português que NÃO são ódio e que mencionam um termo neutro de
identidade, quantas o modelo sinaliza? Compara com a taxa de falso positivo de
fundo do mesmo modelo (o excesso é o que a menção do grupo causa).

Uso: python scripts/bias_gate_v5.py
Saída: reports/tables/bias_gate_v5_pt.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsc.bias_probe import IDENTITY_TERMS, _compile, _mentions  # noqa: E402

CANDIDATOS = [
    "stack_strict_s42",           # o que está no ar
    "pt_logreg_strict_s42",       # o clássico PT barrado (referência do dano)
    "bertimbau_pt_strict_s42",    # melhor PT do painel neural
    "twitter_xlmr_strict_s42",
    "xlmr_multilingual_strict_s42",
    "lfm25_encoder_strict_s42",
]


def main() -> None:
    corpus = pd.read_parquet(ROOT / "data/processed/corpus_strict.parquet").set_index("id")
    pats = {g: _compile(t) for g, t in IDENTITY_TERMS.items()}

    rows = []
    for mid in CANDIDATOS:
        f = ROOT / f"reports/predictions/{mid}_test.parquet"
        if not f.exists():
            print(f"(sem predições: {mid})")
            continue
        df = pd.read_parquet(f).set_index("id").join(corpus[["text_clean"]], how="left")
        pt = df[df.language == "pt"]
        nao_odio = pt[pt.y_true == 0]
        if not len(nao_odio):
            continue
        fundo = float((nao_odio.y_pred == 1).mean())

        # linhas PT não-ódio que mencionam QUALQUER termo neutro de identidade
        mask = nao_odio.apply(
            lambda r: any(_mentions(r["text_clean"], "pt", p) for p in pats.values()), axis=1)
        sub = nao_odio[mask]
        odio_pt = pt[pt.y_true == 1]
        rows.append({
            "modelo": mid.replace("_strict_s42", ""),
            "recall_odio_pt": round(float((odio_pt.y_pred == 1).mean()), 4),
            "fp_fundo_pt": round(fundo, 4),
            "fp_com_identidade": round(float((sub.y_pred == 1).mean()), 4),
            "excesso": round(float((sub.y_pred == 1).mean()) - fundo, 4),
            "n_linhas": int(len(sub)),
        })

    out = pd.DataFrame(rows).sort_values("recall_odio_pt", ascending=False)
    dest = ROOT / "reports/tables/bias_gate_v5_pt.csv"
    out.to_csv(dest, index=False)
    print("=== fatia PT do teste v5: recall de ódio x super-marcação de identidade ===")
    print("fp_com_identidade: entre textos PT que NÃO são ódio e mencionam um grupo,")
    print("quantos o modelo sinaliza. 'excesso' é o quanto passa da taxa de fundo dele.\n")
    print(out.to_string(index=False))
    print("\n->", dest)


if __name__ == "__main__":
    main()
