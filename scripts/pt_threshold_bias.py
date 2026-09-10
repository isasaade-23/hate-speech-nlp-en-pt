"""O limiar por idioma em PT compra recall pagando em viés de identidade.

`pt_boost.py` mostrou que trocar o limiar global (0,3372) pelo limiar ajustado na
validação PT (0,1092) leva o recall de ódio de 0,316 para 0,556 sem tocar no
modelo. A tabela de `methodology/pt_recall_x_vies.md` mediu falso positivo de
identidade para as configurações que trocam de MODELO, mas nunca para essa, que
só troca o limiar.

Este script fecha o buraco: varre o limiar na fatia PT do teste v5 e, em cada
ponto, mede junto o alerta nas 20 frases neutras da sonda de identidade (as
mesmas de `pt_identity_probe.py`, pessoas falando de si). Nenhuma delas é ódio,
então ali toda marcação é falso positivo puro.

Uso: python scripts/pt_threshold_bias.py
Saída: reports/tables/pt_threshold_bias.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsc.inference import HateClassifier  # noqa: E402

# a sonda é a mesma de pt_identity_probe.py: as frases não podem divergir entre os dois
from pt_identity_probe import FRASES  # noqa: E402

# limiares de interesse: o global servido, o ajustado na validação PT, e a grade
# entre os dois, para achar onde o viés começa a subir
GRADE = [0.3372, 0.3200, 0.3000, 0.2500, 0.2000, 0.1500, 0.1092]


def main() -> None:
    clf = HateClassifier("stack_strict_s42")
    # scores das frases neutras, uma vez só: o que varia daqui pra frente é o corte
    neutras = np.array([clf.predict(f)["score"] for f in FRASES])

    te = pd.read_parquet(ROOT / "reports/predictions/stack_strict_s42_test.parquet")
    pt = te[te.language == "pt"]
    y, s = pt.y_true.values, pt.y_score.values

    rows = []
    for thr in GRADE:
        yp = (s >= thr).astype(int)
        rows.append({
            "limiar": round(thr, 4),
            "recall_hate": round(recall_score(y, yp, zero_division=0), 4),
            "precision_hate": round(precision_score(y, yp, zero_division=0), 4),
            "macro_f1": round(f1_score(y, yp, average="macro", zero_division=0), 4),
            "fp_identidade": int((neutras >= thr).sum()),
            "n_sonda": len(FRASES),
        })

    out = pd.DataFrame(rows)
    dest = ROOT / "reports/tables/pt_threshold_bias.csv"
    out.to_csv(dest, index=False)
    print(f"fatia PT do teste v5: {len(pt)} linhas, {int(y.sum())} de ódio\n")
    print(out.to_string(index=False))

    seguro = out[out.fp_identidade <= 3].limiar.min()
    ganho = out[out.limiar == seguro].recall_hate.iloc[0] - out.recall_hate.iloc[0]
    print(f"\nMenor limiar que mantém o viés na linha de base (3/20): {seguro:.4f}")
    print(f"Recall que ele compra sobre o limiar global: +{ganho:.4f}")
    print("->", dest)


if __name__ == "__main__":
    main()
