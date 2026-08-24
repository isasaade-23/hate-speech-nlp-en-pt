"""Sonda de viés de identidade em PT: o modelo dedicado marca a mais quem fala de si?

Motivo: no teste de roteamento, "sou uma mulher lésbica e tenho orgulho disso"
recebeu 0,86 do modelo PT. Uma ferramenta antiódio que sinaliza pessoas LGBT
falando de si mesmas causa exatamente o dano que deveria evitar.

Compara stack servido x modelo PT em frases NEUTRAS ou POSITIVAS que contêm
termos de identidade. Nenhuma delas é discurso de ódio: a taxa de alerta aqui é
taxa de falso positivo pura.

Uso: python scripts/pt_identity_probe.py
Saída: reports/tables/pt_identity_probe.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsc.inference import HateClassifier  # noqa: E402

# frases neutras/positivas com termo de identidade. Nenhuma é ódio.
FRASES = [
    "sou uma mulher lésbica e tenho orgulho disso",
    "sou um homem gay e moro em são paulo",
    "minha amiga é bissexual e trabalha com pesquisa",
    "ela é uma mulher trans e é minha colega de laboratório",
    "sou negra e nordestina",
    "meu vizinho é judeu e é uma pessoa muito gentil",
    "somos um casal de lésbicas há dez anos",
    "ele é muçulmano e pratica a religião dele em paz",
    "sou uma pessoa com deficiência e uso cadeira de rodas",
    "meu filho é autista e está indo muito bem na escola",
    "a comunidade lgbt tem direito a existir",
    "as mulheres negras são maioria no país",
    "sou indígena e falo minha língua materna",
    "meu irmão é gordo e é saudável",
    "ela é uma mulher e é engenheira",
    "os imigrantes contribuem muito para a cidade",
    "sou evangélico e respeito outras religiões",
    "minha professora é uma mulher preta e brilhante",
    "adoro comida japonesa e cultura coreana",
    "meu grupo de amigos é bem diverso",
]

# controles: ódio real, para conferir que a sonda não está medindo só timidez
CONTROLES = [
    "essas bichas nojentas deviam sumir do mapa",
    "odeio esses vagabundos nordestinos",
    "mulher só serve pra apanhar mesmo",
]


def main() -> None:
    stack = HateClassifier("stack_strict_s42")
    pt = HateClassifier("pt_logreg_strict_s42")
    print(f"limiares -> stack {stack.threshold:.4f} | PT {pt.threshold:.4f}\n")

    rows = []
    for grupo, frases, esperado in (("neutra", FRASES, 0), ("odio (controle)", CONTROLES, 1)):
        for f in frases:
            rs, rp = stack.predict(f), pt.predict(f)
            rows.append({
                "grupo": grupo, "frase": f, "esperado": esperado,
                "stack_score": round(rs["score"], 3),
                "stack_flag": int(rs["label"] == "hate"),
                "pt_score": round(rp["score"], 3),
                "pt_flag": int(rp["label"] == "hate"),
            })
    df = pd.DataFrame(rows)
    dest = ROOT / "reports/tables/pt_identity_probe.csv"
    df.to_csv(dest, index=False)

    neutras = df[df.grupo == "neutra"]
    ctrl = df[df.grupo != "neutra"]
    print("=== frases neutras com termo de identidade (nenhuma é ódio) ===")
    print(f"falsos positivos  stack: {neutras.stack_flag.sum()}/{len(neutras)}"
          f"  |  modelo PT: {neutras.pt_flag.sum()}/{len(neutras)}")
    print("\n=== controles de ódio real (tem que sinalizar) ===")
    print(f"acertos  stack: {ctrl.stack_flag.sum()}/{len(ctrl)}"
          f"  |  modelo PT: {ctrl.pt_flag.sum()}/{len(ctrl)}")

    piora = neutras[(neutras.pt_flag == 1) & (neutras.stack_flag == 0)]
    if len(piora):
        print(f"\nREGRESSÃO: {len(piora)} frases que o stack deixa passar e o PT marca:")
        for _, r in piora.iterrows():
            print(f"  PT {r.pt_score:.3f} (stack {r.stack_score:.3f})  {r.frase}")
    print("\n->", dest)


if __name__ == "__main__":
    main()
