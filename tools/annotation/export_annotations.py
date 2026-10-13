"""Exportacao e metricas da validacao humana (servidor).

- Baixa uma copia consistente do banco do doccano (sqlite .backup via ssh) e le as
  anotacoes direto dela; a copia e apagada ao final.
- Exige EXATAMENTE UM entre 'Ódio' e 'Não ódio' por item e anotador; itens sem isso
  sao listados como incompletos e ficam fora das metricas.
- Kappa de Cohen entre os dois anotadores SO no par Ódio/Não ódio.
- 'LLM acertou/errou' e calculado aqui, comparando a anotacao com o mapeamento
  (rotulo da LLM) que fica fora do doccano; o anotador nunca ve o rotulo da LLM.
- Os rotulos opcionais '⚑ ...' sao apenas contados.

Uso: python export_annotations.py <mapeamento.json> [--a anotador1] [--b anotador2] [--primeiros 100] [--db copia.sqlite3]
Isa e Vini: --a isa --b vini --primeiros 100 (amostra comum).
"""
import argparse
import os
import csv
import json
import sqlite3
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

KEY = os.environ.get("LUCIOLA_SSH_KEY")    # chave SSH do servidor (fora do repositorio)
HOST = os.environ.get("LUCIOLA_SSH_HOST")  # usuario@host do servidor
ODIO, NAO = "Ódio", "Não ódio"


def fetch_db() -> Path:
    if not (KEY and HOST):
        raise SystemExit("defina LUCIOLA_SSH_KEY e LUCIOLA_SSH_HOST (ou passe --db)")
    tmp = Path(tempfile.mkdtemp()) / "doccano.db"
    subprocess.run(["ssh", "-i", KEY, HOST,
                    "sqlite3 /srv/doccano-data/doccano.db \".backup '/tmp/dl.db'\""], check=True)
    subprocess.run(["scp", "-q", "-i", KEY, f"{HOST}:/tmp/dl.db", str(tmp)], check=True)
    subprocess.run(["ssh", "-i", KEY, HOST, "rm -f /tmp/dl.db"], check=True)
    return tmp


def kappa(x, y):
    n = len(x)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in zip(x, y)) / n
    cats = set(x) | set(y)
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in cats)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("mapa")
    ap.add_argument("--a", default="anotador1")
    ap.add_argument("--b", default="anotador2")
    ap.add_argument("--db")
    ap.add_argument("--primeiros", type=int, help="restringe aos N primeiros itens (ordem do doccano)")
    args = ap.parse_args()
    mp = json.loads(Path(args.mapa).read_text(encoding="utf-8"))
    pid, itens = mp["project_id"], mp["itens"]

    db = Path(args.db) if args.db else fetch_db()
    con = sqlite3.connect(db)
    rows = con.execute(
        """SELECT c.example_id, u.username, t.text FROM labels_category c
           JOIN label_types_categorytype t ON t.id = c.label_id
           JOIN auth_user u ON u.id = c.user_id
           JOIN examples_example e ON e.id = c.example_id WHERE e.project_id = ?""", (pid,)).fetchall()
    ex_ids = [r[0] for r in con.execute("SELECT id FROM examples_example WHERE project_id=?", (pid,))]
    con.close()
    ex_ids = sorted(ex_ids)[: args.primeiros] if args.primeiros else sorted(ex_ids)
    if not args.db:
        db.unlink()
        db.parent.rmdir()

    lab = defaultdict(lambda: defaultdict(set))  # user -> example -> labels
    for ex, user, text in rows:
        lab[user][ex].add(text)

    def decisao(user, ex):
        s = lab[user].get(ex, set())
        core = s & {ODIO, NAO}
        return (next(iter(core)) if len(core) == 1 else None), sorted(s - {ODIO, NAO})

    print(f"projeto {pid}: {len(ex_ids)} itens")
    final = {}
    for u in (args.a, args.b):
        inc = [ex for ex in ex_ids if decisao(u, ex)[0] is None]
        feitos = len(ex_ids) - len(inc)
        flags = sum(1 for ex in ex_ids if decisao(u, ex)[1])
        acertos = [decisao(u, ex)[0] == (ODIO if itens[str(ex)]["rotulo_llm"] == 1 else NAO)
                   for ex in ex_ids if decisao(u, ex)[0] is not None]
        print(f"\n{u}: completos {feitos}/{len(ex_ids)}, incompletos {len(inc)}, com bandeira ⚑ {flags}")
        if inc:
            print("  incompletos (ids doccano):", inc[:20], "..." if len(inc) > 20 else "")
        if acertos:
            print(f"  LLM acertou: {sum(acertos)}/{len(acertos)} = {sum(acertos) / len(acertos):.3f}  "
                  f"| LLM errou: {len(acertos) - sum(acertos)}")
        final[u] = {ex: decisao(u, ex)[0] for ex in ex_ids}

    ambos = [ex for ex in ex_ids if final[args.a][ex] and final[args.b][ex]]
    x = [final[args.a][ex] for ex in ambos]
    y = [final[args.b][ex] for ex in ambos]
    print(f"\nPar Ódio/Não ódio, itens completos nos dois: {len(ambos)}")
    if ambos:
        print(f"  concordância bruta {sum(a == b for a, b in zip(x, y)) / len(ambos):.3f}  "
              f"kappa de Cohen {kappa(x, y):.3f}")
    out = Path(args.mapa).with_name(f"resultado_projeto{pid}.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id_doccano", "id_corpus", "rotulo_llm", args.a, args.b])
        for ex in ex_ids:
            m = itens[str(ex)]
            w.writerow([ex, m["id_corpus"], ODIO if m["rotulo_llm"] == 1 else NAO,
                        final[args.a][ex] or "", final[args.b][ex] or ""])
    print(f"\nCSV por item (sem texto): {out}")


if __name__ == "__main__":
    main()
