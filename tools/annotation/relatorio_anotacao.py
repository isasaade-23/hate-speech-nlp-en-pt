"""Relatorio da validacao humana: kappa, acerto por tradutor, bandeiras, comentarios, pendencias.

Para cada mapeamento (*_projeto*.json em LUCIOLA_MAP_DIR) roda export_annotations.py com
--a isa --b vini --primeiros 100 (saida vai para o log) e monta reports/anotacao_<data>.md.
O relatorio nao traz o texto dos itens, so ids; os comentarios sao dos anotadores.

Variaveis de ambiente: LUCIOLA_SSH_KEY, LUCIOLA_SSH_HOST, LUCIOLA_MAP_DIR. Nenhuma senha.
Uso: python relatorio_anotacao.py [--a isa] [--b vini] [--primeiros 100]
"""
import argparse
import json
import os
import sqlite3
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import export_annotations as ex

ROOT = Path(__file__).resolve().parents[2]
ODIO, NAO = ex.ODIO, ex.NAO


def decisao(rotulos):
    core = rotulos & {ODIO, NAO}
    return next(iter(core)) if len(core) == 1 else None


def analisar(con, mapa, a, b, n):
    pid, itens = mapa["project_id"], mapa["itens"]
    ids = sorted(r[0] for r in con.execute("SELECT id FROM examples_example WHERE project_id=?", (pid,)))[:n]
    lab = defaultdict(lambda: defaultdict(set))
    for ex_id, user, text in con.execute(
            """SELECT c.example_id, u.username, t.text FROM labels_category c
               JOIN label_types_categorytype t ON t.id = c.label_id
               JOIN auth_user u ON u.id = c.user_id
               JOIN examples_example e ON e.id = c.example_id WHERE e.project_id=?""", (pid,)):
        lab[user][ex_id].add(text)
    coment = [(i, u, t) for i, u, t in con.execute(
        """SELECT c.example_id, u.username, c.text FROM examples_comment c
           JOIN auth_user u ON u.id = c.user_id
           JOIN examples_example e ON e.id = c.example_id WHERE e.project_id=? ORDER BY c.example_id""", (pid,))
        if i in set(ids)]
    final = {u: {i: decisao(lab[u].get(i, set())) for i in ids} for u in (a, b)}
    both = [i for i in ids if final[a][i] and final[b][i]]
    x, y = [final[a][i] for i in both], [final[b][i] for i in both]
    r = {"n": len(ids), "kind": mapa["kind"], "pid": pid, "ambos": len(both),
         "po": (sum(p == q for p, q in zip(x, y)) / len(both)) if both else None,
         "kappa": ex.kappa(x, y) if len(both) >= 2 else None, "faltam": {}, "acerto": {}, "bandeiras": [],
         "comentarios": coment}
    for u in (a, b):
        r["faltam"][u] = sum(1 for i in ids if final[u][i] is None)
        h = defaultdict(lambda: [0, 0])  # modelo -> [acertos, n]
        for i in ids:
            if final[u][i] is None:
                continue
            m = itens[str(i)]
            alvo = ODIO if m["rotulo_llm"] == 1 else NAO
            h[m.get("modelo", "?")][0] += final[u][i] == alvo
            h[m.get("modelo", "?")][1] += 1
        r["acerto"][u] = dict(h)
    for i in ids:
        for u in (a, b):
            fl = sorted(l for l in lab[u].get(i, set()) if l.startswith("⚑"))
            if fl:
                r["bandeiras"].append((i, itens[str(i)]["id_corpus"], u, ", ".join(fl)))
    return r


def md(res, a, b, n, hoje):
    L = [f"# Validação humana, {hoje}", "",
         f"Isa ({a}) e Vini ({b}), primeiros {n} itens de cada projeto. Sem texto dos itens, só ids.", ""]
    for r in res:
        L += [f"## Projeto {r['pid']} ({r['kind']})", ""]
        k = "n/d" if r["kappa"] is None or r["kappa"] != r["kappa"] else f"{r['kappa']:.3f}"
        po = "n/d" if r["po"] is None else f"{r['po']:.3f}"
        L += [f"- Itens completos nos dois (exatamente um entre Ódio e Não ódio): {r['ambos']} de {r['n']}",
              f"- Concordância bruta: {po}; kappa de Cohen: {k}"
              + ("" if r["ambos"] >= 2 else " (menos de 2 itens em comum, sem cálculo)"),
              "- Itens que faltam: " + ", ".join(f"{u} {r['faltam'][u]}" for u in (a, b)), "",
              "**Taxa de \"LLM acertou\"** (anotação igual ao rótulo da máquina; na tradução, o rótulo original da fonte), por " +
              ("tradutor" if r["kind"] == "traducao" else "modelo") + ":", "",
              "| anotador | modelo | acertou | itens | % |", "|---|---|---|---|---|"]
        linhas = 0
        for u in (a, b):
            for m, (ac, tot) in sorted(r["acerto"][u].items()):
                L.append(f"| {u} | {m} | {ac} | {tot} | {100 * ac / tot:.0f}% |")
                linhas += 1
        if not linhas:
            L.append("| (sem anotações completas) | | | | |")
        L += ["", "**Itens com ⚑:**", ""]
        L += [f"- item {i} (corpus {c}), {u}: {f}" for i, c, u, f in r["bandeiras"]] or ["- nenhum"]
        L += ["", "**Comentários:**", ""]
        L += [f"- item {i}, {u}: {t}" for i, u, t in r["comentarios"]] or ["- nenhum"]
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="isa")
    ap.add_argument("--b", default="vini")
    ap.add_argument("--primeiros", type=int, default=100)
    args = ap.parse_args()
    mapdir = Path(os.environ["LUCIOLA_MAP_DIR"])
    mapas = sorted(mapdir.glob("*_projeto*.json"))
    if not mapas:
        raise SystemExit(f"nenhum mapeamento *_projeto*.json em {mapdir}")
    hoje = date.today().isoformat()
    db = ex.fetch_db()
    try:
        for m in mapas:  # a exportacao propriamente dita, com a saida no log
            print(f"=== export_annotations.py {m.name} --a {args.a} --b {args.b} --primeiros {args.primeiros}", flush=True)
            subprocess.run([sys.executable, str(Path(__file__).with_name("export_annotations.py")), str(m),
                            "--a", args.a, "--b", args.b, "--primeiros", str(args.primeiros), "--db", str(db)],
                           check=True)
        con = sqlite3.connect(db)
        res = [analisar(con, json.loads(m.read_text(encoding="utf-8")), args.a, args.b, args.primeiros) for m in mapas]
        con.close()
    finally:
        db.unlink(missing_ok=True)
        db.parent.rmdir()
    saida = ROOT / "reports" / f"anotacao_{hoje}.md"
    saida.write_text(md(res, args.a, args.b, args.primeiros, hoje), encoding="utf-8")
    print("relatorio:", saida)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
