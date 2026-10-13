"""Cohen's kappa entre dois anotadores, lendo a EXPORTACAO do doccano (nao o
banco direto). Gera a exportacao individual (1 arquivo por anotador, porque
o projeto tem collaborative_annotation=False) chamando a mesma funcao que a
API de exportacao usa, so que sincronamente (sem precisar de worker Celery
rodando para este teste local).

Calcula kappa separadamente para os dois conjuntos de rotulo:
  - Odio / Nao odio
  - LLM acertou / LLM errou

Uso: python kappa.py <project_id> <anotador_a> <anotador_b>
"""
import json
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")
# pasta "backend" do pacote doccano instalado no venv do doccano local
sys.path.insert(0, os.environ["DOCCANO_BACKEND_PATH"])

# mesmo bootstrap que o CLI `doccano` faz em cli.py antes de configurar o Django --
# sem isso, DATABASE_URL cai no default (backend/db.sqlite3, vazio) em vez do banco
# real que `doccano webserver` usa (~/doccano/db.sqlite3).
_DOCCANO_HOME = os.path.expanduser(os.environ.get("DOCCANO_HOME", "~/doccano"))
os.environ.setdefault("DATABASE_URL", os.path.join(f"sqlite:///{_DOCCANO_HOME}", "db.sqlite3"))
os.environ.setdefault("MEDIA_ROOT", os.path.join(_DOCCANO_HOME, "media"))

import django  # noqa: E402

django.setup()

from collections import Counter  # noqa: E402

from data_export.celery_tasks import export_dataset  # noqa: E402

HATE_SET = {"Ódio", "Não ódio"}
LLM_SET = {"LLM acertou", "LLM errou"}


def cohen_kappa_score(y_a: list[str], y_b: list[str]) -> float:
    """Cohen's kappa, sem depender de scikit-learn (conflito de versão de
    numpy com as dependencias fixadas do doccano 1.8.4). Formula padrao:
    kappa = (p_o - p_e) / (1 - p_e)."""
    n = len(y_a)
    po = sum(1 for a, b in zip(y_a, y_b) if a == b) / n
    labels = sorted(set(y_a) | set(y_b))
    ca, cb = Counter(y_a), Counter(y_b)
    pe = sum((ca.get(lab, 0) / n) * (cb.get(lab, 0) / n) for lab in labels)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def load_jsonl_from_zip(zip_path: str, member: str) -> dict[int, list[str]]:
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        with z.open(f"{member}.jsonl") as f:
            for line in f:
                row = json.loads(line)
                out[row["id"]] = row.get("label", [])
    return out


def kappa_for_subset(a: dict, b: dict, label_set: set[str]) -> tuple[float, int]:
    common_ids = sorted(set(a) & set(b))
    ya, yb = [], []
    for i in common_ids:
        la = [x for x in a[i] if x in label_set]
        lb = [x for x in b[i] if x in label_set]
        if len(la) == 1 and len(lb) == 1:
            ya.append(la[0])
            yb.append(lb[0])
    if len(ya) < 2:
        return float("nan"), len(ya)
    return cohen_kappa_score(ya, yb), len(ya)


def main():
    project_id = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    user_a = sys.argv[2] if len(sys.argv) > 2 else "anotador1"
    user_b = sys.argv[3] if len(sys.argv) > 3 else "anotador2"

    zip_path = export_dataset(project_id=project_id, file_format="JSONL", confirmed_only=False)
    print(f"exportacao gerada: {zip_path}")

    a = load_jsonl_from_zip(zip_path, user_a)
    b = load_jsonl_from_zip(zip_path, user_b)
    print(f"{user_a}: {len(a)} itens anotados | {user_b}: {len(b)} itens anotados")

    k_hate, n_hate = kappa_for_subset(a, b, HATE_SET)
    k_llm, n_llm = kappa_for_subset(a, b, LLM_SET)

    print(f"\nCohen's kappa [Ódio/Não ódio]: {k_hate:.4f} (n={n_hate} itens em comum, ambos rotulados)")
    print(f"Cohen's kappa [LLM acertou/errou]: {k_llm:.4f} (n={n_llm} itens em comum, ambos rotulados)")


if __name__ == "__main__":
    main()
