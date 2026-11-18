"""Baixa as bases candidatas (OLID-BR, TuPy) para esta pasta. Só leitura do Hub."""
from pathlib import Path
from huggingface_hub import snapshot_download

base = Path(__file__).parent
for repo in ["dougtrajano/olid-br", "Silly-Machine/TuPy-Dataset"]:
    destino = base / repo.split("/")[1]
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=destino)
    arquivos = [p for p in destino.rglob("*") if p.is_file() and ".cache" not in p.parts]
    print(repo, len(arquivos), "arquivos")
    for p in arquivos:
        print("  ", p.relative_to(destino), p.stat().st_size)
