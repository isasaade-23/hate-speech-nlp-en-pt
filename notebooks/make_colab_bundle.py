"""Package everything the Colab neural notebook needs into a single small zip.

The corpus parquets and the hsc source are all that Colab lacks (the repo has no GitHub
remote and the processed data is gitignored). Run this locally after `hsc split`:

    .venv/Scripts/python.exe notebooks/make_colab_bundle.py

It writes `colab_bundle.zip` (~13 MB) to the project root. Upload that one file to your
Google Drive; the notebook unzips it and installs the package. Excludes the raw data,
the 125 MB embedding cache, caches and weights — only code, configs and the frozen
corpus travel.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Explicit includes: code + build metadata + configs + the frozen (corrected) corpus.
INCLUDE_FILES = [
    "pyproject.toml",
    "requirements-colab.txt",
    "data/processed/corpus_strict.parquet",
    "data/processed/corpus_broad.parquet",
]
INCLUDE_TREES = ["src/hsc", "configs"]
SKIP_SUFFIXES = {".pyc"}
SKIP_DIRS = {"__pycache__"}


def _add_tree(zf: zipfile.ZipFile, rel: str) -> int:
    n = 0
    base = ROOT / rel
    for p in base.rglob("*"):
        if p.is_dir() or any(part in SKIP_DIRS for part in p.parts) or p.suffix in SKIP_SUFFIXES:
            continue
        zf.write(p, p.relative_to(ROOT).as_posix())
        n += 1
    return n


def main() -> None:
    out = ROOT / "colab_bundle.zip"
    missing = [f for f in INCLUDE_FILES if not (ROOT / f).exists()]
    if missing:
        raise SystemExit(
            f"missing required files: {missing}\nRun `hsc split --policy strict` and "
            "`--policy broad` first so the corpus parquets exist."
        )
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in INCLUDE_FILES:
            zf.write(ROOT / f, f)
            count += 1
        for tree in INCLUDE_TREES:
            count += _add_tree(zf, tree)
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out} — {count} files, {size_mb:.1f} MB")
    print("Next: upload colab_bundle.zip to your Google Drive, then open notebooks/colab_neural.ipynb in Colab.")


if __name__ == "__main__":
    main()
