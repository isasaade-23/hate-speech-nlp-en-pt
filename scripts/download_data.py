"""Download the public source datasets and pack them as the zips `hsc ingest` expects.

Each file is fetched from the authors' repository at a pinned commit, checked against
the sha256 of the copy used in corpus v5, and written to data/source_zips/<zip> under
the member name in configs/data.yaml. Licenses: data/MANIFEST.csv.

Sources without a public direct download are listed at the end with what to do.
corpus v5 needs tweets_ip (20,302 rows); multioff and memotion are external tests only.

Usage: python scripts/download_data.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GH = "https://raw.githubusercontent.com"

# source_id -> (url at pinned commit, sha256 of the file used in corpus v5)
PUBLIC = {
    "hatebr": (
        f"{GH}/franciellevargas/HateBR/2d18c5b9/dataset/HateBR.csv",
        "ae4f64b73b6833ee32585ceea2b3995bb829d27513e4dbec8d7ddff7236e5702",
    ),
    "toldbr": (
        f"{GH}/JAugusto97/ToLD-Br/6b325d26a9d25b321a3e9ba98ef98832b56729f5/ToLD-BR.csv",
        "a905b90d5886b9c80d737aa26c41dd29077155f23b9b857080634a78f8203c90",
    ),
    "vidgen": (
        f"{GH}/bvidgen/Dynamically-Generated-Hate-Speech-Dataset/"
        "36f9dc849626bc2f6d2e1c2861db556294a2a947/"
        "Dynamically%20Generated%20Hate%20Dataset%20v0.2.3.csv",
        "6e0aa7e598ab5b648cb1f0b48626a3e5f2e39a5abbd9fee998efa5a0b6cab4ca",
    ),
    "hatexplain": (
        f"{GH}/hate-alert/HateXplain/01d742279dac941981f53806154481c0e15ee686/Data/dataset.json",
        "63bb3340fee0ec469b09690d04cb68f7c187787dd8b83807f071892c084967fb",
    ),
    "pt_fortuna": (
        f"{GH}/paulafortuna/Portuguese-Hate-Speech-Dataset/"
        "4208e1e22fca62a2ee7d3ed74316e13b4e67b731/"
        "2019-05-28_portuguese_hate_speech_binary_classification.csv",
        "dc2370fc58a127a17d24ce2277c42f457b92b6b4270a07a90708912f9b2d3999",
    ),
}

MANUAL = {
    "tweets_ip": "no public source (origin page lost). Needed for corpus v5: "
    "ask the maintainers for 'Hate and offensive speech detection.csv' "
    "(sha256 8fefaa68593a11d6575608d1427b989907ee61f9c9219e9119d425a9629c5751).",
    "multioff": "Google Drive folder linked in the authors' README "
    "(https://drive.google.com/drive/folders/1hKLOtpVmF45IoBmJPwojgq6XraLtHmV6); "
    "zip the 'Split Dataset' CSVs under Dataset/Split Dataset/. External test only.",
    "memotion": "Kaggle (kaggle datasets download williamscott701/memotion-dataset-7k), "
    "needs a Kaggle account. External test only.",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite existing zips")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text(encoding="utf-8"))
    out_dir = ROOT / cfg["paths"]["source_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for sid, (url, sha) in PUBLIC.items():
        src = cfg["sources"][sid]
        zip_path = out_dir / src["zip"]
        if zip_path.exists() and not args.force:
            print(f"[skip] {sid}: {zip_path.name} already present")
            continue
        with urllib.request.urlopen(url) as r:
            data = r.read()
        got = hashlib.sha256(data).hexdigest()
        if got != sha:
            print(f"[FAIL] {sid}: sha256 {got} differs from corpus v5 ({sha})")
            failed.append(sid)
            continue
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(src["member"], data)
        print(f"[ok]   {sid}: {zip_path.name} ({len(data):,} bytes, sha256 matches)")

    print("\nManual sources:")
    for sid, how in MANUAL.items():
        present = (out_dir / cfg["sources"][sid]["zip"]).exists()
        print(f"  {sid} ({cfg['sources'][sid]['zip']}{', present' if present else ''}): {how}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
