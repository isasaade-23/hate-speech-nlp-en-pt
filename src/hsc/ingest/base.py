"""Shared ingestion helpers: extract tabular members from the source zips into
data/raw, read them with the verified per-dataset encoding, and assemble the
common interim schema.

Text-only project: only CSV/label members are extracted; meme images are ignored.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from hsc.schema import INTERIM_COLUMNS, VALID_DOMAINS, VALID_LANGUAGES, VALID_SOURCES
from hsc.utils import ensure_dir, get_logger, sha256_file

log = get_logger("hsc.ingest")


def _prior_zip_sha(source_id: str, raw_root: Path) -> str:
    """Carry the zip sha256 recorded in the previous PROVENANCE.json forward when
    the zip itself is gone; the raw members are what the pipeline actually reads."""
    import json

    prov_path = raw_root / "PROVENANCE.json"
    if prov_path.exists():
        with open(prov_path, encoding="utf-8") as f:
            for s in json.load(f).get("sources", []):
                if s.get("source_id") == source_id and s.get("zip_sha256"):
                    return s["zip_sha256"] + " (recorded before zip loss)"
    return "unknown (zip lost before hashing could be repeated)"


def _members(cfg_source: dict) -> list[str]:
    if "members" in cfg_source:
        return list(cfg_source["members"])
    return [cfg_source["member"]]


def extract_to_raw(source_id: str, cfg_source: dict, source_dir: Path, raw_root: Path) -> dict:
    """Copy this source's tabular member(s) out of its zip into data/raw/<source_id>/.
    Returns provenance: zip name, sha256, and the extracted file paths."""
    zip_path = source_dir / cfg_source["zip"]
    out_dir = ensure_dir(raw_root / source_id)
    if not zip_path.exists():
        # The original source zips for datasets 1-4 were lost from the source dir
        # (verified 2026-08-16: not on any local drive). The extracted members in
        # data/raw survive with their provenance recorded; reuse them and say so
        # explicitly instead of pretending the zip was re-read.
        members = _members(cfg_source)
        extracted = [out_dir / Path(m).name for m in members]
        missing = [p for p in extracted if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"source zip not found: {zip_path} and raw members missing: {missing}"
            )
        log.warning("%s: source zip absent, reusing previously extracted raw files", source_id)
        prior = _prior_zip_sha(source_id, raw_root)
        return {
            "source_id": source_id,
            "zip": cfg_source["zip"],
            "zip_sha256": prior,
            "zip_absent_reused_raw": True,
            "members": members,
            "encoding": cfg_source["encoding"],
            "extracted": [str(p.relative_to(raw_root.parent)) for p in extracted],
        }
    extracted = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in _members(cfg_source):
            raw = zf.read(member)
            dest = out_dir / Path(member).name
            dest.write_bytes(raw)
            extracted.append(str(dest.relative_to(raw_root.parent)))
    return {
        "source_id": source_id,
        "zip": cfg_source["zip"],
        "zip_sha256": sha256_file(zip_path),
        "members": _members(cfg_source),
        "encoding": cfg_source["encoding"],
        "extracted": extracted,
    }


def read_csv_member(path: str | Path, encoding: str) -> pd.DataFrame:
    """Read a CSV keeping everything as string and NOT interpreting NA tokens, so
    labels and text are preserved verbatim (record count respects quoted newlines)."""
    return pd.read_csv(path, encoding=encoding, dtype=str, keep_default_na=False, na_values=[])


def read_raw_csv(source_id: str, member: str, encoding: str, raw_root: Path) -> pd.DataFrame:
    path = raw_root / source_id / Path(member).name
    return read_csv_member(path, encoding)


def build_interim(
    *,
    source: str,
    language: str,
    domain: str,
    text: pd.Series,
    label_original: pd.Series,
) -> pd.DataFrame:
    """Assemble the common interim schema with stable ids."""
    assert language in VALID_LANGUAGES, language
    assert domain in VALID_DOMAINS, domain
    assert source in VALID_SOURCES, source
    n = len(text)
    out = pd.DataFrame(
        {
            "id": [f"{source}_{i}" for i in range(n)],
            "text": text.astype(str).str.strip().values,
            "label_original": label_original.astype(str).str.strip().values,
            "language": language,
            "source_dataset": source,
            "domain": domain,
        }
    )
    return out[INTERIM_COLUMNS]


def validate_interim(df: pd.DataFrame, source: str) -> dict:
    """Basic integrity checks; return a small stats dict for logging/provenance."""
    assert list(df.columns) == INTERIM_COLUMNS, f"{source}: columns mismatch {list(df.columns)}"
    assert df["id"].is_unique, f"{source}: ids not unique"
    n_empty = int((df["text"].str.len() == 0).sum())
    label_counts = df["label_original"].value_counts().to_dict()
    return {"source": source, "rows": int(len(df)), "empty_text": n_empty, "labels": label_counts}
