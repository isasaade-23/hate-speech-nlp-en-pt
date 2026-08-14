"""Ingestion orchestration: extract each source's tabular member(s) into data/raw,
normalize to the common schema, and write data/interim/<source>.parquet.
"""

from __future__ import annotations

from pathlib import Path

from hsc.config import data_config, resolve
from hsc.ingest import (
    dataset1_memotion,
    dataset2_tweets,
    dataset3_pt,
    dataset4_multioff,
    dataset5_hatebr,
    dataset6_toldbr,
)
from hsc.ingest.base import extract_to_raw, log, validate_interim
from hsc.utils import ensure_dir, write_json, write_parquet

REGISTRY = {
    "memotion": dataset1_memotion.load,
    "tweets_ip": dataset2_tweets.load,
    "pt_fortuna": dataset3_pt.load,
    "multioff": dataset4_multioff.load,
    "hatebr": dataset5_hatebr.load,
    "toldbr": dataset6_toldbr.load,
}


def run_ingest(cfg: dict | None = None) -> dict:
    cfg = cfg or data_config()
    source_dir = Path(cfg["paths"]["source_dir"])
    raw_root = resolve(cfg["paths"]["raw"])
    interim_root = resolve(cfg["paths"]["interim"])
    ensure_dir(raw_root)
    ensure_dir(interim_root)

    provenance, stats = [], []
    for sid, loader in REGISTRY.items():
        cfg_source = cfg["sources"][sid]
        prov = extract_to_raw(sid, cfg_source, source_dir, raw_root)
        df = loader(cfg_source, raw_root)
        st = validate_interim(df, sid)
        write_parquet(df, interim_root / f"{sid}.parquet")
        prov["rows"] = st["rows"]
        provenance.append(prov)
        stats.append(st)
        log.info(
            "%s: %d rows | %d empty text | labels=%s",
            sid,
            st["rows"],
            st["empty_text"],
            st["labels"],
        )

    write_json({"sources": provenance}, raw_root / "PROVENANCE.json")
    log.info("wrote provenance -> %s", raw_root / "PROVENANCE.json")
    return {"stats": stats, "provenance": provenance}


__all__ = ["run_ingest", "REGISTRY"]
