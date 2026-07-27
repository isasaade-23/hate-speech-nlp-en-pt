"""Harmonize per-dataset interim frames into a unified binary corpus.

Applies configs/labels.yaml under a chosen policy (strict|broad), attaches
label_confidence and label_policy, cleans text, and concatenates the sources that
are `include_in_primary` for that policy. Sources excluded from primary can still be
returned as auxiliary (kept for external testing).

Downstream, splits.py adds `dup_cluster_id` + `split`, and langid.py adds `lang_pred`.
"""

from __future__ import annotations

import pandas as pd

from hsc.clean import clean_series, get_profile
from hsc.config import data_config, labels_config, resolve
from hsc.utils import get_logger, read_parquet, write_parquet

log = get_logger("hsc.harmonize")

_PROBE_FILE = "data/interim/tweets_ip_probe.json"  # written by Fase 2 notebook


def _map_label(value: str, mapping: dict, match: str):
    if match == "prefix":
        for key, lab in mapping.items():
            if value.startswith(key):
                return lab
        return None
    return mapping.get(value, None)


def _probe_recorded() -> bool:
    return resolve(_PROBE_FILE).exists()


def harmonize_source(
    sid: str,
    policy: str,
    labels_cfg: dict,
    data_cfg: dict,
    clean_profile: str,
    force_gated: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """Return (frame, include_in_primary) for one source under one policy."""
    src_cfg = labels_cfg["sources"][sid]
    pol = src_cfg[policy]
    mapping = {str(k): v for k, v in pol["map"].items()}
    match = src_cfg.get("label_match", labels_cfg.get("label_match", "exact"))
    conf_map = {str(k): v for k, v in src_cfg.get("confidence", {}).items()}

    interim_path = resolve(data_cfg["paths"]["interim"]) / f"{sid}.parquet"
    df = read_parquet(interim_path)

    label = df["label_original"].astype(str).map(lambda v: _map_label(v, mapping, match))
    keep = label.notna()
    dropped = int((~keep).sum())
    if dropped:
        log.info("%s [%s]: dropped %d rows with unmapped labels", sid, policy, dropped)
    df = df.loc[keep].copy()
    df["label"] = label.loc[keep].astype(int).values
    df["label_confidence"] = (
        df["label_original"].astype(str).map(lambda v: _map_conf(v, conf_map, match)).values
    )
    df["label_policy"] = policy

    profile = get_profile(data_cfg, clean_profile)
    df["text_clean"] = clean_series(df["text"].values, profile).values

    include = bool(pol.get("include_in_primary", True))
    if src_cfg.get("probe_gate", False) and not _probe_recorded() and not force_gated:
        if include:
            log.warning(
                "%s: probe_gate active and no probe recorded (%s missing) -> excluding "
                "from PRIMARY corpus (safe default). Run Fase 2 or pass force_gated=True.",
                sid,
                _PROBE_FILE,
            )
        include = False
    return df, include


def _map_conf(value: str, conf_map: dict, match: str):
    if match == "prefix":
        for key, c in conf_map.items():
            if value.startswith(key):
                return c
        return "low"
    return conf_map.get(value, "low")


CORPUS_COLUMNS = [
    "id",
    "text",
    "text_clean",
    "label",
    "label_original",
    "label_confidence",
    "label_policy",
    "language",
    "source_dataset",
    "domain",
]


def build_corpus(
    policy: str = "strict",
    clean_profile: str = "light",
    include_auxiliary: bool = False,
    force_gated: bool = False,
    write: bool = True,
) -> pd.DataFrame:
    labels_cfg = labels_config()
    data_cfg = data_config()

    primary, auxiliary = [], []
    for sid in labels_cfg["sources"]:
        df, include = harmonize_source(
            sid, policy, labels_cfg, data_cfg, clean_profile, force_gated
        )
        df = df[CORPUS_COLUMNS]
        (primary if include else auxiliary).append(df)
        log.info(
            "%s [%s]: %d rows -> %s", sid, policy, len(df), "PRIMARY" if include else "auxiliary"
        )

    frames = primary + (auxiliary if include_auxiliary else [])
    corpus = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CORPUS_COLUMNS)

    if write:
        out = resolve(data_cfg["paths"]["processed"]) / f"corpus_{policy}.parquet"
        write_parquet(corpus, out)
        log.info(
            "corpus [%s]: %d rows (hate=%d) -> %s",
            policy,
            len(corpus),
            int(corpus["label"].sum()) if len(corpus) else 0,
            out,
        )
    return corpus
