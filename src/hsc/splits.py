"""Fase 4 — leakage-safe splitting.

Steps:
  1. Drop EXACT duplicates (normalized text), keeping the higher-confidence label.
  2. Cluster NEAR-duplicates (MinHash/LSH over char shingles) -> dup_cluster_id.
  3. Split into train/val/test with StratifiedGroupKFold: stratified on
     (language, source_dataset, label), grouped by dup_cluster_id so paraphrases
     never straddle splits. Frozen with a content hash.

The hard guarantee (enforced by tests/test_splits.py): no text and no cluster
appears in more than one split.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from hsc.config import data_config, resolve
from hsc.utils import get_logger, read_parquet, sha256_text, write_json, write_parquet

log = get_logger("hsc.splits")

_WS = re.compile(r"\s+")


def normalize_for_dedup(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = _WS.sub(" ", t).strip()
    return t


_CONF_RANK = {"high": 1, "low": 0}


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep one row per normalized text, preferring higher label_confidence."""
    key = df["text_clean"].map(normalize_for_dedup)
    rank = df["label_confidence"].map(_CONF_RANK).fillna(0)
    order = df.assign(_key=key.values, _rank=rank.values).sort_values(
        "_rank", ascending=False, kind="stable"
    )
    kept = order.drop_duplicates(subset="_key", keep="first").drop(columns=["_key", "_rank"])
    kept = kept.sort_index()
    removed = len(df) - len(kept)
    return kept.reset_index(drop=True), removed


def _shingles(text: str, k: int = 5) -> set[str]:
    t = normalize_for_dedup(text)
    if len(t) < k:
        return {t} if t else set()
    return {t[i : i + k] for i in range(len(t) - k + 1)}


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def near_dup_cluster_ids(texts, threshold: float = 0.90, num_perm: int = 64, k: int = 5):
    """Return an array of cluster ids; near-duplicates share an id (MinHash/LSH)."""
    from datasketch import MinHash, MinHashLSH

    texts = list(texts)
    n = len(texts)
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    mh = {}
    for i, t in enumerate(texts):
        m = MinHash(num_perm=num_perm)
        for sh in _shingles(t, k):
            m.update(sh.encode("utf-8"))
        lsh.insert(str(i), m)
        mh[i] = m

    uf = _UnionFind(n)
    for i in range(n):
        for j in lsh.query(mh[i]):
            j = int(j)
            if j != i:
                uf.union(i, j)
    roots = [uf.find(i) for i in range(n)]
    # compact to 0..C-1
    remap = {r: c for c, r in enumerate(sorted(set(roots)))}
    return np.array([remap[r] for r in roots], dtype=int)


def _merge_vidgen_pair_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Vidgen rounds 2-4 ship original/perturbation PAIRS (acl.id.matched): small
    edits, usually label-flipping, that char-3gram cosine at 0.90 mostly misses
    (measured 2026-08-16: only ~1.1k of ~15k pairs clustered). The dataset's own
    links are ground truth for 'these are paraphrases', so union each pair's
    clusters before the group split — pairs then never straddle splits."""
    mask = df["id"].astype(str).str.startswith("vidgen_")
    if not mask.any():
        return df
    raw_csv = resolve("data/raw") / "vidgen" / "dghs.csv"
    if not raw_csv.exists():
        log.warning("vidgen pairs: raw csv missing, skipping pair-cluster merge")
        return df
    import csv as _csv

    with open(raw_csv, encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    # loader ids are row-order: vidgen_<i>
    acl_to_corpusid = {}
    matched = {}
    for i, r in enumerate(rows):
        acl_to_corpusid[r["acl.id"]] = f"vidgen_{i}"
        if r["acl.id.matched"] not in ("NA", ""):
            matched[r["acl.id"]] = r["acl.id.matched"]

    id_to_cluster = dict(zip(df["id"], df["dup_cluster_id"]))
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_linked = 0
    for a, b in matched.items():
        ca = id_to_cluster.get(acl_to_corpusid.get(a))
        cb = id_to_cluster.get(acl_to_corpusid.get(b))
        if ca is None or cb is None:
            continue  # one side dropped by exact dedup
        ra, rb = find(ca), find(cb)
        if ra != rb:
            parent[ra] = rb
            n_linked += 1
    df = df.copy()
    df["dup_cluster_id"] = [find(c) if c in parent else c for c in df["dup_cluster_id"]]
    log.info("vidgen pairs: merged %d cluster links from acl.id.matched", n_linked)
    return df


def assign_splits(df: pd.DataFrame, ratios: dict, seed: int) -> np.ndarray:
    y = (
        df["language"].astype(str)
        + "|"
        + df["source_dataset"].astype(str)
        + "|"
        + df["label"].astype(str)
    ).values
    groups = df["dup_cluster_id"].values
    idx = np.arange(len(df))

    n_test = max(2, round(1 / ratios["test"]))
    sgkf = StratifiedGroupKFold(n_splits=n_test, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(sgkf.split(idx, y, groups))

    val_frac_rem = ratios["val"] / (ratios["train"] + ratios["val"])
    n_val = max(2, round(1 / val_frac_rem))
    sgkf2 = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=seed)
    tr_rel, val_rel = next(
        sgkf2.split(train_val_idx, y[train_val_idx], groups[train_val_idx])
    )
    train_idx = train_val_idx[tr_rel]
    val_idx = train_val_idx[val_rel]

    split = np.empty(len(df), dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"
    return split


def make_splits(df: pd.DataFrame, ratios: dict, seed: int, near_threshold: float = 0.90):
    """Full pipeline on an in-memory corpus frame. Returns (df_with_split, stats)."""
    df0 = df.reset_index(drop=True)
    df1, n_exact = drop_exact_duplicates(df0)
    log.info("exact dedup: removed %d of %d rows", n_exact, len(df0))
    df1 = df1.copy()
    df1["dup_cluster_id"] = near_dup_cluster_ids(df1["text_clean"].values, threshold=near_threshold)
    df1 = _merge_vidgen_pair_clusters(df1)
    n_clusters = int(df1["dup_cluster_id"].nunique())
    n_near = len(df1) - n_clusters
    log.info("near-dup: %d rows -> %d clusters (%d rows share a cluster)", len(df1), n_clusters, n_near)
    df1["split"] = assign_splits(df1, ratios, seed)
    stats = {
        "rows_in": int(len(df0)),
        "exact_removed": int(n_exact),
        "rows_after_exact": int(len(df1)),
        "near_clusters": n_clusters,
        "split_counts": df1["split"].value_counts().to_dict(),
    }
    return df1, stats


def run_split(policy: str = "strict") -> dict:
    data_cfg = data_config()
    seed = int(data_cfg.get("seed", 42))
    ratios = data_cfg["split"]["ratios"]
    near_threshold = float(data_cfg["dedup"]["near_threshold"])

    processed = resolve(data_cfg["paths"]["processed"])
    corpus_path = processed / f"corpus_{policy}.parquet"
    df = read_parquet(corpus_path)

    df_split, stats = make_splits(df, ratios, seed, near_threshold)
    write_parquet(df_split, corpus_path)  # overwrite with dup_cluster_id + split columns

    # freeze a manifest with a content hash of (id, split) pairs
    signature = "\n".join(
        f"{i}:{s}" for i, s in zip(df_split["id"].tolist(), df_split["split"].tolist())
    )
    manifest = {
        "policy": policy,
        "seed": seed,
        "ratios": ratios,
        "near_threshold": near_threshold,
        "stats": stats,
        "split_sha256": sha256_text(signature),
    }
    write_json(manifest, processed / f"splits_{policy}.json")
    log.info("splits [%s]: %s | hash=%s", policy, stats["split_counts"], manifest["split_sha256"][:12])
    return manifest
