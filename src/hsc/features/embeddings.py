"""Multilingual sentence-embedding features.

A strong classical baseline that shares XLM-R's multilingual idea without GPU
fine-tuning: encode each text with a frozen multilingual Sentence-BERT model, then
feed the dense vectors to LogReg / LightGBM. Unlike TF-IDF word features, these embed
EN and PT into the same space, so they can transfer cross-lingually (the Fase 9
experiment).

The encoder is FROZEN (no fitting): `fit` is a no-op, so anti-leakage is automatic —
nothing is learned from any split. Encoding is the expensive part on CPU, so results are
cached PER TEXT in a persistent key-value store keyed by (model, sha1(text)). Any run
that re-sees a text — a LightGBM run reusing a LogReg run's matrix, or a transfer slice
that is a subset of the corpus — reuses the cached vector instead of re-encoding.
"""

from __future__ import annotations

import hashlib
import pickle

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from hsc.config import resolve
from hsc.utils import ensure_dir, get_logger

log = get_logger("hsc.features.embeddings")

# Process-global per-model store so repeated transforms in one run share memory + disk.
_STORES: dict[str, dict[str, np.ndarray]] = {}


def _slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def _store_path(model_name: str):
    return ensure_dir(resolve("data/interim/emb_cache")) / f"{_slug(model_name)}_kv.pkl"


def _load_store(model_name: str) -> dict[str, np.ndarray]:
    if model_name not in _STORES:
        path = _store_path(model_name)
        if path.exists():
            with open(path, "rb") as fh:
                _STORES[model_name] = pickle.load(fh)
            log.info("loaded embedding cache: %d texts (%s)", len(_STORES[model_name]), path.name)
        else:
            _STORES[model_name] = {}
    return _STORES[model_name]


def _save_store(model_name: str) -> None:
    with open(_store_path(model_name), "wb") as fh:
        pickle.dump(_STORES[model_name], fh, protocol=pickle.HIGHEST_PROTOCOL)


def _text_key(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


class EmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """sklearn-compatible wrapper around a frozen sentence-transformers encoder.

    Kept API-compatible with the TF-IDF FeatureUnion (fit / transform / fit_transform)
    so `train.py` treats both feature families identically.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 64,
        normalize: bool = True,
        cache: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.cache = cache
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading encoder %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def __getstate__(self):
        # Never pickle the ~470 MB SentenceTransformer into the model joblib — it is a
        # frozen, re-downloadable asset. Persist only config; reload the encoder on demand.
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._model = None

    def fit(self, X, y=None):  # frozen encoder: nothing to learn
        return self

    def transform(self, X):
        texts = [str(t) for t in X]
        if not self.cache:
            return self._encode(texts).astype(np.float32)

        store = _load_store(self.model_name)
        keys = [_text_key(t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in store]
        if missing:
            enc = self._encode([texts[i] for i in missing])
            for j, i in enumerate(missing):
                store[keys[i]] = enc[j].astype(np.float16)  # halve disk; classifiers upcast
            _save_store(self.model_name)
            log.info("encoded %d new texts (%d cache hits) [%d total]", len(missing), len(texts) - len(missing), len(store))
        else:
            log.info("embedding cache: %d/%d hits (no encoding)", len(texts), len(texts))
        return np.stack([store[k] for k in keys]).astype(np.float32)

    def _encode(self, texts) -> np.ndarray:
        model = self._load()
        return model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )

    def fit_transform(self, X, y=None, **kw):
        return self.fit(X, y).transform(X)


def build_embeddings(cfg_features: dict) -> EmbeddingVectorizer:
    return EmbeddingVectorizer(
        model_name=cfg_features.get(
            "model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
        batch_size=int(cfg_features.get("batch_size", 64)),
        normalize=bool(cfg_features.get("normalize", True)),
        cache=bool(cfg_features.get("cache", True)),
    )


def seed_store_from_arrays(
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    text_col: str = "text_clean",
) -> int:
    """Recover the per-text KV cache from the per-array .npy files written by the earlier
    per-array cache scheme, so the ~40 min already spent encoding the corpus is not
    repeated. Reproduces each split's array hash, loads the matching .npy, and maps each
    text to its row. Idempotent."""
    import pandas as pd

    cache_dir = ensure_dir(resolve("data/interim/emb_cache"))
    store = _load_store(model_name)
    before = len(store)
    for policy in ("strict", "broad"):
        path = resolve("data/processed") / f"corpus_{policy}.parquet"
        if not path.exists():
            continue
        corpus = pd.read_parquet(path)
        for split in ("train", "val", "test"):
            texts = [str(t) for t in corpus[corpus["split"] == split][text_col].values]
            # legacy per-array hash: sha256(model + \x00 + each text + \x00)[:16]
            h = hashlib.sha256()
            h.update(model_name.encode("utf-8"))
            h.update(b"\x00")
            for t in texts:
                h.update(t.encode("utf-8"))
                h.update(b"\x00")
            npy = cache_dir / f"{h.hexdigest()[:16]}.npy"
            if not npy.exists():
                continue
            arr = np.load(npy)
            for t, v in zip(texts, arr):
                store[_text_key(t)] = v.astype(np.float16)
    _save_store(model_name)
    added = len(store) - before
    log.info("seeded embedding KV store: +%d texts (now %d)", added, len(store))
    return added
