"""Configuration loading and project paths.

A run is fully described by (config file + seed + git SHA). This module resolves
the project root, loads YAML configs, and exposes canonical paths so no module
hard-codes a location.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Repo root = two levels up from this file (src/hsc/config.py -> repo)."""
    return Path(__file__).resolve().parents[2]


def load_yaml(rel_or_abs: str | Path) -> dict[str, Any]:
    """Load a YAML config. Relative paths are resolved against the project root."""
    p = Path(rel_or_abs)
    if not p.is_absolute():
        p = project_root() / p
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def data_config() -> dict[str, Any]:
    return load_yaml("configs/data.yaml")


def labels_config() -> dict[str, Any]:
    return load_yaml("configs/labels.yaml")


def resolve(path_str: str | Path) -> Path:
    """Resolve a possibly-relative path from a config against the project root."""
    p = Path(path_str)
    return p if p.is_absolute() else project_root() / p
