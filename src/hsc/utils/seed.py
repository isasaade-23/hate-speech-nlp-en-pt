"""Single seeding entrypoint, called by every train/eval entrypoint and logged."""

from __future__ import annotations

import os
import random


def set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy, and torch (if present). Deterministic on GPU is not
    guaranteed for transformers; rely on multi-seed CIs (see methodology)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
