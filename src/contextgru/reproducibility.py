from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


DEFAULT_SEED = 1570254494


def resolve_device(name: Optional[str] = None) -> torch.device:
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
