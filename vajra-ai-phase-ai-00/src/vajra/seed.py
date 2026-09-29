"""Reproducibility utility for VAJRA-AI.

Seeds Python's random module, NumPy, and PyTorch (CPU + all CUDA devices).
Every training phase from AI-4 onward should call set_seed() before building
the model/dataloaders. Full bitwise determinism is not guaranteed on GPU
(cuDNN algorithm selection can vary), but this makes runs consistent enough
to reason about.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set the random seed across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def check_reproducibility(seed: int = 42) -> bool:
    """Sanity check: seeding twice with the same seed should give the same
    random tensor. Returns True if reproducible, False otherwise.
    """
    set_seed(seed)
    a = torch.rand(4, 4)
    set_seed(seed)
    b = torch.rand(4, 4)
    return torch.equal(a, b)
