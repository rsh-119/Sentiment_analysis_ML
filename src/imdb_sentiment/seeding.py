"""Reproducibility utilities: seed every source of randomness the project touches."""
from __future__ import annotations

import random

import numpy as np
import torch
import transformers


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch (CPU+CUDA) and HF Transformers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int, base_seed: int) -> None:
    """Pass as `DataLoader(worker_init_fn=partial(worker_init_fn, base_seed=seed))`."""
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def make_generator(seed: int) -> torch.Generator:
    """A seeded torch.Generator for DataLoader(generator=...) and shuffling."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
