"""Shared utilities used across experiments and the library."""

from __future__ import annotations

import random

import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Seed Python, torch (CPU) and all CUDA devices for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(module: nn.Module, trainable_only: bool = False) -> int:
    """Total number of parameters in ``module``."""
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def freeze_all(module: nn.Module) -> None:
    """Set ``requires_grad = False`` on every parameter of ``module`` (in place)."""
    for p in module.parameters():
        p.requires_grad_(False)


def default_device() -> str:
    """Return ``"cuda"`` if available, else ``"cpu"``."""
    return "cuda" if torch.cuda.is_available() else "cpu"
