"""Minimal checkpoint save/load for teachers and students.

Stores a module's ``state_dict`` alongside optional optimiser state and a free
-form ``meta`` dict (e.g. config, step count, validation accuracy). Useful for
persisting a trained teacher and a population of generated students so an
ensemble script can reload them without retraining.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Save ``model`` (and optionally ``optimizer``/``meta``) to ``path``.

    Parent directories are created if needed.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    payload: Dict[str, Any] = {"model_state": model.state_dict()}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if meta is not None:
        payload["meta"] = meta
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[Any] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load weights into ``model`` (and optionally ``optimizer``) from ``path``.

    Returns the stored ``meta`` dict (empty if none was saved).
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state"], strict=strict)
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload.get("meta", {})
