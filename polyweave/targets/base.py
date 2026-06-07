"""TargetSpec — the contract for packing/unpacking/installing generated weights.

A hypernetwork emits a flat parameter vector (or a structured set of maps); a
target layer needs that turned into correctly-shaped tensors and copied into the
right place in a student network. Historically this glue was hand-rolled and
error-prone in each experiment. A ``TargetSpec`` centralises it.

Every concrete spec exposes:

    .num_params              total scalar count the hypernetwork must produce
    .pack(weights)   -> Tensor   structured weights  -> flat 1-D vector
    .unpack(flat)    -> weights  flat 1-D vector      -> structured weights
    .install(into, weights)      copy structured weights into a torch object
    .extract(frm)    -> weights  read structured weights back out (for ensembles
                                  / checkpoints / round-trip tests)

"weights" is a spec-specific structure (a dict for FC/Conv, a list of dicts for
attention Q/K). ``pack``/``unpack`` define a canonical, stable flat ordering so a
generated vector and an extracted vector are interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class TargetSpec(ABC):
    """Abstract base for a generated-weight target layer."""

    @property
    @abstractmethod
    def num_params(self) -> int:
        """Total number of scalar parameters this target requires."""

    @abstractmethod
    def pack(self, weights: Any) -> torch.Tensor:
        """Flatten structured ``weights`` into a 1-D tensor of length ``num_params``."""

    @abstractmethod
    def unpack(self, flat: torch.Tensor) -> Any:
        """Inverse of :meth:`pack`: reshape a 1-D tensor into structured weights."""

    @abstractmethod
    def install(self, into: Any, weights: Any) -> None:
        """Copy structured ``weights`` into a concrete torch object (in place)."""

    @abstractmethod
    def extract(self, frm: Any) -> Any:
        """Read structured weights out of a concrete torch object (detached clones)."""

    # -- shared helpers -----------------------------------------------------

    def _check_flat(self, flat: torch.Tensor) -> torch.Tensor:
        flat = flat.reshape(-1)
        if flat.numel() != self.num_params:
            raise ValueError(
                f"{type(self).__name__}.unpack expected {self.num_params} elements, "
                f"got {flat.numel()}"
            )
        return flat
