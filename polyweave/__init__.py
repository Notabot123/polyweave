"""PolyWeave — multiplicative (Sigma-Pi) layers and hypernetworks.

A small, modular library extracted from the Autodidact / Sigma-Pi hypernetwork
experiments. The architecture is layered:

    ops        — pure functions (signed-log, etc.)
    layers     — nn.Module building blocks (ConvSigmaPi2d, ...)
    targets    — pack / unpack / install generated weights for a target layer
    prototypes — compact support-set representations (statistical + learnable)
    students   — networks whose weights a teacher generates (CNN, transformer)
    hypernets  — full weight-generating teachers (vector- and map-head)
    training   — generic teacher-training loop + checkpoint I/O
    viz        — publication-quality plotting (PDF, large fonts, colourblind-safe)
    metrics    — diagnostics (pi-scale, ensemble disagreement)

v0.1 ships what the three paper experiments (FC, conv1, attention Q/K) exercise,
plus a learnable prototype encoder for real-world use. Polynomial / gated layers
and adapters are intentionally deferred to later versions.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import (
    distill,
    evaluation,
    hypernets,
    interpretability,
    metrics,
    ops,
    prototypes,
    students,
    targets,
    training,
    utils,
)
from .layers import ConvSigmaPi2d, PolyLinear, SigmaPiLinear
from .ops import signed_log, signed_log1p

__all__ = [
    "__version__",
    "ops",
    "targets",
    "prototypes",
    "students",
    "hypernets",
    "training",
    "evaluation",
    "distill",
    "interpretability",
    "metrics",
    "utils",
    "ConvSigmaPi2d",
    "SigmaPiLinear",
    "PolyLinear",
    "signed_log",
    "signed_log1p",
]
