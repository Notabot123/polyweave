"""Student networks whose weights a hypernetwork teacher generates.

Two families, mirroring the paper's experiments:

* :class:`CNNStudent` — a small CIFAR-style convolutional classifier with its
  first conv layer (``conv1``/``bn1``/``pool1``) factored out so a teacher can
  replace just those weights (Experiment 2), while still allowing a generated
  linear head (Experiment 1). Three concrete architectures (A/B/C) provide the
  cross-architecture seen/unseen split.
* :class:`TinyTransformerStudent` — an attention-only transformer for the
  synthetic relational-lookup task whose per-layer Q/K projections a teacher
  generates (Experiment 3).
"""

from __future__ import annotations

from .cnn import CNNStudent, make_cnn_student, make_cnn_students
from .sigmapi import SigmaPiStudent
from .transformer import (
    TinyTransformerStudent,
    attn_layers,
    freeze_except_qk,
    mask_qk_grads,
    qk_params,
    reinit_qk,
)

__all__ = [
    "CNNStudent",
    "make_cnn_student",
    "make_cnn_students",
    "SigmaPiStudent",
    "TinyTransformerStudent",
    "attn_layers",
    "freeze_except_qk",
    "mask_qk_grads",
    "qk_params",
    "reinit_qk",
]
