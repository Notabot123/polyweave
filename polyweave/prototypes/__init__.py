"""Prototype builders — compact support-set representations fed to a teacher.

A *prototype* is the teacher's only view of a few-shot support set. Every builder
returns a tensor shaped ``[1, channels, H, W]`` (a batch of one "image" with
``channels`` statistic maps) that a convolutional teacher consumes.

Two flavours are provided:

* **Statistical** (parameter-free), matching the paper:
    - :func:`feature_class_stats`  -> ``[1, 4, num_classes, feature_dim]``
      (per-class mean/variance/kurtosis/contrast of student features; Exp 1)
    - :func:`image_grid_stats`     -> ``[1, 4, num_classes, grid^2 * in_ch]``
      (class-conditional stats over a spatial grid of raw inputs; Exp 2)
    - :func:`relation_cross_moments` -> ``[1, 4, d_model, d_model]``
      (embedding-space query/key cross-moments; Exp 3)

* **Learnable** (:class:`LearnablePrototypeEncoder`): a trainable encoder that
  maps support features to a prototype, for real-world settings where hand-built
  statistics are too rigid. Output shape matches the statistical feature builder
  so it is a drop-in replacement for the teacher.
"""

from __future__ import annotations

from .learnable import LearnablePrototypeEncoder
from .statistical import (
    feature_class_stats,
    image_grid_stats,
    normalize_prototype,
    relation_cross_moments,
)

__all__ = [
    "feature_class_stats",
    "image_grid_stats",
    "relation_cross_moments",
    "normalize_prototype",
    "LearnablePrototypeEncoder",
]
