"""Training utilities: a generic teacher-training loop and checkpoint I/O."""

from __future__ import annotations

from .checkpoint import load_checkpoint, save_checkpoint
from .loop import TeacherTrainResult, train_teacher

__all__ = [
    "train_teacher",
    "TeacherTrainResult",
    "save_checkpoint",
    "load_checkpoint",
]
