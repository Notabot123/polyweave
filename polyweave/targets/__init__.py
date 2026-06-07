"""Target specs: pack / unpack / install generated weights for a target layer."""

from __future__ import annotations

from .attention import AttentionQKTargetSpec
from .base import TargetSpec
from .conv import Conv2dTargetSpec
from .fc import FCTargetSpec
from .sigmapi_conv import SigmaPiConvTargetSpec

__all__ = [
    "TargetSpec",
    "FCTargetSpec",
    "Conv2dTargetSpec",
    "AttentionQKTargetSpec",
    "SigmaPiConvTargetSpec",
]
