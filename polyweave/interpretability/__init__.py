"""Interpretability probes for PolyWeave layers and teachers.

Currently: occlusion sensitivity, used to distinguish additive from
multiplicative (Sigma-Pi) features via their conjunctive AND-signature.
"""

from __future__ import annotations

from .occlusion import (
    conjunction_index,
    group_drops,
    occlusion_sensitivity_1d,
    occlusion_sensitivity_2d,
)

__all__ = [
    "occlusion_sensitivity_1d",
    "occlusion_sensitivity_2d",
    "group_drops",
    "conjunction_index",
]
