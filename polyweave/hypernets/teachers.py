"""Teacher architectures shared across the three experiments.

Vanilla and Sigma-Pi encoders share a common interface (``[1, C, H, W] ->
[1, width, H, W]``); the only difference is whether the middle layer is an
ordinary conv stack or a :class:`~polyweave.layers.ConvSigmaPi2d` block. Each
teacher attaches one of two heads to that encoder.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..layers import ConvSigmaPi2d
from ..targets.base import TargetSpec

LayerQK = Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# Encoders (spatial-preserving)
# ---------------------------------------------------------------------------

def _maybe_dropout(p: float) -> List[nn.Module]:
    """A ``Dropout2d(p)`` if ``p > 0`` else nothing (keeps Sequentials tidy)."""
    return [nn.Dropout2d(p)] if p > 0 else []


class _VanillaEncoder(nn.Module):
    """Two conv-BN-ReLU(-Dropout2d) layers, preserving spatial resolution."""

    def __init__(self, in_ch: int, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            *_maybe_dropout(dropout),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            *_maybe_dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SigmaPiEncoder(nn.Module):
    """conv-BN-ReLU(-Dropout2d) then a Sigma-Pi block, preserving spatial resolution."""

    def __init__(self, in_ch: int, width: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            *_maybe_dropout(dropout),
        )
        self.sigmapi = ConvSigmaPi2d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmapi(self.in_conv(x))


class _TeacherBase(nn.Module):
    """Common encoder construction and pi-scale diagnostic."""

    def _build_encoder(
        self, in_ch: int, width: int, sigma_pi: bool, dropout: float = 0.0
    ) -> nn.Module:
        self.sigma_pi = sigma_pi
        if sigma_pi:
            enc = _SigmaPiEncoder(in_ch, width, dropout=dropout)
            self._sigmapi: Optional[ConvSigmaPi2d] = enc.sigmapi
        else:
            enc = _VanillaEncoder(in_ch, width, dropout=dropout)
            self._sigmapi = None
        return enc

    def pi_scale_mean(self) -> Optional[float]:
        """``exp(pi_scale).mean()`` if this is a Sigma-Pi teacher, else ``None``."""
        return None if self._sigmapi is None else self._sigmapi.pi_scale_mean()


# ---------------------------------------------------------------------------
# Vector-head teacher (conv filter generation, Experiment 2)
# ---------------------------------------------------------------------------

class ConvFilterTeacher(_TeacherBase):
    """Generate a flat parameter vector, unpacked by a :class:`TargetSpec`.

    The encoder output is globally average-pooled to a single ``width``-vector,
    then a linear head emits ``spec.num_params`` values that ``spec.unpack``
    reshapes into the target's structured weights.

    Args:
        spec: target specification (e.g. ``Conv2dTargetSpec``) defining
            ``num_params`` and ``unpack``.
        proto_channels: channels of the input prototype.
        width: encoder width.
        sigma_pi: use a Sigma-Pi encoder (default False).
    """

    def __init__(
        self,
        spec: TargetSpec,
        proto_channels: int = 4,
        width: int = 64,
        sigma_pi: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.encoder = self._build_encoder(proto_channels, width, sigma_pi, dropout)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(width, spec.num_params)

    def forward(self, proto: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.pool(self.encoder(proto)).flatten(1)  # [1, width]
        flat = self.head(h).squeeze(0)  # [num_params]
        return self.spec.unpack(flat)


# ---------------------------------------------------------------------------
# Spatial-map teachers
# ---------------------------------------------------------------------------

class FCMapTeacher(_TeacherBase):
    """Generate a linear head ``{"weight": [K, F], "bias": [K]}`` (Experiment 1).

    The prototype's spatial dims are ``(num_classes, feature_dim)``; the weight
    head emits a single map of exactly that shape, and a small pooled branch
    emits the per-class bias.

    Args:
        num_classes: rows of the linear head (and prototype height).
        feature_dim: input width of the linear head (and prototype width).
        proto_channels: channels of the input prototype.
        width: encoder width.
        sigma_pi: use a Sigma-Pi encoder (default False).
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        proto_channels: int = 4,
        width: int = 64,
        sigma_pi: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.encoder = self._build_encoder(proto_channels, width, sigma_pi, dropout)
        self.weight_head = nn.Conv2d(width, 1, 1)
        self.bias_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((num_classes, 1)),
            nn.Conv2d(proto_channels, 1, 1),
        )

    def forward(self, proto: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(proto)
        weight = self.weight_head(h).squeeze(0).squeeze(0)  # [K, F]
        bias = self.bias_head(proto).squeeze(0).squeeze(0).squeeze(-1)  # [K]
        return {"weight": weight, "bias": bias}


class QKMapTeacher(_TeacherBase):
    """Generate per-layer query/key projections (Experiment 3).

    The prototype is a stack of ``D x D`` cross-moment matrices; the weight head
    emits ``2 * n_layers`` weight maps of shape ``D x D`` (a Wq and Wk per layer)
    and a pooled branch emits the matching biases. ``out_scale`` keeps the
    generated projections small at initialisation.

    Returns a list of ``n_layers`` dicts
    ``{"q_weight", "q_bias", "k_weight", "k_bias"}`` — the format consumed by
    :class:`~polyweave.targets.AttentionQKTargetSpec` and the transformer student.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        proto_channels: int = 4,
        width: int = 64,
        sigma_pi: bool = False,
        out_scale: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.encoder = self._build_encoder(proto_channels, width, sigma_pi, dropout)
        self.weight_head = nn.Conv2d(width, 2 * n_layers, 3, padding=1)
        self.bias_head = nn.Linear(width, 2 * n_layers * d_model)
        self.out_scale = nn.Parameter(torch.tensor(float(out_scale)))

    def forward(self, proto: torch.Tensor) -> List[LayerQK]:
        h = self.encoder(proto)
        wmaps = self.weight_head(h).squeeze(0) * self.out_scale  # [2L, D, D]
        bvec = self.bias_head(h.mean(dim=(-2, -1))).squeeze(0)  # [2L * D]
        return self._unpack(wmaps, bvec)

    def _unpack(self, wmaps: torch.Tensor, bvec: torch.Tensor) -> List[LayerQK]:
        D = self.d_model
        layers: List[LayerQK] = []
        bi = 0
        for l in range(self.n_layers):
            q_weight = wmaps[2 * l]
            k_weight = wmaps[2 * l + 1]
            q_bias = bvec[bi : bi + D]; bi += D
            k_bias = bvec[bi : bi + D]; bi += D
            layers.append(
                {"q_weight": q_weight, "q_bias": q_bias, "k_weight": k_weight, "k_bias": k_bias}
            )
        return layers
