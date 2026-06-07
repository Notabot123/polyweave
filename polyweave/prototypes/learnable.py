"""Learnable prototype encoder.

The paper uses fixed statistical prototypes to keep the parameter budget small
and avoid confounding the pi-branch measurement. For real-world few-shot use a
*learned* support representation is usually preferable: this module trains an
encoder that maps each support example into a small set of channels, then pools
per class to produce a teacher-ready prototype.

The output shape ``[1, out_channels, num_classes, embed_dim]`` matches
:func:`polyweave.prototypes.feature_class_stats`, so a teacher built for the
statistical prototype accepts a learned one unchanged (set ``out_channels`` equal
to the statistical ``proto_channels`` and ``embed_dim`` to the feature dim).

Unlike the statistical builders this is an ``nn.Module`` with trainable weights;
its parameters are typically optimised jointly with the teacher.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .statistical import normalize_prototype


class LearnablePrototypeEncoder(nn.Module):
    """Encode a support set into a learned prototype tensor.

    Each support example's feature vector is passed through a small MLP that
    emits ``out_channels * embed_dim`` values, reshaped to ``[out_channels,
    embed_dim]`` and mean-pooled within each class. Empty classes yield zeros.

    Args:
        in_dim: dimensionality of the input support features.
        num_classes: number of classes (rows of the prototype).
        out_channels: number of statistic channels to emit (match the teacher's
            ``proto_channels``; default 4).
        embed_dim: width of each channel's per-class vector. Defaults to
            ``in_dim`` so the prototype lines up with the statistical builder.
        hidden_dim: hidden width of the encoder MLP. Defaults to ``in_dim``.
        normalize: apply per-channel normalisation to the pooled prototype
            (default True), matching the statistical builders.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        out_channels: int = 4,
        embed_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.embed_dim = embed_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.normalize = normalize

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels * self.embed_dim),
        )

    def forward(self, feats: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Map support features to a prototype.

        Args:
            feats: support features ``[N, in_dim]``.
            y: integer labels ``[N]`` in ``[0, num_classes)``.

        Returns:
            Prototype ``[1, out_channels, num_classes, embed_dim]``.
        """
        N = feats.shape[0]
        enc = self.encoder(feats)  # [N, out_channels * embed_dim]
        enc = enc.view(N, self.out_channels, self.embed_dim)  # [N, C, E]

        # Class-conditional mean pool via scatter-add (keeps autograd intact).
        proto = feats.new_zeros(self.num_classes, self.out_channels, self.embed_dim)
        counts = feats.new_zeros(self.num_classes, 1, 1)
        idx = y.view(N, 1, 1).expand(N, self.out_channels, self.embed_dim)
        proto.scatter_add_(0, idx, enc)
        counts.scatter_add_(
            0, y.view(N, 1, 1), torch.ones(N, 1, 1, device=feats.device, dtype=feats.dtype)
        )
        proto = proto / counts.clamp(min=1.0)

        # [K, C, E] -> [1, C, K, E]
        proto = proto.permute(1, 0, 2).unsqueeze(0)
        return normalize_prototype(proto) if self.normalize else proto
