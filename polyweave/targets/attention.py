"""Attention query/key target spec — Experiment 3 (Q/K generation).

Generates the query and key projection weights for every self-attention layer of
a student. ``nn.MultiheadAttention`` packs Q/K/V into a single ``in_proj_weight``
of shape ``[3*D, D]`` (rows ``0:D`` = Q, ``D:2D`` = K, ``2D:3D`` = V) and a matching
``in_proj_bias`` of shape ``[3*D]``. This spec writes only the Q and K slices,
leaving V untouched — the bilinear query-key product is the explicitly
multiplicative mapping where the pi branch is most active.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .base import TargetSpec

LayerQK = Dict[str, torch.Tensor]  # q_weight[D,D], q_bias[D], k_weight[D,D], k_bias[D]
QKWeights = List[LayerQK]


class AttentionQKTargetSpec(TargetSpec):
    """Query/key projections for ``n_layers`` attention blocks of width ``d_model``."""

    def __init__(self, d_model: int, n_layers: int) -> None:
        self.d_model = d_model
        self.n_layers = n_layers
        self._per_layer = 2 * (d_model * d_model + d_model)  # Wq,bq,Wk,bk

    @property
    def num_params(self) -> int:
        return self.n_layers * self._per_layer

    def pack(self, weights: QKWeights) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        for layer in weights:
            parts.append(layer["q_weight"].reshape(-1))
            parts.append(layer["q_bias"].reshape(-1))
            parts.append(layer["k_weight"].reshape(-1))
            parts.append(layer["k_bias"].reshape(-1))
        return torch.cat(parts)

    def unpack(self, flat: torch.Tensor) -> QKWeights:
        flat = self._check_flat(flat)
        D = self.d_model
        ww = D * D
        layers: QKWeights = []
        i = 0
        for _ in range(self.n_layers):
            q_weight = flat[i : i + ww].reshape(D, D); i += ww
            q_bias = flat[i : i + D].reshape(D); i += D
            k_weight = flat[i : i + ww].reshape(D, D); i += ww
            k_bias = flat[i : i + D].reshape(D); i += D
            layers.append(
                {"q_weight": q_weight, "q_bias": q_bias, "k_weight": k_weight, "k_bias": k_bias}
            )
        return layers

    def install(self, into: List[nn.MultiheadAttention], weights: QKWeights) -> None:
        if len(into) != len(weights):
            raise ValueError(f"expected {len(weights)} attention modules, got {len(into)}")
        with torch.no_grad():
            for attn, g in zip(into, weights):
                D = attn.embed_dim
                attn.in_proj_weight[:D].copy_(g["q_weight"])
                attn.in_proj_bias[:D].copy_(g["q_bias"])
                attn.in_proj_weight[D : 2 * D].copy_(g["k_weight"])
                attn.in_proj_bias[D : 2 * D].copy_(g["k_bias"])

    @torch.no_grad()
    def extract(self, frm: List[nn.MultiheadAttention]) -> QKWeights:
        out: QKWeights = []
        for attn in frm:
            D = attn.embed_dim
            out.append(
                {
                    "q_weight": attn.in_proj_weight[:D].detach().clone(),
                    "q_bias": attn.in_proj_bias[:D].detach().clone(),
                    "k_weight": attn.in_proj_weight[D : 2 * D].detach().clone(),
                    "k_bias": attn.in_proj_bias[D : 2 * D].detach().clone(),
                }
            )
        return out
