"""Non-hypernetwork baselines the teachers are measured against.

* **NCC (nearest-class-centroid)** — the parameter-free reference for FC head
  generation (Experiment 1). A linear classifier whose weights are the class
  feature centroids is exactly a nearest-centroid classifier under squared
  Euclidean distance, so it slots straight into the same ``{"weight","bias"}``
  generated-FC interface.
* **Random initialisation** — the floor every generated initialisation must beat.
  :func:`random_like` mints random weights matching the shape of any generated
  structure, so the same helper serves FC, conv, and Q/K.
"""

from __future__ import annotations

from typing import Dict, List, Union

import torch

WeightStruct = Union[Dict[str, torch.Tensor], List]


# ---------------------------------------------------------------------------
# Nearest-class-centroid
# ---------------------------------------------------------------------------

def class_centroids(feats: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Mean feature vector per class, ``[num_classes, feature_dim]``.

    Empty classes get a zero centroid (they contribute nothing discriminative).
    """
    F = feats.shape[1]
    centroids = torch.zeros(num_classes, F, device=feats.device, dtype=feats.dtype)
    counts = torch.zeros(num_classes, device=feats.device)
    centroids.index_add_(0, y, feats)
    counts.index_add_(0, y, torch.ones_like(y, dtype=feats.dtype))
    nonempty = counts > 0
    centroids[nonempty] /= counts[nonempty].unsqueeze(1)
    return centroids


def centroids_to_fc(centroids: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Turn class centroids into an equivalent linear head ``{"weight","bias"}``.

    For centroid ``c_k``, the score ``-½‖x - c_k‖²`` ranks classes identically to
    nearest-centroid; dropping the shared ``-½‖x‖²`` term leaves the linear form
    ``wₖ·x + bₖ`` with ``wₖ = c_k`` and ``bₖ = -½‖c_k‖²``.
    """
    weight = centroids.clone()
    bias = -0.5 * (centroids ** 2).sum(dim=1)
    return {"weight": weight, "bias": bias}


def ncc_fc(feats: torch.Tensor, y: torch.Tensor, num_classes: int) -> Dict[str, torch.Tensor]:
    """Convenience: build the NCC linear head directly from features and labels."""
    return centroids_to_fc(class_centroids(feats, y, num_classes))


# ---------------------------------------------------------------------------
# Random initialisation
# ---------------------------------------------------------------------------

def random_like(
    reference: WeightStruct,
    *,
    scale: float = 1.0,
    zero_bias: bool = True,
    fan_in_scale: bool = True,
) -> WeightStruct:
    """Random weights matching the shapes of a generated structure.

    Weight-like tensors (``ndim >= 2``) are filled with random normal values; 1-D
    tensors are treated as biases and zeroed when ``zero_bias`` (the experiments'
    random baseline uses random weights with zero bias). Recurses over a list of
    dicts for the per-layer Q/K structure.

    By default (``fan_in_scale=True``) each weight tensor is scaled to a
    Kaiming-linear standard deviation ``1/sqrt(fan_in)`` — matching
    ``nn.init.kaiming_normal_(..., nonlinearity="linear")`` — so the random
    baseline is a *well-conditioned* initialisation rather than a saturating one.
    ``fan_in`` is ``prod(shape[1:])`` (input features, times kernel area for
    convolutions). ``scale`` multiplies on top of this. Passing
    ``fan_in_scale=False`` recovers the raw ``scale * randn`` behaviour.

    Note: a unit-variance random head (``fan_in_scale=False, scale=1``) produces
    logits ~``sqrt(fan_in)`` too large, saturating the softmax and stalling
    fine-tuning — which is why fan-in scaling is the default for the baseline.
    """
    if isinstance(reference, dict):
        out: Dict[str, torch.Tensor] = {}
        for k, v in reference.items():
            if v.ndim >= 2:
                std = scale
                if fan_in_scale:
                    fan_in = int(v[0].numel())  # prod(shape[1:])
                    std = scale / (fan_in ** 0.5)
                out[k] = std * torch.randn_like(v)
            else:
                out[k] = torch.zeros_like(v) if zero_bias else scale * torch.randn_like(v)
        return out
    if isinstance(reference, list):
        return [
            random_like(r, scale=scale, zero_bias=zero_bias, fan_in_scale=fan_in_scale)
            for r in reference
        ]
    raise TypeError(f"cannot build random_like for type {type(reference).__name__}")
