"""Parameter-free statistical prototype builders (the paper's prototypes).

Each builder produces a ``[1, channels, H, W]`` tensor and applies per-channel
normalisation across the spatial dims so the teacher sees scale-invariant
statistics. These are deliberately hand-crafted and confound-free; for a learned
alternative see :class:`polyweave.prototypes.LearnablePrototypeEncoder`.
"""

from __future__ import annotations

import torch

EPS = 1e-6
KURT_EPS = 1e-8


def normalize_prototype(proto: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Per-channel standardisation across the last two (spatial) dims.

    Args:
        proto: tensor shaped ``[B, C, H, W]``.
        eps: stabiliser added to the per-channel std.
    """
    mean = proto.mean(dim=(-2, -1), keepdim=True)
    std = proto.std(dim=(-2, -1), keepdim=True, unbiased=False)
    return (proto - mean) / (std + eps)


def _class_moments(
    feats: torch.Tensor, y: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Per-class mean/variance/kurtosis stacked as ``[3, num_classes, F]``.

    ``feats`` is ``[N, F]``; empty classes yield zero rows.
    """
    F_dim = feats.shape[1]
    device, dtype = feats.device, feats.dtype
    means, vars_, kurts = [], [], []
    for c in range(num_classes):
        mask = y == c
        if mask.any():
            fc = feats[mask]
            mu = fc.mean(0)
            var = fc.var(0, unbiased=False).clamp(min=KURT_EPS)
            diff = fc - mu.unsqueeze(0)
            kurt = (diff ** 4).mean(0) / (var ** 2) - 3.0
        else:
            mu = torch.zeros(F_dim, device=device, dtype=dtype)
            var = torch.zeros(F_dim, device=device, dtype=dtype)
            kurt = torch.zeros(F_dim, device=device, dtype=dtype)
        means.append(mu)
        vars_.append(var)
        kurts.append(kurt)
    return torch.stack([torch.stack(means), torch.stack(vars_), torch.stack(kurts)], dim=0)


@torch.no_grad()
def feature_class_stats(
    feats: torch.Tensor, y: torch.Tensor, num_classes: int, normalize: bool = True
) -> torch.Tensor:
    """Per-class statistics of *student features* (Experiment 1).

    Channels: 0=mean, 1=variance, 2=excess kurtosis, 3=inter-class contrast
    (per-class absolute deviation from the global feature mean).

    Args:
        feats: student features ``[N, feature_dim]``.
        y: integer labels ``[N]``.
        num_classes: number of classes (rows in the prototype).
        normalize: apply per-channel normalisation (default True).

    Returns:
        Prototype ``[1, 4, num_classes, feature_dim]``.
    """
    mvk = _class_moments(feats, y, num_classes)  # [3, K, F]
    means = mvk[0]
    global_mean = feats.mean(0)
    contrast = (means - global_mean.unsqueeze(0)).abs()  # [K, F]
    proto = torch.cat([mvk, contrast.unsqueeze(0)], dim=0).unsqueeze(0)  # [1, 4, K, F]
    return normalize_prototype(proto) if normalize else proto


@torch.no_grad()
def image_grid_stats(
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    grid: int = 4,
    normalize: bool = True,
) -> torch.Tensor:
    """Class-conditional statistics over a spatial grid of *raw inputs* (Exp 2).

    The image is split into a ``grid x grid`` array of cells; per cell and channel
    we take the mean intensity, giving a ``grid^2 * in_ch`` feature vector per
    example. Channels: 0=mean, 1=variance, 2=excess kurtosis, 3=inter-class
    contrast (mean absolute deviation from the other classes' means).

    Args:
        x: input images ``[N, C, H, W]`` (H, W divisible by ``grid``).
        y: integer labels ``[N]``.
        num_classes: number of classes.
        grid: cells per spatial axis.
        normalize: apply per-channel normalisation (default True).

    Returns:
        Prototype ``[1, 4, num_classes, grid^2 * in_ch]``.
    """
    N, C, H, W = x.shape
    gh, gw = H // grid, W // grid
    cells = []
    for gi in range(grid):
        for gj in range(grid):
            patch = x[:, :, gi * gh : (gi + 1) * gh, gj * gw : (gj + 1) * gw]
            cells.append(patch.mean(dim=(-2, -1)))  # [N, C]
    cell_feats = torch.cat(cells, dim=1)  # [N, grid^2 * C]

    mvk = _class_moments(cell_feats, y, num_classes)  # [3, K, F]
    means = mvk[0]
    contrast = []
    for c in range(num_classes):
        others = torch.cat([means[:c], means[c + 1 :]], dim=0)
        if others.numel():
            contrast.append((others - means[c]).abs().mean(0))
        else:
            contrast.append(torch.zeros_like(means[c]))
    contrast_t = torch.stack(contrast)  # [K, F]
    proto = torch.cat([mvk, contrast_t.unsqueeze(0)], dim=0).unsqueeze(0)
    return normalize_prototype(proto) if normalize else proto


@torch.no_grad()
def relation_cross_moments(
    embeddings: torch.Tensor,
    y: torch.Tensor,
    num_key_slots: int,
    normalize: bool = True,
) -> torch.Tensor:
    """Embedding-space query/key cross-moments for the relational task (Exp 3).

    Channels (each a ``D x D`` second-moment matrix over the support batch):
        0. R_qk   = E[ e_query  (x) e_matched_key ]   — the relation signal
        1. C_qq   = E[ e_query  (x) e_query ]          — query geometry
        2. C_kk   = E[ e_key    (x) e_key ]            — key geometry (all slots)
        3. R_qctx = E[ e_query  (x) mean_key ]         — query vs. mean-key context

    Args:
        embeddings: token embeddings ``[B, L, D]`` (query at last position).
        y: matched-key slot index ``[B]`` in ``[0, num_key_slots)``.
        num_key_slots: number of key slots ``K`` at the front of the sequence.
        normalize: apply per-channel normalisation (default True).

    Returns:
        Prototype ``[1, 4, D, D]``.
    """
    B, L, D = embeddings.shape
    K = num_key_slots
    eq = embeddings[:, -1, :]  # [B, D]
    idx = y.view(B, 1, 1).expand(B, 1, D)
    emk = embeddings.gather(1, idx).squeeze(1)  # [B, D]
    ekeys = embeddings[:, :K, :]  # [B, K, D]
    mean_key = ekeys.mean(1)  # [B, D]

    R_qk = torch.einsum("bd,be->de", eq, emk) / B
    C_qq = torch.einsum("bd,be->de", eq, eq) / B
    ek_flat = ekeys.reshape(-1, D)
    C_kk = torch.einsum("nd,ne->de", ek_flat, ek_flat) / ek_flat.shape[0]
    R_qctx = torch.einsum("bd,be->de", eq, mean_key) / B

    proto = torch.stack([R_qk, C_qq, C_kk, R_qctx], dim=0).unsqueeze(0)  # [1, 4, D, D]
    return normalize_prototype(proto) if normalize else proto
