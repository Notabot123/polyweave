"""Occlusion sensitivity for additive vs multiplicative (Sigma-Pi) features.

Occlusion sensitivity (Zeiler & Fergus, 2014) measures how a scalar response
changes when part of the input is replaced by a neutral *baseline* value. We use
it here as a mechanism probe: additive and multiplicative features leave
distinct, *quantifiable* fingerprints under occlusion.

Why this separates the two branches
------------------------------------
Write a response as a function of two disjoint input groups ``A`` and ``B`` and
let ``drop(g) = r(x) - r(x with g occluded)`` be how much response is lost when
group ``g`` is set to the baseline.

* **Additive** ``r = f(A) + g(B)``  ->  ``drop(A) = f(A) - f(baseline)`` is
  independent of ``B``; the joint drop equals the sum of the single drops, so the
  *interaction* ``drop(A&B) - drop(A) - drop(B)`` is ~0. Each factor contributes
  independently.
* **Multiplicative** ``r = f(A)*g(B)``  ->  occluding *either* factor (toward a
  baseline that sends it to ~0) collapses the whole response, so
  ``drop(A) ~= drop(B) ~= drop(A&B)`` and the interaction is strongly *negative*
  (sub-additive). This is the conjunctive **AND-signature**: every factor alone
  is sufficient to switch the feature off.

The :func:`conjunction_index` below normalises that interaction to ``[0, 1]``:
0 for a purely additive feature, 1 for a purely multiplicative (AND) one.

All functions are framework-light: a *response function* maps a batched input
tensor ``[N, ...]`` to a scalar response per item ``[N]`` (e.g. one output unit
of a layer, a logit, or a probability). Nothing here trains or mutates modules.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch

ResponseFn = Callable[[torch.Tensor], torch.Tensor]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_response(out: torch.Tensor) -> torch.Tensor:
    """Coerce a response-function output to a 1-D ``[N]`` tensor."""
    if out.ndim == 0:
        return out.reshape(1)
    return out.reshape(out.shape[0], -1).mean(dim=1) if out.ndim > 1 else out


@torch.no_grad()
def _occlude(x: torch.Tensor, index, baseline) -> torch.Tensor:
    """Return a copy of ``x`` with ``x[:, index]`` replaced by ``baseline``.

    ``index`` selects along the *feature* axis (everything after the batch dim is
    flattened to one feature axis first, then restored). ``baseline`` is a scalar
    or a tensor broadcastable to the occluded slice.
    """
    flat = x.reshape(x.shape[0], -1).clone()
    flat[:, index] = baseline
    return flat.reshape_as(x)


# ---------------------------------------------------------------------------
# 1-D occlusion sensitivity (per input feature)
# ---------------------------------------------------------------------------

@torch.no_grad()
def occlusion_sensitivity_1d(
    response_fn: ResponseFn,
    x: torch.Tensor,
    *,
    baseline: float | torch.Tensor = 0.0,
    window: int = 1,
    stride: int = 1,
) -> torch.Tensor:
    """Per-feature occlusion sensitivity map over the feature axis of ``x``.

    Args:
        response_fn: maps ``[N, ...]`` -> per-item response ``[N]`` (or a tensor
            reduced to ``[N]`` by mean over trailing dims).
        x: input batch ``[N, ...]``; dims after the batch axis are flattened to a
            single feature axis of length ``F``.
        baseline: value substituted into the occluded window (default ``0.0``).
        window: width of the occluding window in features.
        stride: step between successive window starts.

    Returns:
        ``[N, P]`` tensor of *drops* (``response(x) - response(x_occluded)``), one
        column per window position ``P``; large positive entries mark features the
        response most depends on.
    """
    base = _as_response(response_fn(x))  # [N]
    F = x.reshape(x.shape[0], -1).shape[1]
    starts = list(range(0, F - window + 1, stride))
    if starts and starts[-1] != F - window:
        starts.append(F - window)
    cols = []
    for s in starts:
        idx = slice(s, s + window)
        occ = _as_response(response_fn(_occlude(x, idx, baseline)))
        cols.append(base - occ)
    return torch.stack(cols, dim=1) if cols else base.new_zeros(x.shape[0], 0)


# ---------------------------------------------------------------------------
# 2-D occlusion sensitivity (spatial, for image / conv inputs)
# ---------------------------------------------------------------------------

@torch.no_grad()
def occlusion_sensitivity_2d(
    response_fn: ResponseFn,
    x: torch.Tensor,
    *,
    baseline: float | torch.Tensor = 0.0,
    window: int = 3,
    stride: int = 1,
    relative: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Spatial occlusion sensitivity for ``[N, C, H, W]`` inputs.

    A ``window x window`` patch (all channels) is occluded at each spatial stride
    position. Returns an ``[N, Hout, Wout]`` map of response drops, the classic
    Zeiler-Fergus heatmap.

    With ``relative=True`` each drop is divided by the un-occluded base response
    (``drop / (base + eps)``), giving the *fraction* of the response that depends
    on each region. This is what exposes the AND-signature visually: for a
    multiplicative detector every contributing region is ~100% critical (occluding
    it alone collapses the response), whereas for an additive detector each region
    is only partially critical.
    """
    if x.ndim != 4:
        raise ValueError(f"expected [N, C, H, W], got shape {tuple(x.shape)}")
    N, C, H, W = x.shape
    base = _as_response(response_fn(x))  # [N]
    ys = list(range(0, H - window + 1, stride)) or [0]
    xs = list(range(0, W - window + 1, stride)) or [0]
    out = base.new_zeros(N, len(ys), len(xs))
    for i, yy in enumerate(ys):
        for j, xx in enumerate(xs):
            occ = x.clone()
            occ[:, :, yy:yy + window, xx:xx + window] = baseline
            out[:, i, j] = base - _as_response(response_fn(occ))
    if relative:
        out = out / (base.reshape(N, 1, 1).abs() + eps)
    return out


# ---------------------------------------------------------------------------
# Interaction / AND-signature between two input groups
# ---------------------------------------------------------------------------

@torch.no_grad()
def group_drops(
    response_fn: ResponseFn,
    x: torch.Tensor,
    group_a: Sequence[int] | torch.Tensor,
    group_b: Sequence[int] | torch.Tensor,
    *,
    baseline: float | torch.Tensor = 0.0,
) -> dict:
    """Single- and joint-occlusion drops for two disjoint feature groups.

    Returns a dict with per-item tensors ``drop_a``, ``drop_b``, ``drop_ab`` and
    ``interaction = drop_ab - drop_a - drop_b`` (each ``[N]``). Indices select
    along the flattened feature axis.
    """
    a = torch.as_tensor(list(group_a) if not isinstance(group_a, torch.Tensor) else group_a, dtype=torch.long)
    b = torch.as_tensor(list(group_b) if not isinstance(group_b, torch.Tensor) else group_b, dtype=torch.long)
    base = _as_response(response_fn(x))
    drop_a = base - _as_response(response_fn(_occlude(x, a, baseline)))
    drop_b = base - _as_response(response_fn(_occlude(x, b, baseline)))
    ab = torch.cat([a, b])
    drop_ab = base - _as_response(response_fn(_occlude(x, ab, baseline)))
    return {
        "drop_a": drop_a,
        "drop_b": drop_b,
        "drop_ab": drop_ab,
        "interaction": drop_ab - drop_a - drop_b,
    }


@torch.no_grad()
def conjunction_index(
    response_fn: ResponseFn,
    x: torch.Tensor,
    group_a: Sequence[int] | torch.Tensor,
    group_b: Sequence[int] | torch.Tensor,
    *,
    baseline: float | torch.Tensor = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """AND-signature index in ``[0, 1]`` per item.

    ``index = (drop_a + drop_b - drop_ab) / (|drop_ab| + eps)``, clamped to
    ``[0, 1]``:

    * **~0** — additive feature (single drops add up to the joint drop, no
      interaction).
    * **~1** — multiplicative / conjunctive feature (either factor alone already
      collapses the response, so the single drops far exceed the joint drop).

    Returns ``[N]``. Aggregate with ``.mean()`` for a per-feature scalar.
    """
    d = group_drops(response_fn, x, group_a, group_b, baseline=baseline)
    num = d["drop_a"] + d["drop_b"] - d["drop_ab"]
    idx = num / (d["drop_ab"].abs() + eps)
    return idx.clamp(0.0, 1.0)
