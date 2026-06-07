"""Evaluation loops shared across the three experiments.

These are the read-side counterpart to :func:`polyweave.training.train_teacher`:
having trained a teacher, we measure the weights it generates. They reuse the
same callback shape as the training loop — a ``forward(student, batch, gen)`` that
returns ``(logits, target)`` — so an experiment writes one closure and uses it for
training, zero-shot evaluation, and recovery alike.

Three concerns live here:

* **Zero-shot** — install/use generated weights *without* fine-tuning and measure
  accuracy (:func:`evaluate_accuracy`, with weights averaged over a few support
  batches by :func:`generate_averaged`).
* **Recovery** — install the generated weights as an *initialisation*, then
  fine-tune the target layer for a fixed budget and record the accuracy curve
  (:func:`recovery_curve`).
* **BatchNorm re-estimation** — after surgically replacing a conv layer, the
  running BN statistics are stale; :func:`reset_bn_stats` re-estimates them.

Everything is teacher/target-agnostic; experiment-specific glue (which params are
trainable, how weights install, any per-step gradient masking) is supplied via
small callbacks.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

Batch = Any
# A generated-weight structure: a dict of tensors (FC/conv) or a list of such
# dicts (per-layer Q/K). Averaging recurses over both.
WeightStruct = Union[dict, list]
Forward = Callable[[nn.Module, Batch, Any], Tuple[torch.Tensor, torch.Tensor]]


# ---------------------------------------------------------------------------
# Averaging generated weights over support batches
# ---------------------------------------------------------------------------

def average_weights(structures: Sequence[WeightStruct]) -> WeightStruct:
    """Element-wise mean of a list of identically-structured weight pytrees.

    Handles a ``dict`` of tensors and a ``list`` of such dicts (the two shapes
    the teachers emit). Raises on anything else.
    """
    if not structures:
        raise ValueError("need at least one structure to average")
    first = structures[0]
    if isinstance(first, dict):
        return {k: torch.stack([s[k] for s in structures], 0).mean(0) for k in first}
    if isinstance(first, list):
        return [average_weights([s[i] for s in structures]) for i in range(len(first))]
    raise TypeError(f"cannot average weight structure of type {type(first).__name__}")


@torch.no_grad()
def generate_averaged(
    teacher: nn.Module,
    student: nn.Module,
    support_batches: Iterable[Batch],
    build_prototype: Callable[[nn.Module, Batch], torch.Tensor],
) -> WeightStruct:
    """Generate weights for ``student`` from each support batch and average them.

    Averaging over a handful of support batches reduces the variance of the
    prototype statistics, matching the experiments' ``eval_support_batches``.
    """
    teacher.eval()
    structures = [teacher(build_prototype(student, b)) for b in support_batches]
    return average_weights(structures)


# ---------------------------------------------------------------------------
# Zero-shot accuracy
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_accuracy(
    student: nn.Module,
    eval_batches: Iterable[Batch],
    gen: Any,
    forward: Forward,
) -> float:
    """Top-1 accuracy of ``student`` over ``eval_batches`` using weights ``gen``.

    ``gen`` is passed straight to ``forward``; pass the generated structure for
    zero-shot evaluation, or ``None`` to evaluate the student's own (e.g. already
    installed / recovered) weights.
    """
    student.eval()
    correct = total = 0
    for batch in eval_batches:
        logits, target = forward(student, batch, gen)
        correct += (logits.argmax(1) == target).sum().item()
        total += target.numel()
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_macro_f1(
    student: nn.Module,
    eval_batches: Iterable[Batch],
    gen: Any,
    forward: Forward,
    num_classes: int,
) -> float:
    """Macro-averaged F1 of ``student`` over ``eval_batches`` using weights ``gen``.

    The balanced complement to :func:`evaluate_accuracy`: per-class F1 averaged
    with equal weight, so a method cannot score well by ignoring minority classes.
    On the (near-)balanced tasks in this paper macro-F1 tracks accuracy closely;
    it is provided so a re-run can report both at no extra forward-pass cost.
    Classes that are neither present nor predicted are excluded from the mean.
    """
    student.eval()
    tp = fp = fn = None
    for batch in eval_batches:
        logits, target = forward(student, batch, gen)
        if tp is None:
            tp = torch.zeros(num_classes, device=logits.device)
            fp = torch.zeros(num_classes, device=logits.device)
            fn = torch.zeros(num_classes, device=logits.device)
        pred = logits.argmax(1).reshape(-1)
        target = target.reshape(-1)
        for c in range(num_classes):
            pc = pred == c
            tc = target == c
            tp[c] += (pc & tc).sum()
            fp[c] += (pc & ~tc).sum()
            fn[c] += (~pc & tc).sum()
    if tp is None:
        return 0.0
    precision = tp / (tp + fp).clamp(min=1)
    recall = tp / (tp + fn).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    present = (tp + fn) > 0
    return f1[present].mean().item() if bool(present.any()) else 0.0


# ---------------------------------------------------------------------------
# BatchNorm re-estimation
# ---------------------------------------------------------------------------

@torch.no_grad()
def reset_bn_stats(
    model: nn.Module,
    batches: Iterable[Batch],
    run: Callable[[nn.Module, Batch], Any],
    *,
    max_batches: int = 10,
) -> None:
    """Re-estimate BatchNorm running statistics after a weight swap.

    Resets the running mean/var of every ``_BatchNorm`` in ``model`` and re-runs
    ``run(model, batch)`` (a forward pass) in train mode over up to
    ``max_batches`` batches. Restores ``eval`` mode afterwards.
    """
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.reset_running_stats()
    model.train()
    for i, batch in enumerate(batches):
        if i >= max_batches:
            break
        run(model, batch)
    model.eval()


# ---------------------------------------------------------------------------
# Recovery curve
# ---------------------------------------------------------------------------

RecoveryInit = Callable[[nn.Module], Sequence[nn.Parameter]]


def recovery_curve(
    student: nn.Module,
    *,
    init: RecoveryInit,
    sample_batch: Callable[[], Batch],
    forward: Forward,
    eval_fn: Callable[[nn.Module], float],
    steps: int,
    lr: float = 1e-3,
    eval_every: int = 20,
    grad_mask: Optional[Callable[[nn.Module], None]] = None,
) -> List[Tuple[int, float]]:
    """Fine-tune a generated initialisation and record the accuracy curve.

    Args:
        student: the student to recover (modified in place).
        init: installs the initial weights into ``student`` and returns the list
            of parameters to optimise (e.g. just the target layer). Any BN reset
            belongs here too.
        sample_batch: fresh training batch each step.
        forward: ``(student, batch, None) -> (logits, target)`` — called with
            ``gen=None`` so the student uses its own (installed) weights.
        eval_fn: maps ``student`` to a scalar accuracy (evaluated at step 0 and
            every ``eval_every`` steps, and always at the final step).
        steps: fine-tuning budget.
        lr: Adam learning rate.
        eval_every: evaluation cadence in steps.
        grad_mask: optional hook run after ``backward`` and before ``opt.step``
            to zero gradients on parameters that must stay frozen (e.g. the value
            projection when only Q/K is being recovered).

    Returns:
        A list of ``(step, accuracy)`` pairs, starting at ``(0, zero-shot acc)``.
    """
    params = list(init(student))
    opt = torch.optim.Adam(params, lr=lr)

    curve: List[Tuple[int, float]] = [(0, eval_fn(student))]
    for step in range(1, steps + 1):
        student.train()
        batch = sample_batch()
        logits, target = forward(student, batch, None)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        if grad_mask is not None:
            grad_mask(student)
        opt.step()

        if step % eval_every == 0 or step == steps:
            curve.append((step, eval_fn(student)))
    return curve


def mean_curves(curves: Sequence[Sequence[Tuple[int, float]]]) -> List[Tuple[int, float]]:
    """Average several recovery curves that share the same step grid."""
    if not curves:
        raise ValueError("no curves to average")
    steps = [s for s, _ in curves[0]]
    n = len(curves)
    means = [sum(c[i][1] for c in curves) / n for i in range(len(steps))]
    return list(zip(steps, means))
