"""A generic teacher-training loop shared across all three experiments.

The loop is deliberately data-agnostic: the caller supplies three callbacks that
encapsulate everything experiment-specific, so the same optimiser/scheduler/
clipping/diagnostic machinery drives FC, conv-filter, and Q/K generation.

    sample_batch()                  -> batch          (any object)
    build_prototype(student, batch) -> proto          ([1, C, H, W] tensor)
    forward(student, batch, gen)    -> (logits, target)

Each step: pick a random student, build its prototype, optionally perturb it
with Gaussian noise, generate weights, run the student forward, and minimise the
cross-entropy against ``target``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Batch = Any
SampleBatch = Callable[[], Batch]
BuildPrototype = Callable[[nn.Module, Batch], torch.Tensor]
Forward = Callable[[nn.Module, Batch, Any], Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class TeacherTrainResult:
    """Outcome of :func:`train_teacher`."""

    losses: List[float] = field(default_factory=list)
    pi_scales: List[float] = field(default_factory=list)
    # Recruitment metric A (mean|exponent|, weights only) and metric B (pi output
    # share on the current proto). Empty for vanilla (non-Sigma-Pi) teachers.
    exponent_abs_means: List[float] = field(default_factory=list)
    pi_shares: List[float] = field(default_factory=list)

    @property
    def final_loss(self) -> Optional[float]:
        return self.losses[-1] if self.losses else None

    @property
    def final_pi_scale(self) -> Optional[float]:
        return self.pi_scales[-1] if self.pi_scales else None

    @property
    def final_exponent_abs_mean(self) -> Optional[float]:
        return self.exponent_abs_means[-1] if self.exponent_abs_means else None

    @property
    def final_pi_share(self) -> Optional[float]:
        return self.pi_shares[-1] if self.pi_shares else None


def train_teacher(
    teacher: nn.Module,
    students: Sequence[nn.Module],
    *,
    sample_batch: SampleBatch,
    build_prototype: BuildPrototype,
    forward: Forward,
    steps: int,
    lr: float = 1e-3,
    proto_noise_std: float = 0.0,
    grad_clip: Optional[float] = 1.0,
    cosine: bool = True,
    extra_params: Optional[Sequence[nn.Parameter]] = None,
    log_every: int = 0,
    log_fn: Callable[[str], None] = print,
) -> TeacherTrainResult:
    """Train ``teacher`` to generate weights for a population of ``students``.

    Args:
        teacher: the hypernetwork teacher (must expose ``pi_scale_mean()`` if you
            want the pi diagnostic recorded; it is ignored when it returns None).
        students: frozen students sampled uniformly each step.
        sample_batch: returns a fresh training batch.
        build_prototype: maps ``(student, batch)`` to a prototype tensor.
        forward: maps ``(student, batch, generated_weights)`` to
            ``(logits, target)`` for the cross-entropy loss.
        steps: number of optimisation steps.
        lr: Adam learning rate.
        proto_noise_std: std of optional Gaussian noise added to the prototype
            (a regulariser; the paper's matched regime uses 0).
        grad_clip: gradient-norm clip value (None to disable).
        cosine: use cosine LR annealing over ``steps`` (default True).
        extra_params: additional parameters to optimise jointly with the teacher
            (e.g. a :class:`LearnablePrototypeEncoder`'s parameters).
        log_every: print a progress line every ``log_every`` steps (0 = silent).
        log_fn: sink for log lines (default ``print``).

    Returns:
        A :class:`TeacherTrainResult` with per-step losses and pi-scale values.
    """
    for s in students:
        s.eval()
        for p in s.parameters():
            p.requires_grad_(False)

    params = list(teacher.parameters())
    if extra_params is not None:
        params += list(extra_params)
    opt = torch.optim.Adam(params, lr=lr)
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps) if cosine else None
    )

    pi_fn = getattr(teacher, "pi_scale_mean", None)
    exp_fn = getattr(teacher, "exponent_abs_mean", None)  # metric A (weights only)
    share_fn = getattr(teacher, "pi_share", None)         # metric B (needs proto)
    result = TeacherTrainResult()

    teacher.train()
    for step in range(1, steps + 1):
        batch = sample_batch()
        student = random.choice(list(students))
        proto = build_prototype(student, batch)
        if proto_noise_std > 0:
            proto = proto + proto_noise_std * torch.randn_like(proto)

        gen = teacher(proto)
        logits, target = forward(student, batch, gen)
        loss = F.cross_entropy(logits, target)

        opt.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        if sched is not None:
            sched.step()

        result.losses.append(loss.item())
        if pi_fn is not None:
            pv = pi_fn()
            if pv is not None:
                result.pi_scales.append(pv)
        if exp_fn is not None:
            ev = exp_fn()
            if ev is not None:
                result.exponent_abs_means.append(ev)
        if share_fn is not None:
            sv = share_fn(proto)
            if sv is not None:
                result.pi_shares.append(sv)

        if log_every and step % log_every == 0:
            acc = (logits.argmax(1) == target).float().mean().item()
            extra = ""
            if result.pi_scales:
                extra += f"  pi_scale={result.pi_scales[-1]:.5f}"
            if result.exponent_abs_means:
                extra += f"  A|exp|={result.exponent_abs_means[-1]:.5f}"
            if result.pi_shares:
                extra += f"  B_share={result.pi_shares[-1]:.4f}"
            log_fn(f"  step {step:5d}/{steps}: loss={loss.item():.4f}  acc={acc:.3f}{extra}")

    return result
