"""Experiment 2 — first-conv-filter generation on CIFAR-10.

A teacher generates the first convolution's filters ``{"weight", "bias"}`` of a
frozen CNN from class-conditional statistics of the *raw input* over a spatial
grid. Unlike the FC head there is no nearest-centroid analogue: optimal edge/
texture filters encode multiplicative interactions (orientation x frequency x
phase), so the Sigma-Pi pi branch is expected to grow more than in Experiment 1
— the middle of the FC < conv1 < Q/K recruitment ordering.

After installing generated filters the first BatchNorm's running statistics are
stale, so they are re-estimated before recovery fine-tuning.

Run:  python -m polyweave.experiments.cifar_conv1
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..evaluation import (
    evaluate_accuracy,
    generate_averaged,
    mean_curves,
    random_like,
    recovery_curve,
)
from ..hypernets import ConvFilterTeacher
from ..prototypes import image_grid_stats
from ..students import make_cnn_student
from ..targets import Conv2dTargetSpec
from ..training import train_teacher
from ..utils import count_params, default_device, set_seed
from . import _common

ARCHS = ["A", "B", "C"]

CONV1_OUT = 32
CONV1_IN = 3
CONV1_KERNEL = 3


@dataclass
class Config:
    seed: int = 42
    device: str = default_device()
    batch_size: int = 128

    num_architectures: int = 3
    warm_restarts: int = 5
    num_train_groups: int = 2
    base_student_epochs: int = 15
    warm_restart_epochs: int = 5
    student_lr: float = 1e-3

    feature_dim: int = 256
    num_classes: int = 10
    proto_channels: int = 4
    proto_grid: int = 4

    teacher_steps: int = 5000
    teacher_lr: float = 1e-3
    teacher_width: int = 64
    dropout: float = 0.1
    proto_noise_std: float = 0.0
    log_every: int = 250

    eval_support_batches: int = 5
    eval_max_batches: Optional[int] = None
    bn_reset_batches: int = 10
    recovery_steps: int = 300
    recovery_lr: float = 1e-3
    recovery_eval_every: int = 20

    dark_plots: bool = False
    plot_prefix: str = "polyweave_cifar_conv1"

    # If set, save the trained teachers + unseen students (state_dicts + cfg) to
    # this directory so a later script can build vanilla-vs-Sigma-Pi student
    # ensembles without retraining. ``None`` disables saving.
    save_models_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def _forward(student, batch, gen):
    x, y = batch
    return student(x, gen_conv1=gen), y


def _build_prototype(cfg: Config):
    def build(student, batch):
        x, y = batch
        return image_grid_stats(x, y, cfg.num_classes, grid=cfg.proto_grid)
    return build


def _freeze_conv1(model) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.conv1.parameters():
        p.requires_grad_(True)
    for p in model.bn1.parameters():
        p.requires_grad_(True)


def _conv1_bn1_params(model) -> List[nn.Parameter]:
    return list(model.conv1.parameters()) + list(model.bn1.parameters())


@torch.no_grad()
def _reset_bn1(model, batches, cfg: Config) -> None:
    """Re-estimate *only* bn1's running stats (trunk BN stays as trained)."""
    was_training = model.training
    model.train()
    model.bn1.reset_running_stats()
    for i, (x, _y) in enumerate(batches):
        if i >= cfg.bn_reset_batches:
            break
        model(x)  # gen_conv1 None → uses the (just-installed) student.conv1
    if not was_training:
        model.eval()


def _make_population(cfg: Config, train_loader) -> List[List[nn.Module]]:
    def build_base(arch):
        return make_cnn_student(
            arch, feature_dim=cfg.feature_dim, num_classes=cfg.num_classes,
            in_ch=CONV1_IN, conv1_out=CONV1_OUT, kernel_size=CONV1_KERNEL,
        ).to(cfg.device)

    def full_train(model):
        opt = torch.optim.Adam(model.parameters(), lr=cfg.student_lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.base_student_epochs)
        _train_epochs(model, train_loader, opt, cfg.base_student_epochs, cfg, sched)

    def reinit_target(model):
        nn.init.kaiming_normal_(model.conv1.weight, nonlinearity="relu")
        nn.init.zeros_(model.conv1.bias)

    def finetune_target(model):
        opt = torch.optim.Adam(_conv1_bn1_params(model), lr=cfg.student_lr)
        _train_epochs(model, train_loader, opt, cfg.warm_restart_epochs, cfg)

    return _common.build_student_groups(
        ARCHS[: cfg.num_architectures],
        build_base=build_base,
        full_train=full_train,
        freeze_trunk=_freeze_conv1,
        reinit_target=reinit_target,
        finetune_target=finetune_target,
        warm_restarts=cfg.warm_restarts,
        seed=cfg.seed,
    )


def _train_epochs(model, loader, opt, epochs, cfg, sched=None) -> None:
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        if sched is not None:
            sched.step()


# ---------------------------------------------------------------------------
# Zero-shot and recovery
# ---------------------------------------------------------------------------

def _zero_shot(students, teachers: Dict, eval_batches, train_loader, cfg: Config, ref_gen):
    build = _build_prototype(cfg)
    scores = {m: [] for m in teachers}
    for i, student in enumerate(students):
        support = _common.collect_batches(train_loader, cfg.eval_support_batches, cfg.device)
        row = []
        for method, teacher in teachers.items():
            gen = random_like(ref_gen) if method == "random" \
                else generate_averaged(teacher, student, support, build)
            acc = evaluate_accuracy(student, eval_batches, gen, _forward)
            scores[method].append(acc)
            row.append(f"{method}={acc:.4f}")
        print(f"  student {i + 1:02d}: " + "  ".join(row))
    return {m: sum(v) / len(v) for m, v in scores.items()}


def _recovery(students, teachers: Dict, eval_batches, train_loader, cfg: Config, ref_gen):
    build = _build_prototype(cfg)
    all_curves = {m: [] for m in teachers}
    for student in students:
        support = _common.collect_batches(train_loader, cfg.eval_support_batches, cfg.device)
        for method, teacher in teachers.items():
            gen = random_like(ref_gen) if method == "random" \
                else generate_averaged(teacher, student, support, build)

            model = copy.deepcopy(student)
            with torch.no_grad():
                model.conv1.weight.copy_(gen["weight"])
                model.conv1.bias.copy_(gen["bias"])
            _reset_bn1(model, support, cfg)
            _freeze_conv1(model)

            curve = recovery_curve(
                model, init=_conv1_bn1_params,
                sample_batch=lambda: _common.collect_batches(train_loader, 1, cfg.device)[0],
                forward=_forward,
                eval_fn=lambda m: evaluate_accuracy(m, eval_batches, None, _forward),
                steps=cfg.recovery_steps, lr=cfg.recovery_lr,
                eval_every=cfg.recovery_eval_every,
            )
            all_curves[method].append(curve)
    return {m: mean_curves(cs) for m, cs in all_curves.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: Config, make_plots: bool = True) -> "_common.RunResult":
    _common.configure_plots(cfg.dark_plots)
    set_seed(cfg.seed)
    print("=" * 60)
    print(cfg)
    print("=" * 60)
    pi_start: float | None = None
    pi_final: float | None = None

    train_loader, test_loader = _common.cifar10_loaders(cfg.batch_size)
    groups = _make_population(cfg, train_loader)
    train_students, unseen_students = _common.split_seen_unseen(groups, cfg.num_train_groups)
    print(f"\nSeen students: {len(train_students)}  Unseen: {len(unseen_students)}")

    eval_batches = _common.collect_batches(
        test_loader, cfg.eval_max_batches or len(test_loader), cfg.device
    )

    spec = Conv2dTargetSpec(CONV1_OUT, CONV1_IN, CONV1_KERNEL)
    # A reference generated structure (shapes only) for the random baseline.
    ref_gen = {
        "weight": torch.zeros(CONV1_OUT, CONV1_IN, CONV1_KERNEL, CONV1_KERNEL, device=cfg.device),
        "bias": torch.zeros(CONV1_OUT, device=cfg.device),
    }

    teachers = {}
    losses = {}
    for kind, sigma_pi in (("conv", False), ("conv_sigmapi", True)):
        teacher = ConvFilterTeacher(
            spec, proto_channels=cfg.proto_channels, width=cfg.teacher_width,
            sigma_pi=sigma_pi, dropout=cfg.dropout,
        ).to(cfg.device)
        print(f"\n--- training {kind} teacher ({count_params(teacher):,} params) ---")
        result = train_teacher(
            teacher, train_students,
            sample_batch=lambda: _common.collect_batches(train_loader, 1, cfg.device)[0],
            build_prototype=_build_prototype(cfg), forward=_forward,
            steps=cfg.teacher_steps, lr=cfg.teacher_lr,
            proto_noise_std=cfg.proto_noise_std, log_every=cfg.log_every,
        )
        teachers[kind] = teacher
        losses[kind] = result.losses
        if result.final_pi_scale is not None:
            pi_start = result.pi_scales[0]
            pi_final = result.final_pi_scale
            print(f"  pi_scale: start={pi_start:.5f} "
                  f"final={pi_final:.5f} "
                  f"delta={pi_final - pi_start:+.5f}")

    methods = {"random": None, **teachers}

    print("\n=== zero-shot conv1: seen architectures ===")
    seen_means = _zero_shot(train_students, methods, eval_batches, train_loader, cfg, ref_gen)
    print("\n=== zero-shot conv1: unseen architectures ===")
    unseen_means = _zero_shot(unseen_students, methods, eval_batches, train_loader, cfg, ref_gen)

    print("\nSummary zero-shot accuracy:")
    for m in methods:
        print(f"  {m:<14} seen={seen_means[m]:.4f}  unseen={unseen_means[m]:.4f}")

    print("\n=== recovery conv1: unseen architectures ===")
    recovery = _recovery(unseen_students, methods, eval_batches, train_loader, cfg, ref_gen)

    if cfg.save_models_dir is not None:
        _save_models(cfg, teachers, train_students, unseen_students)

    if make_plots:
        _common.plot_recovery_curves(
            recovery, name=f"{cfg.plot_prefix}_recovery",
            title="Conv1 recovery after zero-shot init — unseen architectures",
        )
        _common.plot_lines(losses, title="Teacher training loss",
                           ylabel="cross-entropy", name=f"{cfg.plot_prefix}_teacher_loss")
        _common.plot_zeroshot_bar(seen_means, unseen_means, name=f"{cfg.plot_prefix}_zeroshot_bar")
    print("\nDone.")
    return _common.RunResult(
        seed=cfg.seed, label="conv1", losses=losses,
        seen_means=seen_means, unseen_means=unseen_means, recovery=recovery,
        pi_start=pi_start, pi_final=pi_final,
    )


def _save_models(cfg: Config, teachers: Dict, train_students, unseen_students) -> None:
    """Persist teachers + students so a later script can build student ensembles.

    Saves both teachers (additive ``conv`` and ``conv_sigmapi``) and every seen
    and unseen student as CPU ``state_dict``s, alongside the config and the arch
    shape constants needed to rebuild the modules. A future ensemble experiment
    can reload these, regenerate conv1 per student with each teacher, and compare
    additive vs Sigma-Pi student-ensemble disagreement / majority-vote accuracy.
    """
    import os
    out_dir = os.path.join(cfg.save_models_dir, f"seed{cfg.seed}")
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "cfg": cfg.__dict__,
        "arch": {"conv1_out": CONV1_OUT, "conv1_in": CONV1_IN, "kernel": CONV1_KERNEL,
                 "archs": ARCHS[: cfg.num_architectures], "num_train_groups": cfg.num_train_groups},
        "teachers": {k: {n: v.cpu() for n, v in t.state_dict().items()} for k, t in teachers.items()},
        "seen_students": [{n: v.cpu() for n, v in s.state_dict().items()} for s in train_students],
        "unseen_students": [{n: v.cpu() for n, v in s.state_dict().items()} for s in unseen_students],
    }
    path = os.path.join(out_dir, "conv1_models.pt")
    torch.save(payload, path)
    print(f"saved models for ensemble reuse -> {path}")


if __name__ == "__main__":
    run(Config())
