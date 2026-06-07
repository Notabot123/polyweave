"""Experiment 1 — linear-head (FC) generation on CIFAR-10.

A teacher generates the final classification head ``{"weight", "bias"}`` of a
frozen CNN trunk from class-conditional feature statistics. The optimal head is
(to first order) the nearest-class-centroid rule, so this is the *near-linear*
regime in which the Sigma-Pi pi branch is expected to stay dormant — the low end
of the paper's FC < conv1 < Q/K recruitment ordering.

This is a thin orchestration script over the PolyWeave library:

    student            CNNStudent            (polyweave.students)
    target             FCTargetSpec          (polyweave.targets)
    prototype          feature_class_stats   (polyweave.prototypes)
    teacher            FCMapTeacher          (polyweave.hypernets)
    training           train_teacher         (polyweave.training)
    zero-shot/recovery evaluation primitives (polyweave.evaluation)
    population/plots    _common               (this package)

Run:  python -m polyweave.experiments.cifar_fc
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..evaluation import (
    centroids_to_fc,
    class_centroids,
    evaluate_accuracy,
    generate_averaged,
    mean_curves,
    random_like,
    recovery_curve,
)
from ..hypernets import FCMapTeacher
from ..prototypes import feature_class_stats
from ..students import make_cnn_student
from ..training import train_teacher
from ..utils import count_params, default_device, set_seed
from . import _common

ARCHS = ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    seed: int = 42
    device: str = default_device()
    batch_size: int = 128

    # Student population (one architecture per group; seen groups train teacher).
    num_architectures: int = 3
    warm_restarts: int = 5
    num_train_groups: int = 2
    base_student_epochs: int = 15
    warm_restart_epochs: int = 5
    student_lr: float = 1e-3

    # Shared shapes.
    feature_dim: int = 256
    num_classes: int = 10
    in_ch: int = 3
    proto_channels: int = 4

    # Teacher.
    teacher_steps: int = 5000
    teacher_lr: float = 1e-3
    teacher_width: int = 64
    dropout: float = 0.1          # deliberate regulariser, matched across experiments
    proto_noise_std: float = 0.0
    log_every: int = 250

    # Evaluation.
    eval_support_batches: int = 5
    eval_max_batches: Optional[int] = None
    recovery_steps: int = 300
    recovery_lr: float = 1e-3
    recovery_eval_every: int = 20

    dark_plots: bool = False
    plot_prefix: str = "polyweave_cifar_fc"


# ---------------------------------------------------------------------------
# Callbacks (experiment-specific glue for the library)
# ---------------------------------------------------------------------------

def _forward(student, batch, gen):
    """``(logits, target)`` for a generated (or ``None``) FC head."""
    x, y = batch
    return student(x, generated_fc=gen), y


def _build_prototype(cfg: Config):
    def build(student, batch):
        x, y = batch
        feats = student.extract_features(x)
        return feature_class_stats(feats, y, cfg.num_classes)
    return build


def _make_population(cfg: Config, train_loader) -> List[List[nn.Module]]:
    """Full-train each arch, then warm-restart the FC head on a frozen trunk."""

    def build_base(arch):
        return make_cnn_student(
            arch, feature_dim=cfg.feature_dim, num_classes=cfg.num_classes,
            in_ch=cfg.in_ch,
        ).to(cfg.device)

    def full_train(model):
        opt = torch.optim.Adam(model.parameters(), lr=cfg.student_lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.base_student_epochs)
        _train_epochs(model, train_loader, opt, cfg.base_student_epochs, cfg, sched)

    def reinit_target(model):
        nn.init.kaiming_normal_(model.fc.weight, nonlinearity="linear")
        nn.init.zeros_(model.fc.bias)

    def finetune_target(model):
        opt = torch.optim.Adam(model.fc.parameters(), lr=cfg.student_lr)
        _train_epochs(model, train_loader, opt, cfg.warm_restart_epochs, cfg)

    return _common.build_student_groups(
        ARCHS[: cfg.num_architectures],
        build_base=build_base,
        full_train=full_train,
        freeze_trunk=lambda m: _common.freeze_except(m, "fc"),
        reinit_target=reinit_target,
        finetune_target=finetune_target,
        warm_restarts=cfg.warm_restarts,
        seed=cfg.seed,
    )


def _train_epochs(model, loader, opt, epochs, cfg, sched=None) -> None:
    import torch.nn.functional as F
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

@torch.no_grad()
def _ncc_head(student, support_batches, cfg: Config) -> Dict[str, torch.Tensor]:
    feats, ys = [], []
    for x, y in support_batches:
        feats.append(student.extract_features(x))
        ys.append(y)
    centroids = class_centroids(torch.cat(feats), torch.cat(ys), cfg.num_classes)
    return centroids_to_fc(centroids)


def _zero_shot(students, teachers: Dict, eval_batches, train_loader, cfg: Config):
    build = _build_prototype(cfg)
    scores = {m: [] for m in teachers}
    for i, student in enumerate(students):
        support = _common.collect_batches(train_loader, cfg.eval_support_batches, cfg.device)
        row = []
        for method, teacher in teachers.items():
            if method == "ncc":
                gen = _ncc_head(student, support, cfg)
            elif method == "random":
                gen = random_like(_ncc_head(student, support, cfg))
            else:
                gen = generate_averaged(teacher, student, support, build)
            acc = evaluate_accuracy(student, eval_batches, gen, _forward)
            scores[method].append(acc)
            row.append(f"{method}={acc:.4f}")
        print(f"  student {i + 1:02d}: " + "  ".join(row))
    return {m: sum(v) / len(v) for m, v in scores.items()}


def _recovery(students, teachers: Dict, eval_batches, train_loader, cfg: Config):
    build = _build_prototype(cfg)

    def init(model):
        return [model.fc.weight, model.fc.bias]

    all_curves = {m: [] for m in teachers}
    for student in students:
        support = _common.collect_batches(train_loader, cfg.eval_support_batches, cfg.device)
        for method, teacher in teachers.items():
            if method == "ncc":
                gen = _ncc_head(student, support, cfg)
            elif method == "random":
                gen = random_like(_ncc_head(student, support, cfg))
            else:
                gen = generate_averaged(teacher, student, support, build)

            import copy
            model = copy.deepcopy(student)
            with torch.no_grad():
                model.fc.weight.copy_(gen["weight"])
                model.fc.bias.copy_(gen["bias"])
            _common.freeze_except(model, "fc")  # make the head trainable for recovery

            def sample_batch():
                return _common.collect_batches(train_loader, 1, cfg.device)[0]

            curve = recovery_curve(
                model, init=init, sample_batch=sample_batch, forward=_forward,
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

    teachers = {}
    losses = {}
    for kind, sigma_pi in (("conv", False), ("conv_sigmapi", True)):
        teacher = FCMapTeacher(
            cfg.num_classes, cfg.feature_dim, proto_channels=cfg.proto_channels,
            width=cfg.teacher_width, sigma_pi=sigma_pi, dropout=cfg.dropout,
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

    methods = {"random": None, "ncc": None, **teachers}

    print("\n=== zero-shot: seen architectures ===")
    seen_means = _zero_shot(train_students, methods, eval_batches, train_loader, cfg)
    print("\n=== zero-shot: unseen architectures ===")
    unseen_means = _zero_shot(unseen_students, methods, eval_batches, train_loader, cfg)

    print("\nSummary zero-shot accuracy:")
    for m in methods:
        print(f"  {m:<14} seen={seen_means[m]:.4f}  unseen={unseen_means[m]:.4f}")

    print("\n=== recovery: unseen architectures ===")
    recovery = _recovery(unseen_students, methods, eval_batches, train_loader, cfg)

    if make_plots:
        _common.plot_recovery_curves(
            recovery, name=f"{cfg.plot_prefix}_recovery",
            title="FC recovery after zero-shot init — unseen architectures",
        )
        _common.plot_lines(losses, title="Teacher training loss",
                           ylabel="cross-entropy", name=f"{cfg.plot_prefix}_teacher_loss")
        _common.plot_zeroshot_bar(seen_means, unseen_means, name=f"{cfg.plot_prefix}_zeroshot_bar")
    print("\nDone.")
    return _common.RunResult(
        seed=cfg.seed, label="FC", losses=losses,
        seen_means=seen_means, unseen_means=unseen_means, recovery=recovery,
        pi_start=pi_start, pi_final=pi_final,
    )


if __name__ == "__main__":
    run(Config())
