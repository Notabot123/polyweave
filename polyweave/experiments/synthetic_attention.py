"""Experiment 3 — query/key projection generation on a relational-lookup task.

Attention-only transformer students solve a synthetic key-lookup task whose
*relation* changes every episode. A teacher generates the per-layer query/key
projections (value/output projections and the classifier stay the student's own,
frozen) from embedding-space cross-moment statistics of a support batch. The
query-key product is the explicitly bilinear (multiplicative) mapping where the
Sigma-Pi pi branch is expected to grow most — the high end of the
FC < conv1 < Q/K recruitment ordering.

Run:  python -m polyweave.experiments.synthetic_attention
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from ..evaluation import evaluate_accuracy, generate_averaged, mean_curves, recovery_curve
from ..hypernets import QKMapTeacher
from ..prototypes import relation_cross_moments
from ..students import (
    TinyTransformerStudent,
    attn_layers,
    freeze_except_qk,
    mask_qk_grads,
    qk_params,
    reinit_qk,
)
from ..targets import AttentionQKTargetSpec
from ..training import train_teacher
from ..utils import count_params, default_device, set_seed
from . import _common

LayerQK = Dict[str, torch.Tensor]


@dataclass
class Config:
    seed: int = 42
    emb_seed: int = 7
    device: str = default_device()

    # Synthetic task.
    vocab_size: int = 64
    seq_len: int = 10
    num_classes: int = 5          # K key slots → ~20% chance
    batch_size: int = 128

    # Transformer students (uniform shape so the Q/K target is fixed-size).
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    student_dropout: float = 0.0
    activations: Tuple[str, ...] = ("tanh", "relu", "swish")

    # Student population.
    num_architectures: int = 3
    num_train_groups: int = 2
    warm_restarts: int = 4
    student_base_steps: int = 2000
    warm_restart_steps: int = 2000
    student_lr: float = 1e-3

    # Teacher.
    proto_channels: int = 4
    teacher_width: int = 64
    teacher_steps: int = 5000
    teacher_lr: float = 1e-3
    teacher_dropout: float = 0.1  # matched to the CIFAR experiments
    out_scale: float = 0.1
    proto_noise_std: float = 0.0
    log_every: int = 250

    # Evaluation.
    eval_batches: int = 40
    eval_support_batches: int = 4
    eval_episodes: int = 3
    recovery_steps: int = 200
    recovery_lr: float = 1e-3
    recovery_eval_every: int = 20
    recovery_episodes: int = 2

    dark_plots: bool = False
    plot_prefix: str = "polyweave_synthetic_attention"


# ---------------------------------------------------------------------------
# Task helpers (bound to a Config)
# ---------------------------------------------------------------------------

def _batch(cfg: Config, relation: torch.Tensor):
    return _common.make_relational_batch(
        relation, batch_size=cfg.batch_size, vocab_size=cfg.vocab_size,
        num_key_slots=cfg.num_classes, seq_len=cfg.seq_len, device=cfg.device,
    )


def _proto_from_support(cfg: Config):
    def build(student, support):
        xs, ys = support
        e = student.embed(xs)
        return relation_cross_moments(e, ys, cfg.num_classes)
    return build


def _eval_forward(student, batch, gen):
    """``(logits, target)`` for installed (gen=None) or generated Q/K."""
    x, y = batch
    return student(x, gen_qk=gen), y


# ---------------------------------------------------------------------------
# Student population
# ---------------------------------------------------------------------------

def _make_population(cfg: Config) -> List[List[TinyTransformerStudent]]:
    identity = _common.identity_relation(cfg.vocab_size, cfg.device)

    def build_base(activation):
        return TinyTransformerStudent(
            vocab_size=cfg.vocab_size, seq_len=cfg.seq_len, num_classes=cfg.num_classes,
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
            activation=activation, emb_seed=cfg.emb_seed, dropout=cfg.student_dropout,
        ).to(cfg.device)

    def full_train(model):
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=cfg.student_lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.student_base_steps)
        for _ in range(cfg.student_base_steps):
            x, y = _batch(cfg, identity)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

    def finetune_target(model):
        freeze_except_qk(model)
        opt = torch.optim.Adam(qk_params(model), lr=cfg.student_lr)
        for _ in range(cfg.warm_restart_steps):
            x, y = _batch(cfg, identity)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            mask_qk_grads(model)
            torch.nn.utils.clip_grad_norm_(qk_params(model), 1.0)
            opt.step()

    return _common.build_student_groups(
        list(cfg.activations[: cfg.num_architectures]),
        build_base=build_base,
        full_train=full_train,
        freeze_trunk=freeze_except_qk,
        reinit_target=reinit_qk,
        finetune_target=finetune_target,
        warm_restarts=cfg.warm_restarts,
        seed=cfg.seed,
    )


# ---------------------------------------------------------------------------
# Zero-shot and recovery
# ---------------------------------------------------------------------------

def _random_qk(cfg: Config) -> List[LayerQK]:
    import torch.nn as nn
    D = cfg.d_model
    layers: List[LayerQK] = []
    for _ in range(cfg.n_layers):
        q_w = torch.empty(D, D, device=cfg.device)
        k_w = torch.empty(D, D, device=cfg.device)
        nn.init.xavier_uniform_(q_w)
        nn.init.xavier_uniform_(k_w)
        layers.append({
            "q_weight": q_w, "q_bias": torch.zeros(D, device=cfg.device),
            "k_weight": k_w, "k_bias": torch.zeros(D, device=cfg.device),
        })
    return layers


def _gen_for(method, teacher, student, relation, build, cfg: Config):
    if method == "random":
        return _random_qk(cfg)
    support = [_batch(cfg, relation) for _ in range(cfg.eval_support_batches)]
    return generate_averaged(teacher, student, support, build)


def _zero_shot(students, teachers: Dict, cfg: Config):
    build = _proto_from_support(cfg)
    scores = {m: [] for m in teachers}
    for i, student in enumerate(students):
        row = []
        for method, teacher in teachers.items():
            episode_accs = []
            for _ in range(cfg.eval_episodes):
                relation = _common.sample_relation(cfg.vocab_size, cfg.device)
                gen = _gen_for(method, teacher, student, relation, build, cfg)
                eval_batches = [_batch(cfg, relation) for _ in range(cfg.eval_batches)]
                episode_accs.append(evaluate_accuracy(student, eval_batches, gen, _eval_forward))
            acc = sum(episode_accs) / len(episode_accs)
            scores[method].append(acc)
            row.append(f"{method}={acc:.4f}")
        print(f"  student {i + 1:02d}: " + "  ".join(row))
    return {m: sum(v) / len(v) for m, v in scores.items()}


def _recovery(students, teachers: Dict, cfg: Config):
    build = _proto_from_support(cfg)
    spec = AttentionQKTargetSpec(cfg.d_model, cfg.n_layers)
    all_curves = {m: [] for m in teachers}
    for student in students:
        for method, teacher in teachers.items():
            for _ in range(cfg.recovery_episodes):
                relation = _common.sample_relation(cfg.vocab_size, cfg.device)
                gen = _gen_for(method, teacher, student, relation, build, cfg)
                model = copy.deepcopy(student)
                spec.install(attn_layers(model), gen)
                freeze_except_qk(model)

                eval_batches = [_batch(cfg, relation) for _ in range(cfg.eval_batches)]
                curve = recovery_curve(
                    model, init=qk_params,
                    sample_batch=lambda: _batch(cfg, relation),
                    forward=_eval_forward,
                    eval_fn=lambda m: evaluate_accuracy(m, eval_batches, None, _eval_forward),
                    steps=cfg.recovery_steps, lr=cfg.recovery_lr,
                    eval_every=cfg.recovery_eval_every, grad_mask=mask_qk_grads,
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
    print(f"Generated Q/K params: {AttentionQKTargetSpec(cfg.d_model, cfg.n_layers).num_params:,}")
    print("=" * 60)
    pi_start: float | None = None
    pi_final: float | None = None

    groups = _make_population(cfg)
    train_students, unseen_students = _common.split_seen_unseen(groups, cfg.num_train_groups)
    print(f"\nSeen students: {len(train_students)}  Unseen: {len(unseen_students)}")

    build = _proto_from_support(cfg)

    def sample_episode():
        relation = _common.sample_relation(cfg.vocab_size, cfg.device)
        return {"relation": relation, "support": _batch(cfg, relation),
                "query": _batch(cfg, relation)}

    teachers = {}
    losses = {}
    for kind, sigma_pi in (("conv", False), ("conv_sigmapi", True)):
        teacher = QKMapTeacher(
            cfg.d_model, cfg.n_layers, proto_channels=cfg.proto_channels,
            width=cfg.teacher_width, sigma_pi=sigma_pi, out_scale=cfg.out_scale,
            dropout=cfg.teacher_dropout,
        ).to(cfg.device)
        print(f"\n--- training {kind} teacher ({count_params(teacher):,} params) ---")
        result = train_teacher(
            teacher, train_students,
            sample_batch=sample_episode,
            build_prototype=lambda s, ep: build(s, ep["support"]),
            forward=lambda s, ep, gen: _eval_forward(s, ep["query"], gen),
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

    print("\n=== zero-shot Q/K: seen architectures ===")
    seen_means = _zero_shot(train_students, methods, cfg)
    print("\n=== zero-shot Q/K: unseen architectures ===")
    unseen_means = _zero_shot(unseen_students, methods, cfg)

    print("\nSummary zero-shot accuracy:")
    for m in methods:
        print(f"  {m:<14} seen={seen_means[m]:.4f}  unseen={unseen_means[m]:.4f}")

    print("\n=== recovery Q/K: unseen architectures ===")
    recovery = _recovery(unseen_students, methods, cfg)

    if make_plots:
        _common.plot_recovery_curves(
            recovery, name=f"{cfg.plot_prefix}_recovery",
            title="Q/K recovery after zero-shot init — unseen architectures",
        )
        _common.plot_lines(losses, title="Teacher training loss",
                           ylabel="cross-entropy", name=f"{cfg.plot_prefix}_teacher_loss")
        _common.plot_zeroshot_bar(seen_means, unseen_means, name=f"{cfg.plot_prefix}_zeroshot_bar")
    print("\nDone.")
    return _common.RunResult(
        seed=cfg.seed, label="Q/K", losses=losses,
        seen_means=seen_means, unseen_means=unseen_means, recovery=recovery,
        pi_start=pi_start, pi_final=pi_final,
    )


if __name__ == "__main__":
    run(Config())
