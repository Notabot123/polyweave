"""Single-seed Q/K diagnostic: metric A, metric B, pi_scale across three teachers.

Compares, on ONE shared student population:
    conv                      — additive baseline (no pi branch)
    sigmapi (center=False)    — genuine product, starts at identity (exp(u)~1)
    sigmapi (center=True)     — init-at-0 product (expm1(u)=0), branch starts SILENT

For each sigma-pi teacher we log, before and after teacher training:
    metric A : exponent_abs_mean()   = mean(|bounded exponent|)   (product SHAPE)
    metric B : branch_energy(h)      = pi_rms / (sigma_rms + pi_rms) on real protos
    pi_scale : exp(pi_scale).mean()  (the old "volume knob")
and the zero-shot seen/unseen accuracy.

Throwaway driver (not a permanent experiment); safe to delete.
"""

from __future__ import annotations

import torch

from polyweave.experiments import synthetic_attention as SA
from polyweave.experiments import _common
from polyweave.hypernets import QKMapTeacher
from polyweave.targets import AttentionQKTargetSpec
from polyweave.training import train_teacher
from polyweave.utils import count_params, set_seed


def _metric_B(teacher, proto):
    """pi_share on the activations flowing INTO the teacher's sigma-pi block."""
    if teacher._sigmapi is None:
        return None
    with torch.no_grad():
        h = teacher.encoder.in_conv(proto)
    return teacher._sigmapi.branch_energy(h)


def main():
    cfg = SA.Config()
    _common.configure_plots(cfg.dark_plots)
    set_seed(cfg.seed)
    print("=" * 70)
    print(f"Q/K diagnostic  seed={cfg.seed}  teacher_steps={cfg.teacher_steps}")
    print(f"Generated Q/K params: "
          f"{AttentionQKTargetSpec(cfg.d_model, cfg.n_layers).num_params:,}")
    print("=" * 70)

    # Shared student population (the expensive part — built ONCE).
    groups = SA._make_population(cfg)
    train_students, unseen_students = _common.split_seen_unseen(groups, cfg.num_train_groups)
    print(f"Seen students: {len(train_students)}  Unseen: {len(unseen_students)}")

    build = SA._proto_from_support(cfg)

    def sample_episode():
        relation = _common.sample_relation(cfg.vocab_size, cfg.device)
        return {"relation": relation, "support": SA._batch(cfg, relation),
                "query": SA._batch(cfg, relation)}

    # A representative prototype for metric B (built from a seen student).
    rel0 = _common.sample_relation(cfg.vocab_size, cfg.device)
    support0 = SA._batch(cfg, rel0)
    proto0 = build(train_students[0], support0)

    conditions = [
        ("conv", False, False),
        ("sigmapi_cp_false", True, False),
        ("sigmapi_cp_true", True, True),
    ]

    teachers = {}
    diag = {}
    for name, sigma_pi, center_product in conditions:
        set_seed(cfg.seed)  # same init draw order per teacher for fairness
        teacher = QKMapTeacher(
            cfg.d_model, cfg.n_layers, proto_channels=cfg.proto_channels,
            width=cfg.teacher_width, sigma_pi=sigma_pi, out_scale=cfg.out_scale,
            dropout=cfg.teacher_dropout, center_product=center_product,
        ).to(cfg.device)
        print(f"\n--- training {name} teacher ({count_params(teacher):,} params) "
              f"sigma_pi={sigma_pi} center_product={center_product} ---")

        A0 = teacher._sigmapi.exponent_abs_mean() if teacher._sigmapi else None
        B0 = _metric_B(teacher, proto0)
        ps0 = teacher.pi_scale_mean()

        result = train_teacher(
            teacher, train_students,
            sample_batch=sample_episode,
            build_prototype=lambda s, ep: build(s, ep["support"]),
            forward=lambda s, ep, gen: SA._eval_forward(s, ep["query"], gen),
            steps=cfg.teacher_steps, lr=cfg.teacher_lr,
            proto_noise_std=cfg.proto_noise_std, log_every=cfg.log_every,
        )
        teachers[name] = teacher

        A1 = teacher._sigmapi.exponent_abs_mean() if teacher._sigmapi else None
        B1 = _metric_B(teacher, proto0)
        ps1 = teacher.pi_scale_mean()
        diag[name] = dict(A0=A0, A1=A1, B0=B0, B1=B1, ps0=ps0, ps1=ps1)

        if A0 is not None:
            print(f"  metric A (mean|exp|): start={A0:.5f} final={A1:.5f} "
                  f"delta={A1 - A0:+.5f}")
            print(f"  metric B (pi_share) : start={B0['pi_share']:.4f} "
                  f"final={B1['pi_share']:.4f} "
                  f"(sigma_rms {B0['sigma_rms']:.3g}->{B1['sigma_rms']:.3g}, "
                  f"pi_rms {B0['pi_rms']:.3g}->{B1['pi_rms']:.3g})")
            print(f"  pi_scale (volume)   : start={ps0:.5f} final={ps1:.5f} "
                  f"delta={ps1 - ps0:+.5f}")

    methods = {"random": None, **teachers}
    print("\n=== zero-shot Q/K: seen ===")
    seen = SA._zero_shot(train_students, methods, cfg)
    print("\n=== zero-shot Q/K: unseen ===")
    unseen = SA._zero_shot(unseen_students, methods, cfg)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'method':<20} {'seen':>8} {'unseen':>8}  "
          f"{'A_final':>8} {'B_final':>8} {'pi_scale':>9}")
    for m in methods:
        d = diag.get(m)
        a = f"{d['A1']:.4f}" if d and d['A1'] is not None else "-"
        b = f"{d['B1']['pi_share']:.4f}" if d and d['B1'] is not None else "-"
        p = f"{d['ps1']:.4f}" if d and d['ps1'] is not None else "-"
        print(f"{m:<20} {seen[m]:>8.4f} {unseen[m]:>8.4f}  {a:>8} {b:>8} {p:>9}")
    print("\nDone.")


if __name__ == "__main__":
    main()
