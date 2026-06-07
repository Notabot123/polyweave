"""Smoke test for the random-recoveries-only mini driver.

Confirms the driver builds populations, runs random-only recovery (no teacher),
caches per seed, aggregates across seeds, and writes JSON + band PDFs — on tiny
CPU configs with fabricated CIFAR data. Numbers are not checked.
"""

from __future__ import annotations

import json

import torch

from polyweave.experiments import _common, cifar_conv1, cifar_fc, random_recovery


def _fake_cifar(num_classes=3, batches=4, batch_size=8):
    def loaders(*_a, **_k):
        def make():
            return [
                (torch.randn(batch_size, 3, 32, 32),
                 torch.randint(0, num_classes, (batch_size,)))
                for _ in range(batches)
            ]
        return make(), make()
    return loaders


def test_random_recovery_driver_runs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_common, "cifar10_loaders", _fake_cifar())

    tiny_fc = cifar_fc.Config(
        device="cpu", batch_size=8, num_architectures=2, warm_restarts=1,
        num_train_groups=1, base_student_epochs=1, warm_restart_epochs=1,
        feature_dim=16, num_classes=3, teacher_steps=1, teacher_width=4,
        log_every=0, eval_support_batches=1, eval_max_batches=1,
        recovery_steps=2, recovery_eval_every=1)
    tiny_c1 = cifar_conv1.Config(
        device="cpu", batch_size=8, num_architectures=2, warm_restarts=1,
        num_train_groups=1, base_student_epochs=1, warm_restart_epochs=1,
        feature_dim=16, num_classes=3, teacher_steps=1, teacher_width=4,
        log_every=0, eval_support_batches=1, eval_max_batches=1,
        bn_reset_batches=1, recovery_steps=2, recovery_eval_every=1)
    monkeypatch.setattr(cifar_fc, "Config", lambda: tiny_fc)
    monkeypatch.setattr(cifar_conv1, "Config", lambda: tiny_c1)

    random_recovery.main(["--seeds", "42", "43"])

    summary = json.loads((tmp_path / "plots" / "random_recovery_results.json").read_text())
    assert summary["seeds"] == [42, 43]
    assert set(summary["experiments"]) == {"fc", "conv1"}
    for key in ("fc", "conv1"):
        exp = summary["experiments"][key]
        assert set(exp["per_seed_final"]) == {"42", "43"}  # JSON keys are strings
        assert "final_mean" in exp and "final_std" in exp
    assert (tmp_path / "plots" / "polyweave_cifar_fc_random_recovery_fixed.pdf").exists()
    assert (tmp_path / "plots" / "polyweave_cifar_conv1_random_recovery_fixed.pdf").exists()
    # Per-seed caches enable resume.
    assert (tmp_path / "plots" / "raw_random" / "fc_seed42.json").exists()
