"""End-to-end smoke tests for the three experiment scripts.

These run each experiment's full ``run(cfg)`` pipeline — population build, teacher
training (vanilla + Sigma-Pi), zero-shot, recovery, and plotting — on
deliberately tiny CPU configs. They assert the pipeline executes and produces
plots; they do NOT check accuracy (that needs the full GPU regime). The CIFAR
loader is monkeypatched with a fabricated tensor dataset so no download happens.
"""

from __future__ import annotations

import torch

from polyweave.experiments import _common, cifar_conv1, cifar_fc, synthetic_attention


def _fake_cifar(num_classes: int, batches: int = 4, batch_size: int = 8):
    def loaders(*_args, **_kwargs):
        def make():
            return [
                (torch.randn(batch_size, 3, 32, 32), torch.randint(0, num_classes, (batch_size,)))
                for _ in range(batches)
            ]
        return make(), make()
    return loaders


def test_cifar_fc_runs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_common, "cifar10_loaders", _fake_cifar(num_classes=3))
    cfg = cifar_fc.Config(
        device="cpu", batch_size=8, num_architectures=2, warm_restarts=1,
        num_train_groups=1, base_student_epochs=1, warm_restart_epochs=1,
        feature_dim=16, num_classes=3, teacher_steps=2, teacher_width=4,
        log_every=0, eval_support_batches=1, eval_max_batches=1,
        recovery_steps=2, recovery_eval_every=1,
    )
    cifar_fc.run(cfg)
    assert (tmp_path / "plots" / f"{cfg.plot_prefix}_recovery.pdf").exists()


def test_cifar_conv1_runs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_common, "cifar10_loaders", _fake_cifar(num_classes=3))
    cfg = cifar_conv1.Config(
        device="cpu", batch_size=8, num_architectures=2, warm_restarts=1,
        num_train_groups=1, base_student_epochs=1, warm_restart_epochs=1,
        feature_dim=16, num_classes=3, teacher_steps=2, teacher_width=4,
        log_every=0, eval_support_batches=1, eval_max_batches=1,
        bn_reset_batches=1, recovery_steps=2, recovery_eval_every=1,
    )
    cifar_conv1.run(cfg)
    assert (tmp_path / "plots" / f"{cfg.plot_prefix}_recovery.pdf").exists()


def test_synthetic_attention_runs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg = synthetic_attention.Config(
        device="cpu", vocab_size=16, seq_len=6, num_classes=3, batch_size=8,
        d_model=16, n_heads=2, n_layers=1, num_architectures=2, num_train_groups=1,
        warm_restarts=1, student_base_steps=3, warm_restart_steps=3,
        teacher_steps=2, teacher_width=4, log_every=0,
        eval_batches=2, eval_support_batches=1, eval_episodes=1,
        recovery_steps=2, recovery_eval_every=1, recovery_episodes=1,
    )
    synthetic_attention.run(cfg)
    assert (tmp_path / "plots" / f"{cfg.plot_prefix}_recovery.pdf").exists()
