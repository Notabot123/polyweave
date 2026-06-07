"""Unit tests for the multi-seed aggregation + paper-plot helpers.

These use fabricated ``RunResult``s (no training) so they are fast; they check
that the mean/std math is correct and that every aggregate figure is written as
a PDF. The full experiment pipelines are covered by ``test_experiments_smoke``.
"""

from __future__ import annotations

import math

import json

from polyweave.experiments import _common, multiseed


def _fake_result(seed, label, pi_start, pi_final, recov, seen, unseen):
    return _common.RunResult(
        seed=seed, label=label,
        losses={"conv": [1.0, 0.5], "conv_sigmapi": [1.0, 0.4]},
        seen_means=seen, unseen_means=unseen, recovery=recov,
        pi_start=pi_start, pi_final=pi_final,
    )


def _curve(vals):
    return [(i * 10, v) for i, v in enumerate(vals)]


def _results():
    methods_seen = [{"random": 0.1, "conv": 0.5, "conv_sigmapi": 0.4},
                    {"random": 0.2, "conv": 0.6, "conv_sigmapi": 0.5}]
    methods_unseen = [{"random": 0.1, "conv": 0.3, "conv_sigmapi": 0.25},
                      {"random": 0.1, "conv": 0.4, "conv_sigmapi": 0.35}]
    recov = [
        {"random": _curve([0.1, 0.2, 0.3]), "conv": _curve([0.4, 0.6, 0.8])},
        {"random": _curve([0.2, 0.3, 0.4]), "conv": _curve([0.5, 0.7, 0.9])},
    ]
    return [
        _fake_result(42, "conv1", 0.1353, 0.16, recov[0], methods_seen[0], methods_unseen[0]),
        _fake_result(43, "conv1", 0.1353, 0.18, recov[1], methods_seen[1], methods_unseen[1]),
    ]


def test_pi_delta_property_and_aggregate():
    results = _results()
    assert math.isclose(results[0].pi_delta, 0.16 - 0.1353, rel_tol=1e-9)
    mean, std = _common.aggregate_pi_delta(results)
    deltas = [0.16 - 0.1353, 0.18 - 0.1353]
    exp_mean = sum(deltas) / 2
    exp_std = math.sqrt(sum((d - exp_mean) ** 2 for d in deltas) / 1)  # sample std, n-1
    assert math.isclose(mean, exp_mean, rel_tol=1e-9)
    assert math.isclose(std, exp_std, rel_tol=1e-9)


def test_pi_delta_none_when_no_sigmapi():
    r = _common.RunResult(seed=1, label="x")
    assert r.pi_delta is None


def test_aggregate_zeroshot_mean_std():
    seen, unseen = _common.aggregate_zeroshot(_results())
    assert math.isclose(seen["conv"][0], 0.55, rel_tol=1e-9)        # (0.5+0.6)/2
    assert math.isclose(unseen["random"][0], 0.10, rel_tol=1e-9)
    assert math.isclose(unseen["random"][1], 0.0, abs_tol=1e-12)    # identical -> std 0


def test_aggregate_recovery_alignment():
    bands = _common.aggregate_recovery(_results())
    steps, mean, std = bands["random"]
    assert steps == [0, 10, 20]
    assert math.isclose(mean[0], 0.15, rel_tol=1e-9)   # (0.1+0.2)/2
    assert math.isclose(mean[2], 0.35, rel_tol=1e-9)   # (0.3+0.4)/2


def test_aggregate_plots_written(tmp_path):
    results = _results()
    bands = _common.aggregate_recovery(results)
    seen, unseen = _common.aggregate_zeroshot(results)
    pi = {"FC": (0.009, 0.001), "conv1": (0.022, 0.002), "Q/K": (0.025, 0.003)}

    _common.plot_recovery_band(bands, name="t_recovery", title="t", plots_dir=tmp_path)
    _common.plot_zeroshot_grouped_std(seen, unseen, name="t_zeroshot", plots_dir=tmp_path)
    _common.plot_pi_ordering(pi, name="t_ordering", plots_dir=tmp_path)

    for stem in ("t_recovery", "t_zeroshot", "t_ordering"):
        assert (tmp_path / f"{stem}.pdf").exists()
        assert (tmp_path / f"{stem}.png").exists()


def test_run_result_to_dict_roundtrips_curve_types():
    r = _results()[0]
    d = r.to_dict()
    assert d["seed"] == 42 and d["label"] == "conv1"
    # curves serialised as [[int, float], ...]
    assert d["recovery"]["random"][0] == [0, 0.1]
    assert math.isclose(d["pi_delta"], 0.16 - 0.1353, rel_tol=1e-9)


def test_driver_orchestration_writes_outputs(monkeypatch, tmp_path):
    """main() wires run->aggregate->plot->JSON without running real training."""
    monkeypatch.chdir(tmp_path)

    # Fabricate fast per-(experiment, seed) results keyed by the label run() returns.
    deltas = {"FC": (0.135, 0.145), "conv1": (0.135, 0.157), "Q/K": (0.135, 0.160)}

    def fake_run_factory(label):
        def fake_run(cfg, make_plots=True):
            d = deltas[label]
            # vary final pi a touch per seed so std is nonzero
            jitter = (cfg.seed % 10) * 1e-4
            return _common.RunResult(
                seed=cfg.seed, label=label,
                losses={"conv": [1.0], "conv_sigmapi": [0.9]},
                seen_means={"random": 0.1, "conv": 0.5, "conv_sigmapi": 0.45},
                unseen_means={"random": 0.1, "conv": 0.3, "conv_sigmapi": 0.28},
                recovery={"random": _curve([0.1, 0.2]), "conv": _curve([0.4, 0.6])},
                pi_start=d[0], pi_final=d[1] + jitter,
            )
        return fake_run

    monkeypatch.setattr(multiseed.cifar_fc, "run", fake_run_factory("FC"))
    monkeypatch.setattr(multiseed.cifar_conv1, "run", fake_run_factory("conv1"))
    monkeypatch.setattr(multiseed.synthetic_attention, "run", fake_run_factory("Q/K"))

    multiseed.main(["--seeds", "42", "43", "--save-models-dir", ""])

    results_path = tmp_path / "plots" / "multiseed_results.json"
    assert results_path.exists()
    summary = json.loads(results_path.read_text())
    assert summary["seeds"] == [42, 43]
    assert set(summary["experiments"]) == {"fc", "conv1", "qk"}
    # Ordering should come out FC < conv1 < Q/K on the fabricated deltas.
    fc = summary["experiments"]["fc"]["pi_delta_mean"]
    c1 = summary["experiments"]["conv1"]["pi_delta_mean"]
    qk = summary["experiments"]["qk"]["pi_delta_mean"]
    assert fc < c1 < qk
    assert (tmp_path / "plots" / "polyweave_pi_ordering.pdf").exists()
    assert (tmp_path / "plots" / "polyweave_cifar_fc_recovery_multiseed.pdf").exists()
