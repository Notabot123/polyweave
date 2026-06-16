"""Tests for the new polyweave.viz helpers (rule exponents, chaining trace)."""

from __future__ import annotations

import torch

from polyweave.logic import SoftSignedLiteral
from polyweave.reasoning import ForwardChainer, PropKB
from polyweave.viz import plot_chaining_trace, plot_rule_exponents


def test_plot_rule_exponents_single(tmp_path):
    lit = SoftSignedLiteral(4)
    with torch.no_grad():
        lit.w.copy_(torch.tensor([1.0, -1.0, 0.0, 0.05]))
    weights = dict(zip(["bird", "penguin", "d2", "d3"], lit.w.tolist()))
    paths = plot_rule_exponents({"bird & not penguin": weights}, "rule_demo", plots_dir=tmp_path)
    assert len(paths) == 2 and all(p.exists() for p in paths)


def test_plot_rule_exponents_multi(tmp_path):
    rules = {"r0": {"bird": 1.0, "penguin": -1.0}, "r1": {"bat": 0.9, "broken": -0.8}}
    paths = plot_rule_exponents(rules, "rules_demo", plots_dir=tmp_path)
    assert all(p.exists() for p in paths)


def test_plot_chaining_trace_from_history(tmp_path):
    kb = PropKB()
    kb.add_rule(["raining"], "wet_grass")
    kb.add_rule(["wet_grass"], "slippery")
    _, history = ForwardChainer(kb)(kb.initial_facts(["raining"]), return_history=True)
    paths = plot_chaining_trace(history, kb.fact_names, "trace_demo", plots_dir=tmp_path)
    assert len(paths) == 2 and all(p.exists() for p in paths)


def test_plot_chaining_trace_accepts_2d_array(tmp_path):
    trace = [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
    paths = plot_chaining_trace(trace, ["a", "b", "c"], "trace2d", plots_dir=tmp_path)
    assert all(p.exists() for p in paths)
