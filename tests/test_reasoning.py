"""Tests for polyweave.reasoning — propositional KB + differentiable forward chaining."""

from __future__ import annotations

import pytest
import torch

from polyweave.reasoning import ForwardChainer, ForwardChainingStep, PropKB, print_facts


def rain_kb() -> PropKB:
    kb = PropKB()
    kb.add_rule(["raining"], "wet_grass")
    kb.add_rule(["wet_grass"], "slippery")
    kb.add_rule(["raining"], "umbrella")
    kb.add_rule(["umbrella"], "prepared")
    kb.add_rule(["wet_grass", "sunny"], "rainbow")   # a conjunction
    return kb


# ---------------------------------------------------------------------------
# PropKB
# ---------------------------------------------------------------------------

def test_kb_registers_facts_and_rules():
    kb = rain_kb()
    assert kb.num_rules == 5
    # facts auto-registered: raining, wet_grass, slippery, umbrella, prepared, sunny, rainbow
    assert set(kb.fact_names) == {
        "raining", "wet_grass", "slippery", "umbrella", "prepared", "sunny", "rainbow",
    }
    f0 = kb.initial_facts(["raining"])
    assert f0.shape == (1, kb.num_facts)
    assert f0[0, kb.idx("raining")] == 1.0 and f0.sum() == 1.0


# ---------------------------------------------------------------------------
# Forward chaining — closure & entailment
# ---------------------------------------------------------------------------

def test_forward_chaining_derives_closure():
    kb = rain_kb()
    chainer = ForwardChainer(kb, max_steps=10)
    closure = chainer(kb.initial_facts(["raining"])).reshape(-1)
    # raining -> wet_grass -> slippery, raining -> umbrella -> prepared
    for derived in ("wet_grass", "slippery", "umbrella", "prepared"):
        assert closure[kb.idx(derived)].item() == pytest.approx(1.0)
    # rainbow needs sunny, which was never asserted
    assert closure[kb.idx("rainbow")].item() == pytest.approx(0.0)


def test_conjunction_rule_needs_all_premises():
    kb = rain_kb()
    chainer = ForwardChainer(kb)
    closure = chainer(kb.initial_facts(["raining", "sunny"])).reshape(-1)
    assert closure[kb.idx("rainbow")].item() == pytest.approx(1.0)


def test_entails_helper():
    kb = rain_kb()
    chainer = ForwardChainer(kb)
    f0 = kb.initial_facts(["raining"])
    assert chainer.entails(f0, "prepared")[0] is True
    assert chainer.entails(f0, "rainbow")[0] is False


def test_multi_hop_transitivity():
    kb = PropKB()
    kb.add_rule(["parent_tom_bob"], "ancestor_tom_bob")
    kb.add_rule(["parent_bob_ann"], "ancestor_bob_ann")
    kb.add_rule(["parent_tom_bob", "ancestor_bob_ann"], "ancestor_tom_ann")
    kb.add_rule(["ancestor_tom_ann"], "older_gen_tom_ann")
    chainer = ForwardChainer(kb, max_steps=10)
    closure = chainer(kb.initial_facts(["parent_tom_bob", "parent_bob_ann"])).reshape(-1)
    assert closure[kb.idx("ancestor_tom_ann")].item() == pytest.approx(1.0)
    assert closure[kb.idx("older_gen_tom_ann")].item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Soft / graded truth
# ---------------------------------------------------------------------------

def test_product_t_norm_propagates_graded_truth():
    kb = rain_kb()
    chainer = ForwardChainer(kb, t_norm="product")
    f0 = kb.initial_facts([])
    f0[0, kb.idx("raining")] = 0.7
    closure = chainer(f0).reshape(-1)
    # raining=0.7 -> wet_grass=0.7 -> slippery=0.7 (chain of single-premise rules)
    assert closure[kb.idx("wet_grass")].item() == pytest.approx(0.7, abs=1e-5)
    assert closure[kb.idx("slippery")].item() == pytest.approx(0.7, abs=1e-5)


def test_product_vs_min_differ_on_conjunction():
    # rainbow <- wet_grass & sunny. With wet_grass=0.5 (raining=0.5) and sunny=0.6 the
    # two t-norms separate: product = 0.5*0.6 = 0.30, min = min(0.5, 0.6) = 0.50.
    kb = rain_kb()
    f0 = kb.initial_facts([])
    f0[0, kb.idx("raining")] = 0.5
    f0[0, kb.idx("sunny")] = 0.6
    prod = ForwardChainer(kb, t_norm="product")(f0).reshape(-1)[kb.idx("rainbow")].item()
    mins = ForwardChainer(kb, t_norm="min")(f0).reshape(-1)[kb.idx("rainbow")].item()
    assert prod == pytest.approx(0.5 * 0.6)
    assert mins == pytest.approx(min(0.5, 0.6))
    assert mins > prod


# ---------------------------------------------------------------------------
# Differentiability, monotonicity, convergence
# ---------------------------------------------------------------------------

def test_chaining_is_differentiable_in_facts():
    kb = rain_kb()
    chainer = ForwardChainer(kb, t_norm="product")
    f0 = kb.initial_facts([])
    soft = torch.tensor(0.8, requires_grad=True)
    f0 = f0.clone()
    f0[0, kb.idx("raining")] = soft
    closure = chainer(f0)
    closure.reshape(-1)[kb.idx("slippery")].backward()
    assert soft.grad is not None and torch.isfinite(soft.grad) and soft.grad.item() > 0


def test_one_step_is_monotone():
    kb = rain_kb()
    step = ForwardChainingStep(kb)
    f = kb.initial_facts(["raining"])
    f_new = step(f)
    assert torch.all(f_new >= f - 1e-7)   # facts never decrease


def test_history_records_convergence():
    kb = rain_kb()
    chainer = ForwardChainer(kb, max_steps=20)
    _, history = chainer(kb.initial_facts(["raining"]), return_history=True)
    # converges well before max_steps; last two states identical
    assert torch.allclose(history[-1], history[-2])
    assert len(history) < 20


def test_invalid_t_norm_raises():
    with pytest.raises(ValueError):
        ForwardChainingStep(rain_kb(), t_norm="lukasiewicz")


def test_describe_and_print_facts_run(capsys):
    kb = rain_kb()
    kb.describe()
    print_facts(kb.initial_facts(["raining"]), kb)
    out = capsys.readouterr().out
    assert "Facts (7)" in out and "Rules (5)" in out
    assert "[x] raining" in out
