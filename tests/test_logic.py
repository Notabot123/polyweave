"""Tests for polyweave.ops.radbas and the polyweave.logic fuzzy gates."""

from __future__ import annotations

import itertools

import pytest
import torch

from polyweave import radbas
from polyweave.logic import (
    FuzzyAnd,
    FuzzyNand,
    FuzzyNor,
    FuzzyNot,
    FuzzyOr,
    FuzzyXnor,
    FuzzyXor,
    fuzzy_and,
    fuzzy_nand,
    fuzzy_nor,
    fuzzy_not,
    fuzzy_or,
    fuzzy_xnor,
    fuzzy_xor,
)

T_NORMS = ("product", "min")
CORNERS = list(itertools.product([0.0, 1.0], repeat=2))  # (a, b) Boolean inputs


# ---------------------------------------------------------------------------
# radbas
# ---------------------------------------------------------------------------

def test_radbas_peaks_at_zero_and_decays():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = radbas(x, epsilon=1.0)
    assert torch.isclose(y[2], torch.tensor(1.0))          # peak at x=0
    assert torch.all(y > 0) and torch.all(y <= 1.0)        # in (0, 1]
    assert torch.allclose(y, y.flip(0))                    # even / symmetric
    assert y[1] > y[0] and y[3] > y[4]                     # decays away from 0


def test_radbas_epsilon_sharpens():
    x = torch.tensor([0.1])
    # larger epsilon -> sharper bump -> smaller value away from the centre
    assert radbas(x, epsilon=10.0).item() < radbas(x, epsilon=1.0).item()


def test_radbas_is_differentiable():
    x = torch.tensor([0.5], requires_grad=True)
    radbas(x, epsilon=2.0).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Gate truth tables (exact on the Boolean corners, for both t-norms)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("t_norm", T_NORMS)
def test_gate_truth_tables(t_norm):
    for a_v, b_v in CORNERS:
        a, b = torch.tensor(a_v), torch.tensor(b_v)
        ai, bi = int(a_v), int(b_v)
        assert fuzzy_not(a).item() == pytest.approx(1 - ai)
        assert fuzzy_and(a, b, t_norm).item() == pytest.approx(ai & bi)
        assert fuzzy_or(a, b, t_norm).item() == pytest.approx(ai | bi)
        assert fuzzy_nand(a, b, t_norm).item() == pytest.approx(1 - (ai & bi))
        assert fuzzy_nor(a, b, t_norm).item() == pytest.approx(1 - (ai | bi))
        assert fuzzy_xor(a, b, t_norm).item() == pytest.approx(ai ^ bi)
        assert fuzzy_xnor(a, b, t_norm).item() == pytest.approx(1 - (ai ^ bi))


@pytest.mark.parametrize("t_norm", T_NORMS)
def test_gates_stay_in_unit_interval(t_norm):
    torch.manual_seed(0)
    a, b = torch.rand(1000), torch.rand(1000)
    for gate in (fuzzy_and, fuzzy_or, fuzzy_nand, fuzzy_nor, fuzzy_xor, fuzzy_xnor):
        out = gate(a, b, t_norm)
        assert torch.all(out >= -1e-6) and torch.all(out <= 1 + 1e-6)


def test_product_xor_is_the_sigma_pi_identity():
    # The product-t-norm XOR must be exactly a + b - 2ab (a degree-2 neuron).
    torch.manual_seed(0)
    a, b = torch.rand(500), torch.rand(500)
    assert torch.allclose(fuzzy_xor(a, b, "product"), a + b - 2 * a * b, atol=1e-6)


def test_de_morgan_consistency():
    torch.manual_seed(0)
    a, b = torch.rand(500), torch.rand(500)
    # NOT(a AND b) == (NOT a) OR (NOT b)
    assert torch.allclose(
        fuzzy_not(fuzzy_and(a, b, "product")),
        fuzzy_or(fuzzy_not(a), fuzzy_not(b), "product"),
        atol=1e-6,
    )


def test_radbas_xor_route_matches_truth_table():
    # 1 - radbas(a - b) is a fuzzy XOR (radbas(a-b) is XNOR / equality).
    for a_v, b_v in CORNERS:
        a, b = torch.tensor(a_v), torch.tensor(b_v)
        xor = 1.0 - radbas(a - b, epsilon=10.0)
        assert xor.item() == pytest.approx(int(a_v) ^ int(b_v), abs=1e-3)


def test_module_gates_match_functions():
    torch.manual_seed(0)
    a, b = torch.rand(64), torch.rand(64)
    assert torch.allclose(FuzzyNot()(a), fuzzy_not(a))
    binary = [
        (FuzzyAnd, fuzzy_and),
        (FuzzyOr, fuzzy_or),
        (FuzzyNand, fuzzy_nand),
        (FuzzyNor, fuzzy_nor),
        (FuzzyXor, fuzzy_xor),
        (FuzzyXnor, fuzzy_xnor),
    ]
    for module_cls, fn in binary:
        for t_norm in T_NORMS:
            assert torch.allclose(module_cls(t_norm)(a, b), fn(a, b, t_norm))
    assert "t_norm='min'" in repr(FuzzyAnd("min"))


def test_invalid_t_norm_raises():
    with pytest.raises(ValueError):
        fuzzy_and(torch.tensor(1.0), torch.tensor(1.0), t_norm="lukasiewicz")
    with pytest.raises(ValueError):
        FuzzyOr("nonsense")


# ---------------------------------------------------------------------------
# The headline: one multiplicative neuron learns XOR; a linear one cannot.
# ---------------------------------------------------------------------------

def _fit_xor(model, steps=4000, lr=0.05, seed=0):
    torch.manual_seed(seed)
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(X), Y)
        loss.backward()
        opt.step()
    return torch.nn.functional.mse_loss(model(X), Y).item()


def test_poly_neuron_learns_xor_but_linear_cannot():
    from polyweave import PolyLinear

    # A single rank-1 degree-2 neuron has the bilinear term XOR needs.
    poly_mse = _fit_xor(PolyLinear(2, 1, rank=1))
    # A plain linear neuron cannot separate XOR — best it can do is predict ~0.5
    # everywhere (MSE ~0.25).
    linear_mse = _fit_xor(torch.nn.Linear(2, 1))

    assert poly_mse < 0.02, f"PolyLinear failed to learn XOR (mse={poly_mse:.4f})"
    assert linear_mse > 0.15, f"Linear unexpectedly fit XOR (mse={linear_mse:.4f})"
