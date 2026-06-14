"""Differentiable forward chaining over a propositional Horn knowledge base.

Forward chaining repeatedly applies every rule to the current facts until nothing
new is derived (the *fixpoint* / deductive closure). For propositional Horn clauses
this is **sound and complete for entailment**: a goal is entailed iff its truth value
is driven high by chaining. We implement one chaining *step* as a small differentiable
module and iterate it (weight-tied, like an unrolled RNN) to the fixpoint.

One step, given a fact vector ``f in [0,1]^N``:

1. **AND the premises** of each rule ``r``:
   ``fire_r = prod_{j in premises(r)} f_j``  (product t-norm — a Pi neuron, exactly
   :func:`polyweave.logic.fuzzy_and`), or ``min`` for crisp Gödel logic.
2. **OR the conclusions**: ``derived_j = max_r (fire_r if r concludes j)``.
3. **Update** monotonically: ``f_new = max(f, derived)``.

Because the premise-AND is a product, gradients flow through *every* premise, so the
whole stack is differentiable in the facts — you can backpropagate a downstream loss
to soft inputs, or embed chaining as a reasoning layer inside a larger network. (Rule
structure is frozen here; making the masks learnable ``nn.Parameter``s — soft rule
induction — is the natural next extension.)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .kb import PropKB


class ForwardChainingStep(nn.Module):
    """One differentiable forward-chaining step over a :class:`PropKB`.

    Args:
        kb: the knowledge base (facts + rules) to compile into frozen masks.
        t_norm: ``"product"`` (default; ``and = prod``, smooth, the Sigma-Pi form)
            or ``"min"`` (crisp Gödel conjunction).
    """

    def __init__(self, kb: PropKB, t_norm: str = "product") -> None:
        super().__init__()
        if t_norm not in ("product", "min"):
            raise ValueError(f"t_norm must be 'product' or 'min', got {t_norm!r}")
        self.t_norm = t_norm

        N, R = kb.num_facts, kb.num_rules
        premise_mask = torch.zeros(R, N)
        conclusion_oh = torch.zeros(R, N)
        for r, (prems, concl) in enumerate(kb.rules):
            for p in prems:
                premise_mask[r, p] = 1.0
            conclusion_oh[r, concl] = 1.0
        self.register_buffer("premise_mask", premise_mask)    # (R, N)
        self.register_buffer("conclusion_oh", conclusion_oh)  # (R, N)

    def forward(self, facts: torch.Tensor) -> torch.Tensor:
        """Apply one chaining step. ``facts``: ``(batch, N)`` -> ``(batch, N)``."""
        f = facts.unsqueeze(1)              # (batch, 1, N)
        mask = self.premise_mask            # (R, N)

        # Set non-premise positions to 1 (neutral for both product and min), so the
        # reduction sees only the rule's premises.
        masked = f * mask + (1.0 - mask)    # (batch, R, N)
        if self.t_norm == "product":
            fire = masked.prod(dim=-1)      # (batch, R)
        else:
            fire = masked.min(dim=-1).values

        # OR the rule activations onto their conclusion facts.
        derived = (fire.unsqueeze(-1) * self.conclusion_oh).max(dim=1).values  # (batch, N)
        return torch.max(facts, derived)


class ForwardChainer(nn.Module):
    """Iterate :class:`ForwardChainingStep` to the fixpoint (deductive closure).

    Args:
        kb: the knowledge base.
        max_steps: maximum chaining iterations (bounds reasoning depth).
        t_norm: ``"product"`` or ``"min"``.
        tol: stop early once a step changes the facts by less than this.
    """

    def __init__(self, kb: PropKB, max_steps: int = 10,
                 t_norm: str = "product", tol: float = 1e-6) -> None:
        super().__init__()
        self.kb = kb
        self.max_steps = max_steps
        self.tol = tol
        self.step = ForwardChainingStep(kb, t_norm=t_norm)

    def forward(
        self, facts: torch.Tensor, return_history: bool = False
    ) -> "torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]":
        """Chain ``facts`` to the fixpoint.

        Args:
            facts: ``(batch, N)`` initial truth vector.
            return_history: also return the per-step list of fact tensors.

        Returns:
            The closure ``(batch, N)``, or ``(closure, history)`` if requested.
        """
        history: Optional[List[torch.Tensor]] = [facts.clone()] if return_history else None
        f = facts
        for _ in range(self.max_steps):
            f_new = self.step(f)
            if return_history:
                history.append(f_new.clone())
            if (f_new - f).abs().max().item() < self.tol:
                f = f_new
                break
            f = f_new
        return (f, history) if return_history else f

    @torch.no_grad()
    def entails(self, facts: torch.Tensor, goal: str, threshold: float = 0.5) -> Tuple[bool, float]:
        """Does the KB entail ``goal`` from ``facts``? Returns ``(entailed, truth)``."""
        closure = self(facts)
        truth = closure.reshape(-1)[self.kb.idx(goal)].item()
        return truth >= threshold, truth


def print_facts(facts: torch.Tensor, kb: PropKB, threshold: float = 0.5) -> None:
    """Print each fact's truth value, ticking those at/above ``threshold``."""
    f = facts.reshape(-1)
    for name in kb.fact_names:
        v = f[kb.idx(name)].item()
        print(f"  {'[x]' if v >= threshold else '[ ]'} {name:<22s} {v:.4f}")
