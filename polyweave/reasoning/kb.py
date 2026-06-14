"""Propositional knowledge base — named facts and Horn-clause rules.

A :class:`PropKB` is the symbolic scaffolding the differentiable chainers operate
over. It assigns each named atom a fixed integer index (so a *fact vector*
``f in [0,1]^N`` has a slot per atom) and stores rules as ``(premise indices,
conclusion index)`` Horn clauses ``a_1 & ... & a_k -> c``. Compiling those into the
tensors a layer consumes is :class:`~polyweave.reasoning.ForwardChainingStep`'s job;
the KB itself stays lightweight and dependency-free.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


class PropKB:
    """A propositional knowledge base: named facts + Horn-clause rules.

    Example:
        >>> kb = PropKB()
        >>> kb.add_rule(["raining"], "wet_grass")        # raining -> wet_grass
        >>> kb.add_rule(["wet_grass"], "slippery")       # wet_grass -> slippery
        >>> kb.add_rule(["wet_grass", "sunny"], "rainbow")  # a conjunction
        >>> f0 = kb.initial_facts(["raining"])           # (1, N) truth vector
    """

    def __init__(self) -> None:
        self._fact_index: Dict[str, int] = {}
        self.rules: List[Tuple[List[int], int]] = []   # (premise_indices, conclusion_idx)
        self.rule_names: List[str] = []

    # -- facts -----------------------------------------------------------------

    def add_fact(self, name: str) -> int:
        """Register ``name`` (idempotent) and return its integer index."""
        if name not in self._fact_index:
            self._fact_index[name] = len(self._fact_index)
        return self._fact_index[name]

    @property
    def fact_names(self) -> List[str]:
        """Fact names in index order."""
        return [k for k, _ in sorted(self._fact_index.items(), key=lambda kv: kv[1])]

    @property
    def num_facts(self) -> int:
        return len(self._fact_index)

    def idx(self, name: str) -> int:
        """Integer index of a registered fact (raises ``KeyError`` if unknown)."""
        return self._fact_index[name]

    def initial_facts(self, true_facts: List[str]) -> torch.Tensor:
        """A ``(1, N)`` truth vector with ``1.0`` at each name in ``true_facts``."""
        f = torch.zeros(1, self.num_facts)
        for name in true_facts:
            f[0, self.idx(name)] = 1.0
        return f

    # -- rules -----------------------------------------------------------------

    def add_rule(self, premises: List[str], conclusion: str, name: str = "") -> None:
        """Add a Horn clause ``and(premises) -> conclusion``.

        All names are auto-registered as facts. ``name`` is an optional human label
        (defaults to a rendered ``"a, b -> c"``).
        """
        for p in premises:
            self.add_fact(p)
        self.add_fact(conclusion)
        prem_indices = [self.idx(p) for p in premises]
        concl_index = self.idx(conclusion)
        self.rules.append((prem_indices, concl_index))
        self.rule_names.append(name or (", ".join(premises) + " -> " + conclusion))

    @property
    def num_rules(self) -> int:
        return len(self.rules)

    def describe(self) -> None:
        """Print a short summary of facts and rules."""
        print(f"Facts ({self.num_facts}): {self.fact_names}")
        print(f"Rules ({self.num_rules}):")
        for i, name in enumerate(self.rule_names):
            print(f"  R{i}: {name}")
