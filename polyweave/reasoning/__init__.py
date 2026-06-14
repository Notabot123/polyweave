"""Differentiable symbolic reasoning — forward chaining over a propositional KB.

Define facts and Horn-clause rules in a :class:`PropKB`, then run a
:class:`ForwardChainer` to the deductive closure. The premise conjunction is a product
t-norm (a Pi neuron, see :mod:`polyweave.logic`), so chaining is differentiable in the
facts: you can backpropagate through it or embed it as a reasoning layer.

For propositional Horn clauses, forward chaining to the fixpoint is sound and complete
for entailment, so a goal-directed *backward* chainer is not needed to answer "does the
KB entail the goal?". A genuinely differentiable backward/Neural-Theorem-Prover-style
prover (soft unification + proof trees) is a deliberate future addition, not required
here.
"""

from __future__ import annotations

from .forward import ForwardChainer, ForwardChainingStep, print_facts
from .kb import PropKB

__all__ = [
    "PropKB",
    "ForwardChainingStep",
    "ForwardChainer",
    "print_facts",
]
