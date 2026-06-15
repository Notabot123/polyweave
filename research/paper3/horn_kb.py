"""Synthetic propositional Horn-KB generator with a controlled proof-depth knob.

This is the controlled core of Paper 3 (see ../paper3_scope.md). Each instance is a
``(facts, rules, query, label, depth)`` problem whose **minimal proof depth** we set
exactly, so we can train models on shallow proofs and test on deep ones.

How depth is controlled (construct-then-verify)
-----------------------------------------------
For a positive instance of depth ``d`` we plant an explicit chain of fresh atoms
``c0 -> c1 -> ... -> cd`` (one rule per hop, optionally with an extra always-true base
premise to exercise conjunctions). ``c0`` is asserted; the query is ``cd``. Distractor
rules only ever conclude *noise* atoms disjoint from the chain, so they can neither
shorten nor create the proof — the minimal depth to derive ``cd`` is exactly ``d``.

Negatives are *hard and count-balanced*: ``c0`` is still asserted, but one rule on the
query's chain has its load-bearing premise swapped for an always-unsatisfiable atom,
breaking the proof. Positives and negatives then have identical fact, atom, and rule
counts — so the label is decidable only by tracing the chain, never by a surface count
(an earlier "withhold c0" design leaked the label as fact-count and a transformer solved
it without reasoning).

Every instance is then checked against :class:`polyweave.reasoning.ForwardChainer`
(sound & complete for Horn): the step at which the query first crosses 0.5 must equal
the planted depth (positives), or the query must stay false (negatives). Generation
raises if the oracle ever disagrees, so the dataset is correct by construction *and*
by verification.

Run:  python research/paper3/horn_kb.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from polyweave.reasoning import ForwardChainer, PropKB

Rule = Tuple[List[str], str]


@dataclass
class Instance:
    facts: List[str]            # initially-true atoms
    rules: List[Rule]           # (premises, conclusion) Horn clauses
    query: str                  # atom to decide
    label: bool                 # is query entailed?
    depth: Optional[int]        # minimal proof depth (None if not entailed)

    def build(self) -> Tuple[PropKB, torch.Tensor]:
        """Compile to a ``PropKB`` and an initial ``(1, N)`` truth vector."""
        kb = PropKB()
        for prems, concl in self.rules:
            kb.add_rule(prems, concl)
        for f in self.facts:           # ensure isolated facts/queries are registered
            kb.add_fact(f)
        kb.add_fact(self.query)
        return kb, kb.initial_facts(self.facts)


def _measured_depth(inst: Instance, max_steps: int) -> Optional[int]:
    """Step at which ``query`` first becomes true under forward chaining (or None)."""
    kb, f0 = inst.build()
    chainer = ForwardChainer(kb, max_steps=max_steps)
    _, history = chainer(f0, return_history=True)
    qi = kb.idx(inst.query)
    for step, h in enumerate(history):
        if h.reshape(-1)[qi].item() >= 0.5:
            return step            # step 0 = already a fact
    return None


def generate_instance(
    rng: random.Random,
    depth: int,
    *,
    label: bool = True,
    n_distractor_rules: int = 6,
    n_base_facts: int = 4,
    conjunction_prob: float = 0.4,
) -> Instance:
    """Generate one instance with planted minimal proof depth ``depth``.

    Args:
        rng: seeded ``random.Random`` for reproducibility.
        depth: minimal proof depth of the (positive) chain (>= 1).
        label: ``True`` for an entailed instance; ``False`` for a hard negative
            (same chain, base premise withheld).
        n_distractor_rules: irrelevant rules concluding noise atoms.
        n_base_facts: pool of always-true base atoms (used as extra AND premises).
        conjunction_prob: chance each chain hop gains an extra base-fact premise.
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")
    uid = [0]

    def fresh(tag: str) -> str:
        uid[0] += 1
        return f"{tag}{uid[0]}"

    base_facts = [fresh("b") for _ in range(n_base_facts)]
    chain = [fresh("c") for _ in range(depth + 1)]   # c0 .. cd

    chain_rules: List[list] = []
    for i in range(1, depth + 1):
        prems = [chain[i - 1]]
        if rng.random() < conjunction_prob:
            prems.append(rng.choice(base_facts))     # extra AND premise (always true)
        chain_rules.append([prems, chain[i]])

    # Distractors conclude *noise* atoms only (never chain atoms) -> no proof shortcut.
    noise = [fresh("n") for _ in range(max(3, n_distractor_rules))]
    distractors: List[list] = []
    for _ in range(n_distractor_rules):
        k = rng.randint(1, 2)
        prems = rng.sample(base_facts + noise, k)
        concl = rng.choice([n for n in noise if n not in prems])  # no trivial self-loops
        distractors.append([prems, concl])

    # An always-unsatisfiable atom, present in BOTH labels (keeps atom/rule counts equal).
    dead = fresh("d")
    distractors.append([[dead], rng.choice(noise)])  # never fires (dead is never true)

    if not label:
        # Break the query's chain at a random hop: swap its load-bearing premise for
        # `dead`. Facts, atoms, and rule count match a positive exactly — only one buried
        # premise differs, so the label is decidable solely by tracing the chain.
        h = rng.randrange(depth)                     # the rule producing chain[h + 1]
        chain_rules[h][0][0] = dead

    rules: List[Rule] = [(list(p), c) for p, c in chain_rules + distractors]
    rng.shuffle(rules)
    facts = base_facts + [chain[0]]                  # c0 asserted in BOTH cases
    return Instance(facts, rules, chain[depth], label, depth if label else None)


def make_dataset(
    rng: random.Random,
    depths: List[int],
    n_per_depth: int,
    *,
    frac_positive: float = 0.5,
    verify: bool = True,
    **kw,
) -> List[Instance]:
    """Build and (by default) verify a balanced dataset across ``depths``."""
    data: List[Instance] = []
    for d in depths:
        for j in range(n_per_depth):
            label = (j / n_per_depth) < frac_positive
            inst = generate_instance(rng, d, label=label, **kw)
            if verify:
                measured = _measured_depth(inst, max_steps=d + 5)
                if label and measured != d:
                    raise AssertionError(f"depth mismatch: planted {d}, measured {measured}")
                if not label and measured is not None:
                    raise AssertionError(f"negative was entailed at step {measured}")
            data.append(inst)
    return data


if __name__ == "__main__":
    rng = random.Random(0)
    depths = list(range(1, 9))
    data = make_dataset(rng, depths, n_per_depth=50, verify=True)

    pos = [d for d in data if d.label]
    neg = [d for d in data if not d.label]
    print(f"Generated + VERIFIED {len(data)} instances "
          f"({len(pos)} positive, {len(neg)} negative) over depths {depths[0]}..{depths[-1]}")
    print("All planted depths match the forward-chaining oracle; all negatives unprovable.\n")

    avg_atoms = sum(len(i.build()[0].fact_names) for i in data) / len(data)
    avg_rules = sum(len(i.rules) for i in data) / len(data)
    print(f"avg atoms/instance ~ {avg_atoms:.1f} | avg rules/instance ~ {avg_rules:.1f}")

    ex = pos[0]
    print(f"\nExample positive (depth {ex.depth}):")
    print(f"  facts: {ex.facts}")
    print(f"  query: {ex.query}  -> label {ex.label}")
    print(f"  {len(ex.rules)} rules, e.g. {ex.rules[:3]}")
