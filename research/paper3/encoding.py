"""Canonicalized-slot encoding of Horn-KB instances for a sequence model.

Atom names in an instance are arbitrary (``c5``, ``b2``, ``n7`` …), so we map each atom
to a slot id. By default the id is its **order of first appearance** (consistent across
instances): a slot id has no fixed meaning — the same id plays different roles in
different problems — so the model must follow the rule structure, but the consistency
makes the underlying pointer-chasing *learnable* at shallow depth (the regime where the
depth-generalization question is meaningful). Setting ``randomize=True`` shuffles the ids
per instance — a maximally-abstract ablation that we found a small transformer can only
memorize, not learn.

A problem is serialized as a token sequence::

    [CLS] [FACT] s_f1 ... [QUERY] s_q [RULE] s_p1 s_p2 [IMPLIES] s_c [RULE] ...

and the label (is the query entailed?) is read from the ``[CLS]`` position.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

import torch

# Special tokens, then slot tokens at SLOT_OFFSET .. SLOT_OFFSET + max_slots - 1.
PAD, CLS, FACT, RULE, IMPLIES, QUERY = range(6)
SLOT_OFFSET = 6


def vocab_size(max_slots: int) -> int:
    return SLOT_OFFSET + max_slots


def encode(inst, max_slots: int, *, randomize: bool = False,
           rng: Optional[random.Random] = None) -> Tuple[List[int], int]:
    """Encode one :class:`Instance` to ``(token_ids, label)``.

    Slot ids are assigned by order of first appearance (consistent) unless
    ``randomize=True``, which shuffles them per instance (needs ``rng``).
    """
    atoms = []
    seen = set()
    for a in [*inst.facts, inst.query, *(p for prems, _ in inst.rules for p in prems),
              *(c for _, c in inst.rules)]:
        if a not in seen:
            seen.add(a)
            atoms.append(a)
    if len(atoms) > max_slots:
        raise ValueError(f"instance has {len(atoms)} atoms > max_slots {max_slots}")

    if randomize:
        if rng is None:
            raise ValueError("randomize=True requires an rng")
        ids = rng.sample(range(max_slots), len(atoms))
    else:
        ids = list(range(len(atoms)))                     # consistent: first-appearance
    tok = {a: SLOT_OFFSET + i for a, i in zip(atoms, ids)}

    seq = [CLS]
    for f in inst.facts:
        seq += [FACT, tok[f]]
    seq += [QUERY, tok[inst.query]]
    for prems, concl in inst.rules:
        seq += [RULE, *[tok[p] for p in prems], IMPLIES, tok[concl]]
    return seq, int(inst.label)


def encode_batch(
    instances: Sequence, max_slots: int, *, randomize: bool = False,
    rng: Optional[random.Random] = None, max_len: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode + pad a list of instances.

    Returns ``(tokens [B, L], pad_mask [B, L] (True = pad), labels [B])``.
    """
    seqs, labels = [], []
    for inst in instances:
        s, y = encode(inst, max_slots, randomize=randomize, rng=rng)
        seqs.append(s)
        labels.append(y)
    L = max_len or max(len(s) for s in seqs)
    seqs = [s[:L] for s in seqs]
    tokens = torch.full((len(seqs), L), PAD, dtype=torch.long)
    pad_mask = torch.ones(len(seqs), L, dtype=torch.bool)
    for i, s in enumerate(seqs):
        tokens[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        pad_mask[i, : len(s)] = False
    return tokens, pad_mask, torch.tensor(labels, dtype=torch.long)
