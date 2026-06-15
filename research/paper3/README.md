# Paper 3 exploration — PARKED (research record)

This directory holds the exploratory work for a possible third paper on **differentiable
logical reasoning**. It is **parked**, kept as a record of what we tried and why we
stopped — not active work.

## Why parked

Two candidate framings were explored and neither yielded a defensible novel contribution:

1. **Depth generalization** (train shallow proofs, test deep). The synthetic gate caught
   the task being either *shortcut-solvable* (a fact-count leak) or, once balanced,
   *unlearnable in-distribution* by a small transformer — so no clean "learns shallow,
   fails deep" curve emerged. See `run_depth_gen.py`.
2. **A multiplicative rule-induction mechanism** (signed-exponent soft literals). A
   literature check found this is well-covered prior art — **Logical Neural Networks**
   (Riegel et al. 2020), **RL-Net**, **DR-Net**, **RNS** all learn interpretable DNF
   rules with negation. Our version is a re-derivation, not a new capability.

Full reasoning is in [`../paper3_scope.md`](../paper3_scope.md).

## What was salvaged (the real wins)

- **`SoftSignedLiteral` / `SoftRuleLayer`** were promoted into **`polyweave.logic`** as a
  genuine library feature (interpretable differentiable rule induction), independent of
  the paper.
- `polyweave.reasoning` (forward chaining) and `polyweave.logic` (gates, radbas) shipped
  to `main` and underpin the differentiable-logic blog post.

## Contents

| File | What it is |
|---|---|
| `horn_kb.py` | Synthetic Horn-KB generator with a verified proof-depth knob (reusable). |
| `encoding.py` | Canonicalized-slot encoding of instances for a sequence model. |
| `run_depth_gen.py` | Depth-generalization harness (the negative result). |
| `soft_literal_probe.py` | The signed-exponent rule-induction probe (mechanism now in the library). |
| `results/` | Saved run metrics. |
