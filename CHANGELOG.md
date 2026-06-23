# Changelog

All notable changes to PolyWeave are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); the project is pre-1.0, so minor
versions may include API changes.

## [0.2.0] - 2026-06-23

Differentiable **logic, reasoning, and rule induction** — broadening PolyWeave into a
small "AI-maths" toolkit alongside the multiplicative layers. The product-t-norm AND that
underpins all of it is a literal Pi neuron, tying logic back to the library's core.

### Added
- **`polyweave.logic` — fuzzy gates.** `fuzzy_and/or/not/nand/nor/xor/xnor` (product or
  `min` t-norm), as functions and parameter-free `nn.Module`s (`FuzzyAnd`, `FuzzyOr`, …).
- **`polyweave.logic` — rule induction.** `SoftSignedLiteral` and `SoftRuleLayer`:
  interpretable learnable rules with *native negation* via signed log-space exponents.
  `literals()` / `rules_text()` read the learned rule (e.g. `bird & not penguin`) straight
  off the weights; `exponent_abs_mean()` is the recruitment diagnostic.
- **`polyweave.ops.radbas`** — radial-basis activation `exp(-(eps*x)**2)` (fuzzy equality;
  the RBF route to XOR).
- **`polyweave.reasoning`** — differentiable forward chaining over a propositional Horn
  knowledge base: `PropKB`, `ForwardChainingStep`, `ForwardChainer` (product/`min` t-norm,
  `entails` helper). Sound and complete for Horn entailment, and differentiable in the facts.
- **`polyweave.viz`** — `plot_rule_exponents` (signed-exponent rule chart) and
  `plot_chaining_trace` (truth-propagation heatmap).

### Documentation
- New example pages: *Logic gates as neurons* and *Forward chaining as a differentiable layer*.
- New blog post: *Logic, in Products* — gates → XOR in one neuron → forward chaining → rule
  induction — with embedded figures rendered by the new viz helpers.

### Notes
- The rule-induction layers are deliberately in the lineage of Logical Neural Networks and
  RL-Net / DR-Net; PolyWeave offers them in its geometric-product / recruitment framing
  rather than as a new capability.

## [0.1.0] - 2026-06-12

Initial release.

### Added
- Multiplicative layers: `SigmaPiLinear` (genuine geometric-product pi branch),
  `ConvSigmaPi2d`, and `PolyLinear` (low-rank bilinear), with recruitment diagnostics.
- Hypernetwork stack: `prototypes`, `students`, `targets`, `hypernets`, a generic `training`
  loop, and `evaluation` (zero-shot / recovery / ensemble).
- Activation-space distillation (`polyweave.distill`), interpretability (occlusion probes),
  and publication-quality plotting (`polyweave.viz`).
- CI/CD (test matrix + trusted-publishing release), an MkDocs documentation site, and the
  Apache-2.0 license.
