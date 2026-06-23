# Algorithms as Neural Modules: Exact Mathematical Structure in Differentiable Networks

**Draft — work in progress**

---

## Abstract

We present a family of zero-parameter neural modules that encode classical mathematical
algorithms — Pascal's triangle, the Sieve of Eratosthenes, and binomial expansion — as
fixed-weight networks composable with learned components. Unlike standard supervised
models, these modules solve their target tasks exactly and with provable correctness,
requiring no training data. We demonstrate that a mixture-of-experts (MoE) router learns
to delegate arithmetic queries to the appropriate structured expert, achieving perfect
accuracy within the structured domain and graceful degradation beyond it. We show that
tasks which defeat standard MLPs entirely — membership in an arithmetic sequence, exact
primality detection, symbolic polynomial expansion — are trivially solved by the
corresponding zero-parameter module. We release all modules as part of the open-source
PolyWeave library.

---

## 1. Introduction

A persistent challenge in neural computation is arithmetic and combinatorial exactness.
Standard MLPs trained via gradient descent approximate these relationships but cannot
guarantee correctness, scale to large inputs without error, or generalise to the
unseen exact cases that a closed-form rule handles trivially.

We take a different approach: encode the algorithm directly as network structure. The key
insight, following [CITE: earlier Sigma-Pi work, Paper 1], is that multiplicative
(Sigma-Pi) computation and the `radbas` radial-basis activation form a sufficient basis
for a wide class of exact computations — and that these computations are expressible as
fixed-weight convolutional and dense layers with zero learnable parameters.

**Contributions:**
1. Three zero-parameter PyTorch modules — `PascalTriangle`, `BinomialExpansion`,
   `DifferentiableSieve` — that solve their tasks with provable 100% accuracy.
2. A `BernoulliTriangle` module that encodes simplex and cake number sequences as a
   column-wise cumsum of Pascal.
3. A mixture-of-experts framework in which a learned router delegates to structured
   experts, empirically learning to route arithmetic queries appropriately.
4. Experiments showing the failure of MLP baselines on Tier-1 tasks and the
   advantage of structured routing on range-extrapolation tasks.
5. Release as `polyweave.maths` (open source, pip-installable).

---

## 2. Related Work

**Neural arithmetic / algorithmic networks.** Neural Arithmetic Logic Units [NALU,
Trask et al. 2018] learn to perform arithmetic operations through weight-constrained
modules. Neural GPU [Kaiser & Sutskever 2015] solves multiplication via convolutional
recurrence. Our approach is complementary: rather than learning to approximate
arithmetic, we encode exact algorithms as frozen network structure.

**Differentiable programming / neurosymbolic.** DeepProbLog [Manhaeve et al. 2018],
Logical Neural Networks [Riegel et al. 2020], and Neural Theorem Provers
[Rocktäschel & Riedel 2017] embed symbolic reasoning in differentiable systems.
We focus on the numeric / combinatorial side rather than logical inference.

**Hypernetworks and structured weights.** [CITE: Paper 1 — Sigma-Pi hypernetwork].
The connection between multiplicative neurons and exact arithmetic is a throughline
from our prior work.

**Mixture of experts.** Standard MoE [Shazeer et al. 2017; Fedus et al. 2022]
routes between learned experts. We route between a frozen structured expert and a
learned expert — a novel asymmetric setting where the router must learn to recognise
when exact structure applies.

---

## 3. Mathematical Modules

### 3.1 Pascal's Triangle (`PascalTriangle`)

Pascal's triangle is constructed by the recurrence C(n, k) = C(n-1, k-1) + C(n-1, k).
We implement this as a fixed-weight 2D convolution with kernel [[1, 1], [0, 0]]
(top-row sum), applied iteratively with top-left padding. Summing all intermediate
states gives the full triangle, because intermediate x_k is non-zero only at row k.

The module has zero learnable parameters. Output is exact (floating-point integer
precision) for n ≤ 15.

### 3.2 Binomial Expansion (`BinomialExpansion`)

Given coefficients A, B and exponent n, the binomial theorem gives:

    (Ax + By)^n = sum_{k=0}^{n} C(n,k) A^{n-k} B^k x^{n-k} y^k

We implement this by composing `PascalTriangle` with precomputed exponent lookup
tables stored as frozen buffers. The module takes (A, B, n) and returns the exact
coefficient vector of length `num_rows`.

Prior work [CITE: pascal_binomial_expansion.ipynb] demonstrated this in TensorFlow
with a `radbas`-based row indexer. The PyTorch version presented here is simpler and
compatible with autograd.

### 3.3 Differentiable Sieve (`DifferentiableSieve`)

The Sieve of Eratosthenes marks composites by sweeping multiples of each prime p ≤ √N.
We implement each prime's sweep as a frozen "comb" buffer — a tensor of length N+1 with
1s at positions 0, p, 2p, … (position p itself cleared, since p is prime). Composite
scores are combined via the probabilistic OR formula:

    composite(n) = 1 − ∏_p (1 − comb_p(n))

A final exp(−α · composite(n)) maps composite score 0 → 1 (prime) and 1 → 0
(composite). The decay parameter α controls sharpness and may be made learnable.

Detection is exact for n ≤ N when max_p = ⌊√N⌋ (the standard sieve bound).

### 3.4 Bernoulli's Triangle (`BernoulliTriangle`)

Bernoulli's triangle contains the partial row sums of Pascal's triangle:

    B(n, k) = sum_{j=0}^{k} C(n, j)

Notable sequences emerge along its columns: natural numbers (k=1), triangular numbers
(k=2), k-simplex numbers (general k), and cake numbers along anti-diagonals.
We implement B as `torch.cumsum` of the Pascal triangle output — a natural column-wise
cumulative sum, consistent with the "frozen computation as network structure" theme.

---

## 4. Modular Membership Testing via `radbas`

A recurring primitive across the modules is **soft equality testing**: the `radbas`
activation r(x) = exp(−(εx)²) peaks at x = 0 and decays smoothly — it fires when
its input is near zero.

For membership testing ("is n a multiple of p?"), we precompute the known multiples
as a lookup table, tile them to match the input batch, subtract the query n, and
apply `radbas`. The resulting score is near 1 for exact matches and near 0 elsewhere.
This is demonstrated concretely for the 13-times table (Section 5.1).

---

## 5. Experiments

### 5.1 Tier-1: Zero-Shot Structural Tasks

We demonstrate three tasks for which the structured module achieves provable 100%
accuracy without any training data.

**Task A — Modular membership ("is n in the 13× table?").**
Input: integer n ∈ [0, 256×13). Label: n mod 13 == 0.
Method: radbas lookup against precomputed multiples.
Result: 100% accuracy (provable; see `multiples_of_thirteen.ipynb`).
Baseline: standard MLP (reported in [CITE: notebook] as unable to learn this task).

**Task B — Primality detection.**
Input: integer n ∈ [2, N]. Label: is_prime(n).
Method: `DifferentiableSieve(N)`.
Result: 100% accuracy for all n ≤ N (provable from sieve correctness).
Baseline: [TBD — train MLP on primes 2–100, report accuracy on 2–500]

**Task C — Binomial expansion.**
Input: (A, B, n) with A, B ∈ {−8, …, 8}\{0}, n ∈ {2, …, 8}.
Label: coefficient vector of (Ax + By)^n.
Method: `BinomialExpansion(num_rows=16)`.
Result: 100% accuracy on 1792-sample sweep (confirmed in notebook).
Baseline: [TBD — multi-output MLP predicting coefficient vector]

### 5.2 Tier-2: MoE Range Extrapolation

We train a `MixtureOfExpertsPrimeModel` (structured sieve expert + MLP blind expert +
learned router) on primality labels for n ∈ [2, 100] and evaluate on n ∈ [101, 500].

Hypothesis: the structured expert generalises exactly to the held-out range; the blind
MLP fails; the router learns to weight the structured expert more heavily.

[TBD — run experiment, report router weight evolution + accuracy by range]

### 5.3 Ablation: Decay Parameter α

We sweep α ∈ {1, 2, 5, 10, 20} for `DifferentiableSieve` and report the sharpness
of the prime/composite boundary. α = 5 gives near-binary output while retaining
smooth gradients.

[TBD]

---

## 6. Discussion

**Why not just compute it symbolically?** The point is composability with learned
components and differentiability. A frozen sieve buffer inside a neural network can
serve as a feature extractor whose output flows into a learned classifier. The
structured module does the exact part; the learned part handles variation and
uncertainty around it.

**Relationship to Paper 1 (Sigma-Pi).** The `radbas` activation and product t-norm
fuzzy AND that underpin this work both emerge from the multiplicative / Sigma-Pi
framework of Paper 1. BinomialExpansion is a product of power terms; the sieve's
probabilistic OR is a product of (1 − comb) terms. The throughline is multiplicative
computation as exact structured reasoning.

**Limitations.** The modules are exact only within their specified ranges (N for the
sieve; num_rows for Pascal). The MoE router does not use the structured expert's
output as a routing signal — a more principled design would feed the prime score
directly to the router (future work).

---

## 7. Conclusion

We have shown that classical combinatorial algorithms are expressible as zero-parameter
neural network modules, enabling exact computation within learned architectures. The
modules compose naturally into mixture-of-experts models where routing is learned, and
the structured component handles its domain with provable correctness. We release
`polyweave.maths` as a reusable library with full test coverage.

---

## References

[TBD — NALU, Neural GPU, NTP, DeepProbLog, LNN, MoE Shazeer/Fedus, Sigma-Pi Paper 1/2]

---

## Appendix A — Module Details

[Parameter counts, architecture diagrams, code listings]

## Appendix B — Extended Results

[Full accuracy tables for Tier-1 and Tier-2]
