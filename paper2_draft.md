# How Linear Is a Transformer Feed-Forward Block? Per-Block Linear Recoverability Is Learned, Not Architectural

> **Draft / working sketch.** Markdown intended for migration to the NeurIPS LaTeX
> template (arXiv submission first). Multi-seed results (seeds 42/43/44) for GPT-2,
> Pythia-160m, and llama-160m; the closed-form linear baseline is deterministic and
> reported as a single number (no ±std). Remaining gaps are prose/citation polish, not
> experiments. Author: Stuart Whipp.

## Abstract

A transformer block's feed-forward network (FFN) — two dense projections with a
pointwise nonlinearity between them — accounts for roughly two-thirds of the block's
parameters, and is widely read as the site of much of a model's stored computation. We
ask a structural question: **how much of an FFN block is a linear map?** We measure it
directly. Decomposing an FFN's input→output map as a best linear approximation plus a
residual, we fit the *exact closed-form least-squares* linear map to a block's cached
activation pairs and call the held-out variance it explains the block's **linear
recoverability** (R²_lin); the rest (1 − R²_lin) is its **residual nonlinearity**. We
then probe the residual with an explicit low-rank bilinear ("poly") layer to ask how
much of it is *low-order multiplicative*. Across three pretrained decoders (GPT-2,
Pythia-160m, llama-160m) and all twelve blocks of each, we find three things. **(1)
Linear recoverability is strikingly heterogeneous and jagged across depth** — some
blocks are near-perfectly linear (R²_lin > 0.99) and immediately-adjacent ones are
strongly nonlinear (< 0.3), with no monotone depth trend. **(2) It is not predicted by
the activation function.** GPT-2 and Pythia-160m are the same size with the same GELU
activation, yet have *opposite* depth profiles (GPT-2's early block is 95 % linear,
Pythia's only 45 %); llama's SwiGLU is different again. Two architecturally-identical
GELU models disagreeing means linear recoverability is a property of the *trained model
and block* — **learned, not architectural.** **(3) The residual is mostly not low-order
multiplicative.** A low-rank bilinear probe recovers only a few points of R² and — the
clean negative result — its gain does *not* scale with residual nonlinearity (Pearson
r ≈ 0 across blocks), so the unrecovered computation is genuinely high-order, not a
single product a position-wise layer could capture. The measurement has a direct
practical payoff: **per-block linear recoverability is a targeted-compression
criterion.** Where a block scores high, it can be replaced by a single position-wise
layer at a large parameter saving — GPT-2's early FFN by one linear map at **×8 fewer
parameters for +0.77 perplexity** — and the recoverability map tells you precisely which
blocks admit this and which do not. Underlying all of this is a methodological caution: transformer activations are ill-conditioned (outlier features),
so a *trained* linear baseline can be badly under-converged and overstate nonlinearity
by tens of points of R² — our own first runs did exactly this. The exact closed-form
baseline removes the confound and is what makes per-block linear recoverability a
trustworthy, reproducible measurement.

## 1. Introduction

Transformer FFNs are interpreted as key–value memories and as a primary store of a
model's learned computation. They are also expensive: for width *d*, the standard
`d → 4d → d` FFN is ≈ 8*d*² multiply-accumulates per token. Both for compression and
for interpretability, a basic question is how much of that computation is *additive*
(captured by a single linear map) versus *genuinely nonlinear* (requiring products of
input features). Rather than argue about it, we **measure** it.

We treat each FFN block as a position-wise map *g* : ℝ^d → ℝ^d (input *x* to output *y*)
and decompose it as

> *g(x) = W\*x + b\* + ρ(x)*,

where *W\*x + b\** is the **best least-squares linear approximation** of *g* over the
block's own activation distribution and *ρ* is the residual. The fraction of output
variance the linear part explains on held-out activations is the block's **linear
recoverability** R²_lin; **residual nonlinearity** is 1 − R²_lin. Crucially the linear
term can be solved in *closed form* (exact least squares), so R²_lin is an exact,
optimiser-free, reproducible property of the block — a measurement instrument rather
than a trained approximation.

We then ask what the residual *is*. We fit an explicit low-rank bilinear layer ("poly":
a linear term plus a rank-*r* degree-2 interaction) on top of the optimal linear map and
read its gain over the ceiling as how much of the residual is *low-order multiplicative*.
Multiplication here is a **probe of the residual**, not the headline.

Applying this to all twelve blocks of three pretrained decoders yields a sharp picture
of FFN structure (and a cautionary tale about how to measure it).

**Contributions.**

1. **Linear recoverability as a measurable per-block property.** A simple,
   cheap, exact (closed-form) distillation protocol that factors the transformer out of
   the fitting loop (it is used once, inference-only, as a feature extractor), making
   full per-block structural surveys feasible on a 6 GB GPU.
2. **A per-block, per-model survey showing linearity is heterogeneous, jagged across
   depth, and *not* predicted by the activation function.** GPT-2 and Pythia-160m
   (same size, same GELU) have opposite depth profiles; we argue linear recoverability
   is a learned property of the specific trained block, not an architectural one. A
   reduced-rank-regression and per-feature analysis further shows a high recoverability
   comes in two structurally opposite kinds — *low-rank, outlier-concentrated* (effective
   rank ≈ 1) versus *broadly linear* (effective rank in the hundreds) — even at equal
   R²_lin (§5.4).
3. **A clean characterisation of the residual.** A low-rank bilinear probe recovers
   only a few points of R², and its gain does not correlate with residual nonlinearity
   (Pearson r ≈ 0 across 36 blocks) — so the residual is genuinely high-order, not a
   single recoverable product. Multiplicative form is neither necessary nor sufficient
   for recoverability.
4. **A methodological caution for activation distillation.** Transformer activations are
   ill-conditioned (outlier features), so a *trained* linear baseline can be badly
   under-converged and overstate a target's nonlinearity by tens of points of R²; the
   exact closed-form least-squares baseline removes the confound and should be standard.
   (Indeed an earlier version of this work reported a spurious "early FFN is strongly
   nonlinear" result that was entirely this artifact; we document it as a worked
   warning, §5.1.)
5. **A targeted-compression payoff.** The same per-block measurement is a practical
   criterion for *where* an FFN can be cheaply replaced: linearly-recoverable blocks
   distil into a single position-wise layer at 4–8× parameter reduction with negligible
   perplexity cost (GPT-2 early: ×8 for +0.77 PPL), while low-recoverability blocks flag
   themselves as not safely compressible this way. Because the survey is heterogeneous,
   the gain is a *selective* one — compress the recoverable blocks, leave the rest.

## 2. Background and Related Work

- **FFNs as key–value memory** (Geva et al., 2021) motivates asking what kind of
  computation FFNs perform and how compressible it is — the structural question we
  measure here per block.
- **Conditioning / outlier features.** Transformer activations carry large-magnitude
  outlier features (Dettmers et al., 2022) that make their covariance ill-conditioned;
  this is what under-converges a naively-trained linear baseline and motivates the exact
  closed-form baseline that makes linear recoverability a trustworthy measurement (§5.1).
- **Knowledge / activation distillation.** We distil a sub-module's I/O map rather than
  logits; related in spirit to layer-wise distillation and to model "stitching." Our
  contribution is to use *exact* linear distillation as a measurement of structure, not
  only as a compression method.
- **Low-rank and bilinear layers.** The poly probe is a low-rank bilinear (degree-2
  polynomial) layer (Rendle, 2010); related to factorized bilinear pooling and to gated
  linear units. We use it to interrogate the residual, not to win a compression race.
- **Higher-order / multiplicative units.** Product units and sigma-pi neurons
  (Rumelhart et al., 1986; Shin & Ghosh, 1991; Jayakumar et al., 2020) compute
  multiplicative interactions; the poly probe is one such form. (A single log-space
  geometric-product layer we also tried proved numerically unstable on these targets and
  is deferred to future work, §8.)
- **Gated FFNs (GLU/SwiGLU).** LLaMA-style FFNs `(xW_gate ⊙ act(xW_up))W_down`
  (Shazeer, 2020) are *explicitly* multiplicative; we include one (llama-160m). A central
  finding is that this nominal multiplicativity does *not* make the residual recoverable
  by a single low-order product (§5.3).
- **Relation to concurrent work (this author).** A companion study (Whipp, in
  preparation) examines a multiplicative *hypernetwork* whose recruitment gate diagnoses
  multiplicative structure in a weight-generation map; there the target *did* benefit from
  a product layer — in direct contrast to the FFN-residual targets here. Taken together,
  the two results sharpen the same point: the multiplicative benefit is target-specific,
  not generic. (This paper is self-contained; the comparison is context, not a dependency.)

## 3. Method

### 3.1 Activation-space distillation

For a chosen block we run a corpus through the frozen model under `no_grad` and tap the
block's `.mlp` submodule with a forward hook, caching one (input *x* ∈ ℝ^d, output
*y* ∈ ℝ^d) row per token. We then fit a single position-wise layer *f* to minimise
‖*f*(*x*) − *y*‖² on a held-out split. The transformer is never back-propagated through;
all learning happens on the small standalone layer.

### 3.2 The linear/residual decomposition and the candidates

We instantiate the decomposition *g(x) = W\*x + b\* + ρ(x)* with three single-layer
candidates at width *d* = 768, matched on parameter budget:

- **linear (closed-form)** — `nn.Linear(d, d)`, solved in **closed form** by least
  squares: the exact best linear approximation *W\*x + b\**, i.e. the **linear ceiling**.
  This is the measurement instrument, not a trained model. ≈ *d*² params/MACs.
- **poly** — `PolyLinear`: a linear term plus a rank-*r* bilinear (degree-2) term
  (*r* = 16 ≪ *d*), ≈ *d*² + 2*dr* params. Used as a **probe of the residual**: how much
  of *ρ* is low-order multiplicative. (For the residual-gain analysis of §5.3 we freeze
  its linear branch at the closed-form optimum and train only the bilinear branch with
  held-out early stopping, so its gain over the ceiling is ≥ 0 by construction and
  isolates exactly what a low-rank product adds.)
- **dense (2×)** — `Linear → GELU → Linear` bottleneck, ≈ 2*d*² params: a depth control
  — the same budget spent on an extra additive hidden layer instead of a product term.

The original FFN is ≈ 8*d*² params (4.72 M for *d* = 768), so the single-layer
candidates are 4–8× compressions. Compression is both the lens (a high-fidelity single
layer *is* the evidence the block is structurally simple) and a practical payoff in its
own right (§6): on a recoverable block the single linear layer is a near-lossless ×8
parameter cut. *(A single log-space geometric-product "Sigma-Pi" layer was
also tried and excluded: it was numerically unstable and never beat the linear ceiling
on these FFN-residual targets — see §8.)*

### 3.3 Metrics

- **Linear recoverability** R²_lin — held-out R² of the closed-form linear map (variance
  explained); **residual nonlinearity** = 1 − R²_lin. Reported per block, exact.
- **Residual recovery** — R²_poly − R²_lin, the held-out R² a low-rank bilinear adds on
  top of the optimal linear map (the residual-gain probe, §5.3).
- **Effective rank** (§5.4) — by *reduced-rank regression*: the smallest *k* whose
  rank-*k* least-squares map reaches 90 % of the full closed-form R². Measures how many
  directions the linear map uses (the raw weight spectrum is uninformative — outlier-scale
  dominated — so we use RRR, not an SVD of *W\**).
- **Per-feature R²** (§5.4) — median over the *d* output features of the closed-form fit;
  an unweighted companion to the variance-weighted R²_lin (which a few high-variance
  outlier features can flatter).
- **Activation fit (worked examples)** — mean per-row **cosine** and **RMSE** (raw units,
  comparable within a block) alongside R², in the two-block detail tables (§5.5).
- **Conjunction index** (occlusion AND-signature): how multiplicatively the response
  collapses under occlusion of two disjoint feature halves — ≈ 0 for an additive map.
- **Recruitment gate** `exp(quad_scale)` and its drift over training (poly only) — how
  much multiplicative branch the fit actually recruits.
- **End-to-end perplexity.** We re-insert each fitted layer into the live model and
  measure WikiText-2 perplexity: (a) **zero-shot ΔPPL** (swap, nothing else changed);
  (b) **healed ΔPPL** after fine-tuning *only the swapped layer* for a small budget; and
  (c) a **heal-original baseline** — the original FFN given the *same* heal budget — so
  healed numbers are read against an equally-adapted original (see §5.6).

### 3.4 Compute / complexity

Per token the original FFN costs ≈ 8*d*² MACs (the 4× inner expansion). The single-layer
replacements collapse this: **linear** and **poly** are ≈ *d*² (poly = *d*² + 2*dr* ≈
1.04*d*² at *r* = 16), an ≈ 8× FLOP reduction; **dense (2×)** is ≈ 2*d*². All are O(*d*²)
versus the FFN's O(*d*²) with an 8× larger constant. The poly probe adds only the small
rank-*r* bilinear term over a plain linear map, so wherever a linear map already suffices
it is nearly free.

## 4. Experimental Setup

- **Models:** three pretrained decoders at matched width/depth (*d* = 768, 12 blocks):
  **GPT-2** (124 M, GELU FFN), **Pythia-160m** (GPT-NeoX, GELU FFN) — a second,
  independently-trained GELU model that lets us test whether linearity is a GELU property
  — and **llama-160m** (JackFram, SiLU **SwiGLU** FFN, intermediate 3072) for a different
  FFN family. (A **TinyLlama-1.1B** SwiGLU model is reported as a *scale* probe — see
  below — not as a fourth survey datapoint.)
- **Blocks:** all twelve blocks for the depth survey (§5.1–5.4); an *early* block
  (index 1) and a *deep* block (index 10) for the worked-example fidelity and perplexity
  tables (§5.5–5.6).
- **Corpus:** WikiText-2-raw (Merity et al., 2017). ~15–30 k token rows captured per
  block (seq len 128); held-out 20 % for fit metrics; perplexity on the test split.
  None of the models were trained on Wikipedia, so absolute PPL is higher than each
  model's headline number — we report *deltas* against each model's own fixed base.
- **Fit:** the linear baseline is solved in **closed form** (exact least squares — the
  linear ceiling); the poly / 2-layer candidates are trained with AdamW (converged,
  verified against the ceiling). For the §5.3 residual probe poly's linear branch is
  frozen at the closed-form optimum and only its bilinear branch is trained, with
  held-out early stopping.
- **Heal:** 200 steps, lr 1e-4, swapped-layer-only (and original-only for the per-block
  baseline).
- **Seeds:** 42 / 43 / 44; stochastic fits report mean ± std. The closed-form linear map
  is deterministic and reported as a single number. *Note on what the ±std captures:* the
  cached activations `(X, Y)` are identical across seeds (a `no_grad` eval-mode forward
  pass is deterministic) and the train/val split is a fixed tail, so a seed varies only
  layer initialisation and minibatch order. The very small spreads (typically ±0.000–0.001
  R²) therefore measure *optimisation reproducibility*, not corpus- or split-sampling
  uncertainty — they say the converged fits are stable, not that the numbers are immune to
  a change of data. We probe the larger, more honest perturbations directly in §5.7: a
  blocked *k*-fold cross-validation that gives the ceiling a real data-split CI (mean std
  0.024 R²), and changes of corpus *domain*.

## 5. Results

### 5.1 Linear recoverability across depth and models

Our central measurement is the per-block linear ceiling R²_lin across all twelve blocks
of all three models (Fig. *depth*). Two things stand out immediately.

**Linearity is jagged and non-monotone across depth, and model-specific.** GPT-2's
recoverability by block runs 0.79, 0.95, **0.996**, 0.65, 0.58, 0.40, 0.30, **0.25**,
0.22, 0.27, 0.67, 0.71 — near-perfectly linear at block 2, strongly *nonlinear* through
the middle (blocks 5–9 sit at ~0.2–0.4), then partially linear again at the end. Pythia
runs 0.92, 0.45, 0.66, **0.96**, 0.72, 0.74, 0.71, 0.78, 0.82, 0.84, 0.86, **0.99** —
rising toward the deep end. llama runs 0.96, 0.47, 0.59, **0.98**, 0.39, 0.37, 0.42,
0.44, 0.50, 0.50, 0.56, 0.90 — linear at the ends, nonlinear in the middle. There is no
universal "early-nonlinear / deep-linear" rule; each model has its own jagged profile,
and individual blocks can be almost perfectly linear right next to strongly nonlinear
ones (GPT-2 block 2 at 0.996, block 8 at 0.22).

**A cautionary methodological note (why the closed-form baseline).** An earlier version
of this study *trained* the linear baseline (3 000 SGD steps) and measured GPT-2's early
block at R² ≈ 0.25, which looked like strong early-block nonlinearity and a large
multiplicative advantage. That was an **optimisation artifact**: GPT-2's FFN activations
are severely ill-conditioned (input covariance condition number ≈ 3×10⁷; one output
feature carries ~100× the median variance — the transformer outlier-feature phenomenon,
Dettmers et al., 2022), so a plain linear layer needs ~15× more steps to converge. The
*exact closed-form* map scores 0.95 on that same block. Every R² and ΔPPL in this paper
therefore uses the closed-form linear ceiling (immune to the optimiser); the factorized
poly layer's main *practical* merit is that it self-conditions and reaches the ceiling
in ~3 000 steps where a plain linear layer needs ~50 000. Reporting the exact
least-squares ceiling is what makes per-block linear recoverability a trustworthy,
reproducible measurement rather than a statement about one's training budget.

### 5.2 Linearity is learned, not architectural

Is linear recoverability set by the activation function? It is not. **GPT-2 and
Pythia-160m are the same size (d = 768, 12 blocks) with the same GELU activation, yet
have opposite depth profiles.** Closed-form linear ceilings (single deterministic value;
zero-shot linear-swap ΔPPL in parentheses, against each model's own base):

| Block | GPT-2 (GELU) | Pythia-160m (GELU) | llama-160m (SwiGLU) |
|---|---:|---:|---:|
| early (1) | **0.95** (+1.1 PPL) | **0.45** (+63 PPL) | **0.47** (+19 PPL) |
| deep (10) | 0.67 (+5.5 PPL) | 0.86 (+12 PPL) | 0.56 (+8.8 PPL) |

GPT-2's early block is 95 % linear and a linear swap is nearly free (+1.1 PPL); the
*same-size, same-GELU* Pythia early block is only 45 % linear and a linear swap is
catastrophic (+63 PPL). Two architecturally identical GELU models disagree on which
blocks are linear and by how much — so "GELU FFNs are near-linear" is false as a general
claim. **Linear recoverability is a property of the specific trained model and block,
not of its architecture or activation function — it is learned, not architectural.** The
near-linear GPT-2 early FFN that an earlier draft built its whole story on is an outlier,
not a rule. (llama's SwiGLU, despite being *explicitly* multiplicative in form, is also
genuinely nonlinear in the early block — its nominal multiplicativity does not make it
more linearly recoverable; §5.3 shows it does not make the residual recoverable either.)

### 5.3 Is the residual multiplicative? A low-rank bilinear probe

Having measured the linear part exactly, we ask what the *residual* (1 − R²_lin) is: is
it low-order multiplicative, the kind of thing a single bilinear layer could recover? We
freeze poly's linear branch at the closed-form optimum and train only its rank-16
bilinear branch with held-out early stopping, so the gain R²_poly − R²_lin is ≥ 0 by
construction and measures exactly what a low-order product adds on top of the optimal
linear map. Plotting this gain against residual nonlinearity for all 36 blocks (Fig.
*residual-gain*) gives a clean negative answer.

**The residual is mostly not low-order multiplicative.** The bilinear probe recovers
only a few points of R² almost everywhere (median gain < 0.01; max 0.13), and — the key
result — **its gain does not scale with residual nonlinearity**: across all 36 blocks the
Pearson correlation between (1 − R²_lin) and the poly gain is **+0.004**, i.e. none. A
block being far from its linear ceiling tells you nothing about whether a product layer
can recover it: llama block 4 (residual 0.61) gains only +0.004, while llama block 1
(residual 0.53, *less* nonlinear) gains +0.078; GPT-2's most nonlinear blocks (residual
> 0.7) gain ≤ 0.008. The handful of blocks where the probe helps at all (Pythia block 2
+0.126, llama block 1 +0.078, Pythia block 7 +0.036) are idiosyncratic, not the
most-nonlinear ones. Within Pythia alone there is a mild positive trend (r = +0.46,
driven by blocks 1–2), but it does not hold for GPT-2 (+0.05) or llama (−0.01). **So the
residual computation FFNs perform is genuinely high-order — not a single product a
position-wise multiplicative layer can capture — and "the FFN is multiplicative in form"
(SwiGLU's gate) neither makes the block more linearly recoverable nor its residual more
bilinearly recoverable.** This is the cleanest disconfirmation in the paper of the idea
that multiplicative form predicts multiplicative recoverability.

Consistent diagnostics: the closed-form linear map reads conjunction index 0.000 (purely
additive, as it must); the 2-layer GELU bottleneck reads high (≈0.90 — genuinely
conjunctive); poly sits in between (≈0.31). And on a near-linear target poly's
multiplicative-recruitment gate *de-recruits* during fitting (exp(quad_scale) 0.135 →
0.055) — the optimiser actively shrinks the quadratic branch when there is little
multiplicative structure to recruit, agreeing with the near-zero gains above.

### 5.4 What kind of linearity? Effective rank and per-feature recoverability

Linear recoverability R²_lin is variance-weighted, so a high value can arise two very
different ways. To tell them apart we add two scale-aware structural readouts of the same
closed-form fit: the **effective rank** of the linear map by *reduced-rank regression*
(the smallest *k* whose rank-*k* least-squares map reaches 90 % of the full closed-form
R² — a metric-careful answer to "how many directions does the map use?", since the raw
weight spectrum is dominated by outlier-feature scale, §6) and the **per-feature R²**
(median over the *d* output features, unweighted by variance). Fig. *kinds* plots the two
against each other for all 36 blocks.

**A high R²_lin hides two structurally opposite regimes.**

- *Low-rank, outlier-concentrated* (effective rank ≈ 1–6, per-feature R² low). GPT-2 block
  2 has R²_lin = 0.996 but a per-feature median R² of only **0.16** and an effective rank
  of **1** — almost all of its *variance-weighted* recoverability lives in a single
  dominant output direction, and the median individual feature is barely linear. The same
  pattern holds for GPT-2 block 1 (0.95 / 0.26 / rank 1), Pythia block 3 (0.96 / 0.63 /
  rank 1) and block 11 (0.99 / 0.90 / rank 4), and llama blocks 3 and 11.
- *High-rank, broadly linear* (effective rank ≈ 190–380, per-feature R² high). Pythia
  block 0 has R²_lin = 0.92, a per-feature median of **0.89**, and an effective rank of
  **376** — genuinely linear across most features. llama block 0 (0.96 / 0.68 / rank 193)
  and GPT-2 block 0 (0.79 / 0.63 / rank 202) are likewise broadly linear.

So **R²_lin and effective rank are largely decoupled**: high recoverability occurs both at
rank 1 and at rank ~380. This *reconciles* rather than undercuts §5.1: GPT-2 block 2's
0.996 is real in the variance-weighted, downstream-relevant sense — a linear swap costs
only +0.77 PPL (§5.6) precisely because the residual stream's high-variance direction is
the one that is linear — even though most of the block's features are not individually
linear. The effective rank says what *kind* of linear object the block is: a low-rank
block behaves like a single (or few) linear "key–value" memory direction(s) (cf. Geva et
al., 2021) and would compress further still — to a near-rank-1 linear map — whereas a
high-rank block needs the full *d*² budget. The decomposition also sharpens "learned, not
architectural" (§5.2): Pythia block 0 (broadly linear, per-feature 0.89) and GPT-2 block 1
(outlier-linear, per-feature 0.26) have nearly the same R²_lin yet opposite internal
structure, so even the *form* of a block's linearity is a learned, per-block property.
(Because effective rank is measured by variance-weighted RRR it inherits R²_lin's
outlier-weighting; the per-feature median R² is its unweighted complement, and we read the
two together.)

### 5.5 Worked example: two blocks in detail (fidelity)

The depth survey (§5.1–5.3) is the evidence; this section zooms in on one early and one
deep block of two contrasting models to show the underlying fit quality (cosine, RMSE,
parameters/compression). **These two blocks are illustrative, not representative — §5.1
shows recoverability is jagged across depth, so no two-block pick is a summary of a
model.** The `linear` row is the exact closed-form ceiling (single value); poly /
dense (2×) / poly (2×) are trained to convergence (mean ± std over seeds 42/43/44; RMSE
omitted for the `poly (2×)` rows, which come from the separate depth-control run). *(These tables
use the higher-token multi-seed runs, ~30 k tokens/block, so the closed-form ceilings
differ slightly from the 15 k-token depth survey of §5.1–5.2 — e.g. GPT-2 deep 0.74 here
vs 0.67 in the survey; the difference is sampling of the activation distribution, not the
map, and is well within the cross-corpus robustness discussed in §7.)*

**GPT-2 (GELU), original FFN 4,722,432 params.**

| Block | Layer | Params | Compress | R² | Cosine | RMSE |
|---|---|---:|---:|---:|---:|---:|
| early (1) | linear (closed-form) | 590,592 | ×8.0 | 0.954 | 0.650 | 0.342 |
| early (1) | poly | 628,224 | ×7.5 | 0.956 ± 0.000 | 0.659 ± 0.000 | 0.335 |
| early (1) | dense (2×) | 1,181,184 | ×4.0 | 0.960 ± 0.000 | 0.693 ± 0.001 | 0.321 |
| early (1) | poly (2×) | 1,256,448 | ×3.8 | 0.961 ± 0.000 | 0.708 ± 0.000 | — |
| deep (10) | linear (closed-form) | 590,592 | ×8.0 | 0.736 | 0.845 | 1.465 |
| deep (10) | poly | 628,224 | ×7.5 | 0.746 ± 0.000 | 0.848 ± 0.000 | 1.437 |
| deep (10) | dense (2×) | 1,181,184 | ×4.0 | 0.768 ± 0.000 | 0.864 ± 0.000 | 1.373 |
| deep (10) | poly (2×) | 1,256,448 | ×3.8 | 0.768 ± 0.000 | 0.863 ± 0.000 | — |

**llama-160m (SwiGLU), original FFN 7,077,888 params.**

| Block | Layer | Params | Compress | R² | Cosine | RMSE |
|---|---|---:|---:|---:|---:|---:|
| early (1) | linear (closed-form) | 590,592 | ×12.0 | 0.402 | 0.588 | — |
| early (1) | poly | 628,224 | ×11.3 | 0.475 ± 0.001 | 0.625 ± 0.001 | — |
| early (1) | dense (2×) | 1,181,184 | ×6.0 | 0.483 ± 0.001 | 0.692 ± 0.001 | — |
| early (1) | poly (2×) | 1,256,448 | ×5.6 | 0.483 ± 0.002 | 0.695 ± 0.001 | — |
| deep (10) | linear (closed-form) | 590,592 | ×12.0 | 0.527 | 0.711 | — |
| deep (10) | poly | 628,224 | ×11.3 | 0.539 ± 0.001 | 0.718 ± 0.001 | — |
| deep (10) | dense (2×) | 1,181,184 | ×6.0 | 0.585 ± 0.001 | 0.744 ± 0.001 | — |
| deep (10) | poly (2×) | 1,256,448 | ×5.6 | 0.588 ± 0.000 | 0.747 ± 0.000 | — |

On GPT-2's near-linear early block all candidates land together at R² ≈ 0.95–0.96:
poly adds +0.002, dense (2×) +0.006 — the residual is small and not bilinearly
recoverable (§5.3). On llama's genuinely-nonlinear early block the ceiling is 0.40 and
the spread is wider (poly +0.073, dense (2×) +0.081), but even the 2-layer additive
control recovers only a fraction of the residual — consistent with the survey's verdict
that the residual is high-order and only partly reachable by any single position-wise
layer, multiplicative or not.

**Multiplicative depth ≈ additive depth.** The `poly (2×)` row (PolyLinear → GELU →
PolyLinear, the multiplicative analog of `dense (2×)`) lets us ask whether *adding
multiplicativity to a two-layer candidate* helps beyond the depth itself. It does not:
poly (2×) matches dense (2×) to within ≤0.002 R² at every block (GPT-2 early 0.961 vs
0.960, GPT-2 deep 0.768 vs 0.768, llama early 0.483 vs 0.482, llama deep 0.588 vs
0.585) — and at a slightly *larger* budget, so the tiny edge is within noise. The gain of
the two-layer candidates over a single layer is therefore the **depth** (an extra hidden
nonlinearity), not the **multiplicativity**: once a hidden layer is present, making its
projections explicitly bilinear adds essentially nothing. This is the depth-axis
counterpart of §5.3's width-axis result — neither multiplicative *width* (poly's bilinear
term) nor multiplicative *depth* (poly (2×)) recovers the FFN residual that an equal
budget of plain additive capacity does not.

**Scale probe (TinyLlama-1.1B SwiGLU, *d* = 2048).** As an external-validity check at
≈9× the width we probe two blocks of TinyLlama-1.1B (SwiGLU, 12× larger FFN, 34.6 M
params; base PPL 19.8) with *scale-aware* fitting (the closed-form-seeded poly of §5.3
and gradient-clipped training, which keep the candidates stable where a naive lr-1e-3 fit
diverges at this width). The fits are stable but the signal is weak: the closed-form
linear ceiling is only R² ≈ 0.04–0.07 *globally*, and the **per-feature** R² is actually
*negative* (median −0.02 early, −0.21 deep) — the median output feature is predicted
worse than its own mean, so the small global R² is propped up by a few high-variance
features rather than a broadly good fit. poly adds nothing over linear (≤0.001) and
dense (2×) only a little (to 0.09/0.25 global), with zero-shot swaps costing +2.9–5.1 PPL.
We therefore treat TinyLlama as *directional only* — it confirms that a billion-parameter
SwiGLU FFN strongly resists single-position-wise-layer distillation (consistent with, and
stronger than, llama-160m) but is not a clean survey datapoint. Methodologically it also
shows the value of the per-feature R² as a stricter companion to the variance-weighted
global R², which here is the *optimistic* reading.

### 5.6 Downstream perplexity: R² and ΔPPL dissociate

Re-inserting each fitted layer into the live model and measuring WikiText-2 perplexity
reveals that **activation-fit R² and downstream perplexity impact measure different
things.** From the depth survey:

- **High R² does not imply low ΔPPL.** llama block 0 has R²_lin = 0.96 yet a linear swap
  costs **+76 PPL**; block 3 has R²_lin = 0.98 yet costs **+40 PPL**. Pythia block 0
  (R² 0.92) costs +52 PPL. Early blocks are **perplexity-critical** almost regardless of
  how linearly fittable they are — a near-perfect activation fit can still wreck the
  model because the residual stream is sensitive to small early perturbations.
- **The criticality is model-specific.** GPT-2 block 0 (R² 0.79) costs only +2.6 PPL,
  while llama / Pythia block 0 cost +76 / +52 — the same depth, very different downstream
  fragility.
- **Where the multiplicative probe helps downstream, it is on these critical blocks.**
  Although poly's *R²* gain is tiny and uncorrelated with residual nonlinearity (§5.3),
  its *ΔPPL* benefit is concentrated on early, perplexity-critical blocks: llama block 0
  +76 → +43, block 3 +40 → +12; Pythia block 1 +63 → +33, block 2 +56 → +27 — and ≈ 0 on
  blocks 4–11. Strikingly, llama block 0 gets a 33-point PPL reduction from poly while its
  R² barely moves (+0.002), underlining that R² is a poor proxy for downstream impact and
  that both should be reported.

The two-block detail (§5.5 models, base PPL GPT-2 64.25 / llama 41.17) confirms the
fidelity ordering downstream:

| Model | Block | Layer | ΔPPL (zero-shot) | ΔPPL (healed) |
|---|---|---|---:|---:|
| GPT-2 | early (1) | linear (closed-form) | **+0.77** | −6.92 |
| GPT-2 | early (1) | poly | **+0.50 ± 0.01** | −11.77 ± 0.05 |
| GPT-2 | early (1) | dense (2×) | **+0.58 ± 0.04** | −17.25 ± 0.07 |
| GPT-2 | early (1) | *orig (healed)* | — | **−20.70 ± 0.06** |
| GPT-2 | deep (10) | linear (closed-form) | +4.70 | +2.02 |
| GPT-2 | deep (10) | poly | +4.65 ± 0.05 | −5.77 ± 0.20 |
| GPT-2 | deep (10) | dense (2×) | +4.50 ± 0.06 | −14.36 ± 0.11 |
| GPT-2 | deep (10) | *orig (healed)* | — | **−20.39 ± 0.01** |
| llama | early (1) | linear (closed-form) | +16.21 | −5.64 |
| llama | early (1) | poly | +14.52 ± 0.06 | −6.52 ± 0.09 |
| llama | early (1) | dense (2×) | +13.83 ± 0.18 | −6.16 ± 0.11 |
| llama | early (1) | *orig (healed)* | — | **−16.01** |

**The healing confound and the heal-original control.** Healed ΔPPL is negative for most
candidates — a *healed* single layer can score below stock GPT-2. This is not
"compression improves the model": healing fine-tunes the swapped layer on WikiText-2
*train*, but the stock models never saw Wikipedia, so the healed variant gains a sliver
of in-domain adaptation the base lacks, and more parameters means more headroom to absorb
it (why the wider dense2× heals "best"). We control with the **heal-original baseline** —
the *same* per-block FFN given the *same* heal budget, rest frozen (a per-block quantity,
not whole-model fine-tuning): −20.70 (GPT-2 early) and −20.39 (GPT-2 deep). The full
4.7 M-param FFN, healed identically, captures almost all the in-domain headroom; no
compressed candidate closes the gap (best, early dense 2× at −17.25, is ~3.5 PPL short).
So healing does not overturn the zero-shot story — what the small layers cannot recover
is precisely the extra capacity of the wide FFN. We treat zero-shot ΔPPL as the fidelity
metric and the (healed − heal-original) gap as a capacity probe; both agree.

### 5.7 Cross-domain robustness

Because R²_lin is measured over the activation distribution a corpus induces, a fair
robustness check is a *different-domain* corpus — not merely a larger same-domain one
(WikiText-103 is still Wikipedia). We re-run the survey on **two** further domains:
literary prose (Project Gutenberg, *Moby-Dick*) and mathematical/logical puzzles (Dudeney,
*Amusements in Mathematics*), each ~1 MB at the same 15 k-token budget, and compare the
per-block profile to WikiText-2 (Fig. *corpus*).

**The linear-recoverability profile is a property of the model, not the corpus.** Across
all three domains the per-block ceilings track tightly: Moby-Dick vs WikiText-2 gives
Pearson *r* = 0.97 / 0.99 / 0.95 (GPT-2 / Pythia / llama) and the math-puzzle corpus 0.97 /
0.98 / 0.97 (Spearman ≥ 0.87 throughout); the mean absolute shift in R² is only 0.03–0.07
(worst-case ≈ 0.20 on a single block). Every qualitative claim survives intact on both
out-of-domain corpora: the jagged non-monotone depth profiles (§5.1), the GPT-2-vs-Pythia
reversal that grounds "learned, not architectural" (§5.2), and each model's near-linear and
strongly nonlinear blocks. Absolute ceilings move a little — as they must, since the input
distribution changes — but the *shape* and the cross-model *contrasts* that carry the
paper's claims do not.

**The downstream and multiplicative findings are corpus-robust too.** Repeating the full
depth sweep (closed-form ceiling + seeded-poly + zero-shot ΔPPL, with its own held-out
train/test split) on *Moby-Dick* reproduces both secondary results of §5.6: the **R²–ΔPPL
dissociation** (llama block 0 has R²_lin = 0.95 yet a linear swap costs **+72 PPL**; block 3
R²_lin = 0.98 yet +57 PPL; Pythia block 2 +31) and **poly's ΔPPL benefit concentrated on the
early perplexity-critical blocks** (llama block 0 +72→+40, block 3 +57→**+17**; Pythia block
2 +31→+15; ≈ 0 elsewhere). So the linear *and* the multiplicative/downstream stories hold on
out-of-domain text.

**A data-split (not just seed) confidence interval.** Finally, to give the ceiling a variance
over *data* rather than the deterministic seed-std of §4, we run blocked 5-fold
cross-validation of the closed-form ceiling (contiguous folds, since adjacent token windows
are correlated). The fold-to-fold std is small — mean 0.024, max 0.062 R² across all 36
blocks — and the well-recovered blocks are the tightest (GPT-2 block 2: 0.996 ± 0.000), while
the larger spreads sit on the low-recoverability blocks (GPT-2 block 10: 0.434 ± 0.062), as
expected. This is the honest companion to the near-zero seed spreads (§4): a genuine
data-split CI, an order of magnitude larger than the seed-std but still small enough that
every conclusion stands.

## 6. Discussion

- **Linear recoverability is heterogeneous, learned, and model-specific — not set by the
  activation function.** Across 36 blocks of three models the linear ceiling ranges from
  ~0.2 to >0.99 with no monotone depth trend, and two same-size GELU models (GPT-2,
  Pythia-160m) have opposite profiles. Which FFN blocks reduce to a single linear map is
  a property of the trained network, not of "GELU vs SwiGLU." This is the paper's main
  empirical message and the reason a careful (closed-form) baseline matters.
- **The residual is genuinely high-order — multiplicative form does not predict
  recoverability.** A low-rank bilinear probe recovers only a few points of R² and its
  gain is uncorrelated with how nonlinear the block is (Pearson r ≈ 0). An explicitly
  multiplicative SwiGLU block is no more linearly recoverable, and its residual no more
  bilinearly recoverable, than a GELU one. What a single position-wise layer cannot
  capture is real, distributed, high-order computation — not a missing product term. The
  same holds along the *depth* axis: a two-layer *multiplicative* candidate (poly (2×))
  matches a two-layer *additive* one (dense (2×)) to within ≤0.002 R² everywhere (§5.5),
  so what little the 2-layer candidates recover is the extra hidden nonlinearity, not the
  multiplicativity.
- **R² and downstream ΔPPL measure different things.** A near-perfect activation fit
  (R² 0.96) can still cost +76 PPL (llama block 0); early blocks are perplexity-critical
  largely independent of fittability, and the multiplicative probe's *downstream* value,
  where it exists, lives on exactly those critical blocks. Studies of FFN compression
  should report both.
- **Closed-form baselines are essential for activation distillation.** Because
  transformer activations are ill-conditioned (outlier features), an under-converged
  trained linear baseline can overstate nonlinearity by tens of points of R² and an order
  of magnitude of ΔPPL — exactly the artifact that produced an earlier, wrong version of
  this paper's central claim. The exact least-squares ceiling removes the confound and
  should be standard practice.
- **"Recoverable" comes in two structurally opposite kinds.** A high R²_lin can mean a
  *low-rank, outlier-concentrated* linear block (GPT-2 block 2: R²_lin 0.996 but effective
  rank 1 and per-feature median R² only 0.16 — one dominant linear direction, tying it to
  the large-magnitude outlier features of Dettmers et al., 2022) or a *high-rank, broadly
  linear* block (Pythia block 0: R²_lin 0.92, effective rank 376, per-feature 0.89). Since
  R²_lin and effective rank are decoupled (§5.4), the variance-weighted recoverability we
  headline should be read together with effective rank and per-feature R² to know *what
  kind* of linear object a block is — the low-rank blocks resemble single linear key–value
  memory directions (Geva et al., 2021) and compress further still. (The raw weight
  spectrum alone is uninformative here — it is scale-dominated by the outlier feature — so
  we measure rank by reduced-rank regression, not by SVD of *W\**.)

## 7. Limitations

Base models at the small end (GPT-2, Pythia-160m, llama-160m) and modest corpora
(WikiText-2, Gutenberg prose, and a math/logic puzzle corpus — all English); a
TinyLlama-1.1B scale probe but no large-model survey; perplexity on a
capped test slice; healing introduces an in-domain adaptation confound addressed but not
eliminated by the heal-original control. The residual probe is a *single* low-rank
bilinear form — a different basis (higher rank, other nonlinearities) might recover more
of the residual; our negative result is specific to low-order bilinear recovery, which is
the natural first probe. The closed-form baseline is exact only for the *linear*
candidate; trained candidates are verified converged against it but could be optimised
further.

*On corpus.* We measured on three domains — WikiText-2 (encyclopedic), Gutenberg prose,
and a math/logic puzzle corpus — and the per-block profile is highly stable across all
three (Pearson 0.95–0.99, §5.7), so the **linear ceiling is largely a property of the FFN
map itself**, sampled by — not determined by — the input distribution. A still-larger or
more varied corpus (WikiText-103, OpenWebText, source code) would mainly tighten absolute
perplexity and reduce the in-domain healing confound rather than move the profile, but our
domains are all English natural-language text. A broader *model* sweep would do more than a
larger corpus to chart which trained networks (and which blocks) admit single-layer linear
distillation.

## 8. Future Work

- A larger model sweep (more architectures and scales) to map how the
  linear-recoverability profile varies — and whether "learned, not architectural" holds
  at scale.
- **Higher-*degree* single-polynomial layers.** Our `poly` probe is degree-2 (a sum of
  low-rank bilinears); since §5.3 shows the residual is not degree-2-recoverable, the
  natural next probe raises the *degree* within one position-wise layer — a degree-3+
  factorized polynomial (higher-order factorization machines; Blondel et al., 2016)
  rather than *stacking* layers (the additive depth of `dense (2×)` / the multiplicative
  depth of `poly (2×)`, §5.5). This separates whether the residual is higher-order
  *product* structure (recoverable by raising the degree) from genuinely non-polynomial
  computation (recoverable only by an added nonlinearity).
- **Mechanism behind the low-rank / broadly-linear split (§5.4).** We *measure* that some
  recoverable blocks are near-rank-1 (outlier-concentrated) and others full-rank; *why* a
  given block lands in one regime — and whether the low-rank blocks correspond to
  identifiable key–value memories (Geva et al., 2021) or to specific token/feature roles —
  is open. A weighted/whitened effective rank (RRR is variance-weighted, so it inherits the
  outlier-weighting) would complement the per-feature view.
- Richer residual bases beyond polynomials (small kernel / feature maps); multi-block
  span distillation (replace several FFNs at once); and a TinyLlama / larger-SwiGLU
  external-validity check with scale-aware fitting.
- **A genuine higher-order / geometric-product (Sigma-Pi) layer — deferred, with
  tempered expectations.** A single log-space weighted product was numerically unstable
  on these targets and, once stabilised, still added no value over the linear ceiling —
  unsurprising given §5.3: a *single monomial* is an even more restrictive form than the
  sum-of-bilinears that already fails to recover the residual, so we do not expect a
  stable reformulation to help on FFN residuals specifically. Its likely value lies
  elsewhere — on targets with genuine low-order *product* structure (bilinear attention
  scores, explicit gating, or the weight-generation hypernetwork of our concurrent work
  (Whipp, in preparation), where the recruitment gate *did* fire) rather than on the
  high-order FFN residual here.

## 9. Reproducibility

All code is in the `polyweave` library. Per-block fitting is
`polyweave/experiments/gpt2_mlp_distill.py`; the depth survey (§5.1–5.3, Fig. *depth*) is
`run_depth_sweep.py` (writes `plots/raw/depth_sweep_wikitext2.{json,csv}`; re-plot
without recompute via `plot_depth_sweep.py`); the residual-gain probe (§5.3, Fig.
*residual-gain*) is `run_residual_gain_clean.py` (frozen-linear, quad-only, early-stopped;
writes `plots/raw/residual_gain_clean.{json,csv}`); the effective-rank / per-feature
analysis (§5.4, Fig. *kinds*) is `run_rrr_rank.py` (closed-form + reduced-rank regression,
no training; re-plot via `plot_rrr_rank.py`, writes `plots/raw/rrr_rank.{json,csv}`); the
cross-domain robustness (§5.7, Fig. *corpus*) is `run_corpus_robustness.py` (closed-form
ceilings on Moby-Dick / math-puzzle corpora; overlay via `plot_corpus_robustness.py`),
`run_depth_sweep_gutenberg.py` (full poly + ΔPPL sweep on Moby-Dick with its own
train/test split), and `run_kfold_ceilings.py` (blocked 5-fold data-split CI); the two-block detail and perplexity
(§5.5–5.6) are driven multi-seed by `run_gpt2_multiseed_v2.py`, `run_pythia_multiseed_v2.py`,
and `run_llama_multiseed_v2.py` (shared Config/protocol, differing only in `model_name`);
the multiplicative-vs-additive depth control (§5.5, `poly (2×)`) is `run_poly2x.py`; the
TinyLlama-1.1B scale probe (§5.5, scale-aware closed-form-seeded poly + gradient-clipped
training + per-feature R²) is `run_tinyllama_scaleaware.py`.
The linear baseline is the exact closed-form least-squares solution
(`linear_closed_form=True`); trained candidates use 8 000 AdamW steps. Optional deps
install via `pip install polyweave[distill]` (`transformers`, `datasets`); WikiText-2 is
cached to plain text on first fetch. Runs target a single 6 GB GPU.

## References

*(Working list — details to be verified and converted to a `.bib` for the LaTeX
submission. Grouped by the role they play in the paper.)*

**Transformer feed-forward structure.**
- Geva, M., Schuster, R., Berant, J., & Levy, O. (2021). *Transformer feed-forward
  layers are key-value memories.* EMNLP.
- Shazeer, N. (2020). *GLU variants improve transformer.* arXiv:2002.05202. (SwiGLU.)

**Conditioning / outlier features** (motivating the closed-form linear baseline).
- Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). *LLM.int8():
  8-bit matrix multiplication for transformers at scale.* NeurIPS. (Documents the
  large-magnitude outlier features that make transformer activations ill-conditioned —
  the cause of the under-converged linear fit this paper corrects.)

**Higher-order / multiplicative units and low-rank bilinear layers.**
- Rumelhart, D. E., Hinton, G. E., & McClelland, J. L. (1986). *A general framework
  for parallel distributed processing.* In *Parallel Distributed Processing*, Vol. 1.
  MIT Press. (Introduces sigma-pi units.)
- Shin, Y., & Ghosh, J. (1991). *The Pi-Sigma network: an efficient higher-order
  neural network for pattern classification and function approximation.* IJCNN.
- Jayakumar, S. M., Czarnecki, W. M., Menick, J., Schwarz, J., Rae, J., Osindero, S.,
  Teh, Y. W., Harley, T., & Pascanu, R. (2020). *Multiplicative interactions and where
  to find them.* ICLR.
- Rendle, S. (2010). *Factorization machines.* ICDM. (Low-rank bilinear / the `poly`
  probe.)
- Blondel, M., Fujino, A., Ueda, N., & Ishihata, M. (2016). *Higher-order factorization
  machines.* NeurIPS. (Degree-≥3 factorized polynomials — the higher-degree single-layer
  probe proposed in §8.)

**Models & data.**
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019).
  *Language models are unsupervised multitask learners.* (GPT-2.)
- Biderman, S., et al. (2023). *Pythia: a suite for analyzing large language models
  across training and scaling.* ICML. (Pythia-160m, GPT-NeoX.)
- Touvron, H., et al. (2023). *LLaMA: open and efficient foundation language models.*
  arXiv:2302.13971. (SwiGLU decoder; the llama-160m secondary model follows this
  architecture.)
- Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2017). *Pointer sentinel mixture
  models.* ICLR. (WikiText-2.)

**Prior work (this author).**
- Whipp, S. (in preparation). *When does the Pi branch fire?* (Multiplicative
  hypernetwork; recruitment-gate diagnostic carried into FFN distillation here. Companion
  to the present paper; cited as context, not a dependency.)
