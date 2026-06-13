# Concepts

PolyWeave's Sigma-Pi layers sum two branches: an ordinary **additive** branch and a
genuine **multiplicative** branch. This page describes what each computes in the
shipped layers ([`SigmaPiLinear`](api/layers.md) and
[`ConvSigmaPi2d`](api/layers.md)) and the diagnostics that read how much the
multiplicative branch is used.

## The two branches

For the dense layer, an output feature `j` is

```
y_j = sigma_j(x) + pi_j(x)
```

**Sigma (additive)** is a plain linear map over the feature-centred input — it carries
sign and behaves like `nn.Linear`:

```
sigma = Linear(x - mean(x))
```

**Pi (multiplicative)** is a *weighted geometric product* of the input magnitudes,
formed in log space and exponentiated back:

```
log_mag = log(|x| + eps)
log_mag = log_mag - mean(log_mag)          # geometric-mean normalisation
u       = sum_i w_ji * log_mag_i           # a linear map in log space
pi_j    = exp(pi_scale_j) * exp(clamp(u))  # = exp(pi_scale_j) * prod_i |x_i| ** w_ji
```

So `pi_j` is literally `exp(pi_scale_j)` times a product `∏_i |x_i| ** w_ji`. This is
what makes it a *genuine* higher-order neuron — it **exponentiates back to a product**.
(The deprecated `tanh(W·signed_log(x))` formulation never exponentiated, so it could not
represent a product; it is retained only as historical context.)

The convolutional block [`ConvSigmaPi2d`](api/layers.md) mirrors this exactly with
`Conv2d` in place of `Linear`, then applies `BatchNorm` + `ReLU` so it drops into a CNN
as a self-contained block.

## Design choices

Each choice is deliberately interpretable:

- **Bounded signed exponents.** The log-space weights are `w = max_exponent * tanh(raw)`
  (default `max_exponent = 0.5`), so each exponent lies in `(-0.5, +0.5)`. Signed
  exponents let the branch *amplify* (`|x| ** +w`) **or** *suppress / divide*
  (`|x| ** -w`); the `tanh` cap keeps any single factor finite and the log-space sum
  well-conditioned.
- **Geometric-mean normalisation.** Centring `log|x|` by its mean makes the product
  *relative* — scale-free in the overall input magnitude — so it does not blow up with
  input dimension, and the learnable gate alone sets its amplitude.
- **Clamp.** The accumulation is clamped to `[-max_log, +max_log]` (default `6`) before
  `exp` as an overflow guard.
- **Learnable amplitude gate.** `pi_scale` is a per-output gate initialised to `-2`
  (`exp(-2) ≈ 0.135`) so the pi branch starts subdominant.

**Sign handling.** The product is magnitude-only by default — the pi branch sees `log|x|`
and sign is carried by the signed sigma branch. A true signed product is available as the
flagged ablation `signed_products=True`.

**`center_product`.** With `center_product=True` the branch uses `expm1(u)` instead of
`exp(u)`, so it is exactly `0` when the exponents are `0` (the multiplicative identity):
the branch starts *silent* and `pi_scale` recovers a clean "volume knob" meaning. The
default `exp(u)` starts at `1` (a constant the bias / BatchNorm can absorb).

## Recruitment diagnostics

A central theme of the research is that you should *measure* whether a task recruits
multiplication, not assume it. Two complementary read-outs are exposed on both layers:

- **Metric A — `exponent_abs_mean()`** reads the *weights*: `mean(|max_exponent ·
  tanh(raw)|)`. Because `w = 0` makes a factor `|x| ** 0 = 1` (a no-op), this measures
  how far the learned product departs from doing nothing. It is meaningful even when the
  product starts at the identity.
- **Metric B — `branch_energy(x)`** reads the *activations*: the RMS of each branch and
  `pi_share = pi_rms / (sigma_rms + pi_rms)` on a batch. The conv block additionally
  reports `pi_effect_postbn`, the post-BatchNorm relative change when the pi branch is
  removed (the honest, normalisation-aware version of "how much does pi matter").

The historical gate read-out `pi_scale_mean() = exp(pi_scale).mean()` is also available.

!!! warning "Read the diagnostics together"
    `pi_scale` (gate amplitude) and `exponent_abs_mean` (product shape) answer different
    questions, and the gate's behaviour depends on the `center_product` choice. A high
    diagnostic indicates the layer *uses* a product; it is **not**, on its own, a
    guarantee that the product improves accuracy. Treat these as descriptive probes of
    structure, not as a performance score.
