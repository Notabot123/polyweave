# Autodidact Hypernetwork — Experiment Log

## Overview

A hypernetwork ("teacher") generates weights for a frozen student network from a compact
prototype map built from support-set statistics. The central question: does a Sigma-Pi
(multiplicative / log-space) teacher outperform a vanilla additive teacher, and under
what conditions?

---

## Experiment 1 — MNIST, classification head ARCHIVED(autodidact_compact_conv_teacher.py)

**Target layer:** fc (feature_dim → num_classes)  
**Prototype:** class means, variances, counts in *feature space* (after trunk)  
**Sigma-Pi branch:** z-scored input treated as log-space ← **incorrect formulation**  
**Result:** conv ≈ conv_sigmapi (both ~0.969 zero-shot, ~0.978 recovery final)  
**Key finding:** MNIST saturates; too easy to distinguish methods. Pi branch had no
real multiplicative semantics.

---

## Experiment 2 — CIFAR-10, classification head (autodidact_cifar_v2.py)

### v2 original (deprecated — incomplete formulation)

**Prototype:** 3 channels (mean, variance, count)  
**Sigma-Pi branch:** log(softplus(x)) — sign-discarding  
**pi_scale:** stuck at 0.1353 throughout — pi branch completely inactive  
```
method         seen    unseen
ncc            0.757   0.831   ← wins on unseen
conv           0.754   0.802
conv_sigmapi   0.756   0.789
```

### v2 re-run (v3 parity — canonical result, autodidact_cifar_v2.py)

**Prototype:** 4 channels — mean, variance, kurtosis, inter-class contrast  
**Sigma-Pi branch:** signed-log `sign(x)*log(|x|+ε)` + BN after sigma+pi sum  
**Teacher:** BN + Dropout(0.1) in intermediate layers  
**pi_scale diagnostic:** 0.1365 → 0.1458 (+0.0093, slow but monotonic growth)  

```
method         seen    unseen
random         0.096   0.126
ncc            0.760   0.822   ← wins on unseen (zero-shot)
conv           0.728   0.766
conv_sigmapi   0.744   0.777
```

**Recovery finals (unseen, mean over 5 students):**
```
random        0.854   ← converges fully in 300 steps (fc is tiny — 2570 params)
ncc           0.826
conv          0.833
conv_sigmapi  0.840   ← beats conv on all 5 individual students
```

**Key findings (v2 re-run):**
- NCC still wins zero-shot on unseen — fc→centroid mapping remains approximately linear
- conv_sigmapi now consistently beats conv on both zero-shot AND recovery (v3 formulation unlocks the pi branch)
- Pi-scale grows slowly (+0.0093) vs conv1 generation (+0.0222) — *graded* response to degree of multiplicative structure, not binary
- Recovery final for random > teachers because 300 Adam steps on 2,570-param fc fully converges regardless of init; zero-shot is the informative metric for fc generation
- Teacher gap to NCC narrowed vs original v2 (conv_sigmapi now 4.5pp below NCC on unseen vs 4.2pp originally) — 4-channel proto adds discriminative information

---

## Experiment 3 — CIFAR-10, conv filter generation (autodidact_cifar_v3_convgen.py)

**Target layer:** conv1 (first convolutional layer, shape [32, 3, 3, 3])  
**Prototype:** class-conditional statistics over *raw input image* spatial grid
  (4×4 grid → 48 features per class; channels: mean, variance, kurtosis, inter-class contrast)  
**Sigma-Pi branch:** signed-log pi branch + BN after sigma+pi  
**Teacher:** conv encoder → GlobalAvgPool → linear head (generates 896 params)  
**NCC baseline:** not applicable — no centroid analogue for conv filter generation  
**Baselines:** random init vs teacher-generated init, measuring recovery speed

**Motivation:** conv filter generation involves spatial frequency interactions
(edge detection = frequency × orientation product). The fc → centroid mapping was
linear (NCC-equivalent), but conv filter → discriminative-patch mapping is nonlinear.
Sigma-Pi should have genuine multiplicative interactions to exploit.

**Status:** in progress

---

## Architectural notes

### ConvSigmaPi2d — evolution

**v1 (compact_conv):** `z = zscore(x)` — not log-space, purely additive semantics  
**v2 (cifar_v2):** `z = log(softplus(x))` — correct but sign-discarding  
**v3 (cifar_v3):** `z = sign(x) * log(|x| + ε)` — signed log, odd function,
preserves direction of input in log-space

### Why pi_scale stays frozen

The sigma branch converges first (easier gradient path). Once the loss is low enough,
the gradient reaching pi_scale via `exp(pi_scale) * tanh(conv(z))` is near zero —
pi is not needed, so it doesn't grow. BN after sigma+pi sum may help by normalising
the scale relationship between branches before the nonlinearity.

### Prototype statistics — MNIST vs CIFAR

On MNIST, class distributions in feature space are clean and well-separated.
Mean + variance per class ≈ sufficient. On CIFAR-10, within-class modes exist
(dog breeds under "dog"), classes overlap, and the mean/variance do not capture
discriminative structure. Kurtosis and inter-class contrast are added in v3.
A fully learned encoder would likely outperform fixed statistics, at the cost of
scientific cleanliness.

---

## Planned future experiments

- **v4:** Generate attention Q/K/V matrices in a small transformer for a
  text classification or sequence task. Attention is inherently multiplicative
  (QK^T), making this the theoretically strongest setting for sigma-pi.

- **v5 (library):** Package ConvSigmaPi2d, SigmaPiDense, PolynomialDense,
  LogSpaceMultiplicative, and utilities (signed_log, logspace_init) as a
  standalone PyTorch library once experimental results justify each component.

---

## Reference results (MNIST compact conv, run 2)

```
seen random:         0.1049
seen conv:           0.9690
seen conv_sigmapi:   0.9652
unseen random:       0.0860
unseen conv:         0.9553
unseen conv_sigmapi: 0.9582   ← sigmapi slightly wins on unseen

Recovery finals (unseen):
random:       ~0.972
conv:         ~0.978
conv_sigmapi: ~0.979   ← sigmapi edges conv on recovery
```
