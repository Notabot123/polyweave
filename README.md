# PolyWeave

**Multiplicative (Sigma-Pi) layers and hypernetworks for weight generation.**

[![CI](https://github.com/Notabot123/polyweave/actions/workflows/ci.yml/badge.svg)](https://github.com/Notabot123/polyweave/actions/workflows/ci.yml)
[![Docs](https://github.com/Notabot123/polyweave/actions/workflows/docs.yml/badge.svg)](https://Notabot123.github.io/polyweave/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

PolyWeave is a small, modular PyTorch library for building *hypernetworks* — networks
that **generate the weights of another network** — with an emphasis on **multiplicative
(Sigma-Pi)** computation. Alongside a vanilla additive teacher, it provides Sigma-Pi
layers whose multiplicative branch is a *genuine geometric product*, together with
diagnostics that let you **measure when multiplicative interactions are actually used**
rather than assuming they help.

The library underpins an ongoing series of papers on multiplicative hypernetworks. The
first, *"When Does the Pi Branch Fire?"*, introduces the pi-scale recruitment diagnostic;
a follow-up studies how linear a transformer's feed-forward block really is by distilling
it into a single layer. A proper citation block will be added here once the papers are
public.

> **Status:** v0.2.0 — alpha. The API may still shift before a PyPI release.
> 📖 **Documentation:** <https://Notabot123.github.io/polyweave/>

---

## Key ideas

- **Genuine geometric-product pi branch.** The multiplicative branch forms a weighted
  product of the input *magnitudes* in log space and exponentiates back:
  `pi = exp(pi_scale) · ∏ᵢ |xᵢ| ^ wᵢ`, with bounded signed exponents
  `w = max_exponent · tanh(raw)` (so factors can amplify or divide), geometric-mean
  normalisation for scale-freeness, and a clamp for stability. The signed sigma (additive)
  branch carries sign. This *really multiplies* — unlike the deprecated `tanh(W·signed_log(x))`
  form, which never exponentiated back to a product.
- **Recruitment diagnostics.** Two complementary read-outs measure how much the product is
  used: `exponent_abs_mean()` reads the *weights* (how far the product departs from a
  no-op), and `branch_energy(x)` reads the *activations* (the pi branch's share of the
  output). The historical gate `pi_scale_mean() = exp(pi_scale).mean()` is also exposed.
- **Measurement, not assumption.** These diagnostics describe *whether a layer uses a
  product* — they are a probe of structure, not a guarantee of accuracy. PolyWeave is
  built to ask that question honestly across different targets. See the
  [Concepts](https://Notabot123.github.io/polyweave/concepts/) guide for details.

---

## Installation

PolyWeave is not yet on PyPI; install from source:

```bash
git clone https://github.com/Notabot123/polyweave.git
cd polyweave
pip install -e .                 # core library (torch, matplotlib)
pip install -e ".[experiments]"  # + torchvision, to run the paper experiments
pip install -e ".[distill]"      # + transformers, datasets, for the distillation study
pip install -e ".[dev]"          # + pytest, pytest-cov, to run the test suite
pip install -e ".[docs]"         # + mkdocs-material, mkdocstrings, to build the docs
```

Once published, installation will simply be `pip install polyweave`.

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

---

## Quickstart

The layers are drop-in `nn.Module`s. Each exposes the `pi_scale_mean` diagnostic.

```python
import torch
from polyweave import ConvSigmaPi2d, SigmaPiLinear

# A channels-preserving Sigma-Pi conv block (additive + signed-log multiplicative).
block = ConvSigmaPi2d(channels=32, kernel_size=3)
y = block(torch.randn(8, 32, 28, 28))
print(block.pi_scale_mean())   # how strongly the pi branch is recruited

# A Sigma-Pi fully-connected layer.
fc = SigmaPiLinear(in_features=128, out_features=64)
out = fc(torch.randn(8, 128))
print(fc.pi_scale_mean())
```

Building a weight-generating teacher, training it, and generating + installing weights
into a student is covered by the `hypernets`, `training`, `targets`, and `evaluation`
modules — see the experiment scripts below for end-to-end examples.

---

## Reproducing the paper experiments

The headline result is a cross-experiment "recruitment ordering" chart. Run the full
multi-seed suite (downloads CIFAR-10 automatically on first run):

```bash
python -m polyweave.experiments.multiseed
```

This trains the FC, conv1, and attention-Q/K teachers across seeds, writes the
aggregated numbers to `plots/multiseed_results.json`, and renders
`plots/polyweave_pi_ordering.{pdf,png}`. It also saves the seed-42 conv1 models to
`models/seed42/conv1_models.pt`, which the analysis scripts below reuse **without
retraining**:

```bash
python -m polyweave.experiments.ensemble          --seed 42   # ensemble diversity
python -m polyweave.experiments.student_occlusion --seed 42   # occlusion sensitivity
```

> **Note:** trained model payloads and downloaded datasets are *not* committed to the
> repo (they are large and fully regenerable). Run `multiseed` first to produce them.

Individual experiments can also be run on their own — see
`polyweave/experiments/` (`cifar_fc.py`, `cifar_conv1.py`, `synthetic_attention.py`).

---

## Project layout

```
polyweave/
  ops/             pure functions (signed-log, ...)
  layers/          nn.Module blocks: ConvSigmaPi2d, SigmaPiLinear, PolyLinear
  targets/         pack / unpack / install generated weights for a target layer
  prototypes/      compact support-set representations (statistical + learnable)
  students/        networks whose weights a teacher generates (CNN, transformer)
  hypernets/       full weight-generating teachers
  training/        generic teacher-training loop + checkpoint I/O
  evaluation/      zero-shot / recovery evaluation, ensembling
  interpretability/ occlusion sensitivity and related probes
  metrics.py       diagnostics (pi-scale, ensemble disagreement)
  viz/             publication-quality plotting (PDF, colourblind-safe palette)
  experiments/     runnable scripts reproducing the paper
```

One-off experiment drivers, plotting scripts, and paper drafts live under `research/`
(kept for provenance, excluded from the installed package). The shipped library is
everything under `polyweave/`.

---

## Tests

```bash
pytest
```

The suite guards behavioural sanity (shapes, gradients, invariants), not bit-exact
reproduction of training runs.

---

## License

Apache License 2.0 © 2026 Stuart Whipp. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
