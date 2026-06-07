# PolyWeave

**Multiplicative (Sigma-Pi) layers and hypernetworks for weight generation.**

PolyWeave is a small, modular PyTorch library for building *hypernetworks* — networks
that **generate the weights of another network** — with an emphasis on **multiplicative
(Sigma-Pi)** computation. Alongside a vanilla additive teacher, it provides a signed-log
Sigma-Pi branch whose recruitment can be *measured* via a single diagnostic, so you can
ask **when multiplicative interactions actually help** rather than assuming they do.

The library underpins an ongoing series of papers on multiplicative hypernetworks.
The first, *"When Does the Pi Branch Fire?"*, introduces the pi-scale recruitment
diagnostic and the graded-recruitment result; later work will build on the same
codebase. A proper citation block will be added here once the papers are public.

> **Status:** v0.1.0 — alpha. The API may still shift before a PyPI release.

---

## Key ideas

- **Signed-log Sigma-Pi branch.** A multiplicative branch operating in log-space,
  `z = sign(x)·log(|x| + ε)`, bounded with `tanh` and gated by a learnable
  `exp(pi_scale)` scalar. Additions in log-space are products in linear space.
- **The pi-scale diagnostic.** `exp(pi_scale).mean()` is a scalar read-out of *how much*
  the multiplicative branch has been recruited during training — a direct, interpretable
  measure of whether a task needs products.
- **Graded recruitment.** Across three weight-generation targets the pi branch is
  recruited in a consistent, monotonic order — least for a fully-connected head, more for
  a convolutional layer, most for attention Q/K — matching how much genuine multiplicative
  structure each target has.

---

## Installation

PolyWeave is not yet on PyPI; install from source:

```bash
git clone https://github.com/Notabot123/polyweave.git
cd polyweave
pip install -e .                 # core library (torch, matplotlib)
pip install -e ".[experiments]"  # + torchvision, to run the paper experiments
pip install -e ".[dev]"          # + pytest, to run the test suite
```

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

---

## Tests

```bash
pytest
```

The suite guards behavioural sanity (shapes, gradients, invariants), not bit-exact
reproduction of training runs.

---

## License

MIT © 2026 Stuart Whipp. See [LICENSE](LICENSE).
