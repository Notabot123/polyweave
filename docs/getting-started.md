# Getting Started

## Installation

PolyWeave targets Python ≥ 3.9 and PyTorch ≥ 2.0.

Until the first PyPI release, install from source:

```bash
git clone https://github.com/Notabot123/polyweave.git
cd polyweave
pip install -e .                 # core library (torch, matplotlib)
pip install -e ".[dev]"          # + pytest, pytest-cov (run the test suite)
pip install -e ".[experiments]"  # + torchvision (run the CIFAR experiments)
pip install -e ".[distill]"      # + transformers, datasets (GPT-2 / LLaMA distillation)
pip install -e ".[docs]"         # + mkdocs-material, mkdocstrings (build these docs)
```

Once published:

```bash
pip install polyweave
```

## Your first Sigma-Pi layer

The layers are drop-in `nn.Module`s. Each exposes the `pi_scale_mean` diagnostic.

```python
import torch
from polyweave import ConvSigmaPi2d, SigmaPiLinear

# Self-contained conv block: sigma + pi, then BatchNorm + ReLU.
block = ConvSigmaPi2d(channels=32, kernel_size=3)
block.eval()  # avoid BatchNorm batch-size-1 issues for a single forward
y = block(torch.randn(8, 32, 28, 28))
assert y.shape == (8, 32, 28, 28)

# Dense layer: emits the raw `sigma + pi` pre-activation (no baked-in nonlinearity),
# so it works as a regression head over continuous, possibly negative, targets.
fc = SigmaPiLinear(in_features=128, out_features=64)
out = fc(torch.randn(8, 128))
assert out.shape == (8, 64)
```

## Building a weight-generating teacher

The hypernetwork side of the library composes a few small pieces:

- **[`targets`](api/targets.md)** — pack / unpack / install generated weights for a
  particular layer type (FC, conv, attention Q/K, Sigma-Pi conv).
- **[`prototypes`](api/prototypes.md)** — compact representations of a support set
  (statistical summaries or a learnable encoder).
- **[`students`](api/students.md)** — the networks whose weights a teacher generates.
- **[`hypernets`](api/hypernets.md)** — the weight-generating teachers themselves.
- **[`training`](api/training.md)** — a generic teacher-training loop with checkpoint I/O.
- **[`evaluation`](api/evaluation.md)** — zero-shot / recovery evaluation and ensembling.

See `polyweave/experiments/` for runnable end-to-end examples, and the
[Concepts](concepts.md) page for what the Sigma-Pi branch actually computes.

## Running the tests

```bash
pytest                 # behavioural sanity: shapes, gradients, invariants
pytest --cov           # with a coverage report (needs the [dev] extra)
```
