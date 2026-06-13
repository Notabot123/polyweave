<p align="center">
  <img src="assets/logo.png" alt="PolyWeave — higher order networks" width="540">
</p>

# PolyWeave

**Multiplicative (Sigma-Pi) layers and hypernetworks for weight generation.**

PolyWeave is a small, modular PyTorch library for building *hypernetworks* — networks
that **generate the weights of another network** — with an emphasis on **multiplicative
(Sigma-Pi)** computation. Alongside a vanilla additive teacher, it provides Sigma-Pi
layers whose multiplicative branch is a *genuine geometric product*, plus diagnostics
that let you **measure when multiplicative interactions actually help** rather than
assuming they do.

The library underpins an ongoing series of papers on multiplicative hypernetworks.

!!! note "Status"
    v0.1.0 — alpha. The public API may still shift before it stabilises.

## Where to go next

- **[Getting Started](getting-started.md)** — install the library and run your first
  Sigma-Pi layer.
- **[Concepts](concepts.md)** — what the sigma and pi branches compute, and how the
  recruitment diagnostics read multiplicative structure.
- **[API Reference](api/layers.md)** — auto-generated documentation for every public
  module.
- **[Blog](blog/index.md)** — plain-language write-ups of the research.

## Quick taste

```python
import torch
from polyweave import ConvSigmaPi2d, SigmaPiLinear

# A channels-preserving Sigma-Pi conv block (additive + geometric-product branch).
block = ConvSigmaPi2d(channels=32, kernel_size=3)
block.eval()
y = block(torch.randn(8, 32, 28, 28))
print(block.pi_scale_mean())     # gate amplitude — how strongly pi is recruited
print(block.exponent_abs_mean()) # product shape — how far it departs from a no-op

# A Sigma-Pi fully-connected layer (raw pre-activation, drop-in for nn.Linear).
fc = SigmaPiLinear(in_features=128, out_features=64)
out = fc(torch.randn(8, 128))
```
