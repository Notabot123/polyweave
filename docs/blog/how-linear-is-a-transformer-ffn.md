# How Linear Is a Transformer's Feed-Forward Block?

A transformer block has two halves: attention, which mixes information *across*
tokens, and a feed-forward network (FFN), which transforms each token *in place*. The
FFN is the bigger half — roughly two-thirds of the block's parameters — and it's widely
read as where a model stores much of its learned computation.

So here's a simple structural question that turns out to have a surprising answer: **how
much of that feed-forward block is just a linear map?**

Not "is it linear" — of course there's a nonlinearity bolted in the middle. The question
is how much of the block's actual input→output behaviour, on the activations it really
sees, a single matrix multiply could reproduce. We measured it across three pretrained
models, and the headline is this: **linear recoverability is learned, not architectural.**
Two models with the *identical* architecture and activation function can disagree
completely about which of their blocks are linear.

This post walks through how to measure it yourself with PolyWeave, what we found, and one
trap that nearly fooled us.

**On this page:**

- [The measurement: linear recoverability](#the-measurement-linear-recoverability)
- [Why *closed-form*, and not a trained probe](#why-closed-form-and-not-a-trained-probe)
- [What we found](#what-we-found)
- [The payoff: a compression criterion](#the-payoff-a-compression-criterion)
- [What's next](#whats-next)
- [Try it out yourself](#try-it-out-yourself)

## The measurement: linear recoverability

Treat an FFN block as a position-wise function `g(x) = y`, mapping a `d`-dimensional
token to a `d`-dimensional token. Decompose it as

> `g(x) = W*·x + b* + ρ(x)`

where `W*·x + b*` is the **best least-squares linear approximation** of the block over its
own activation distribution, and `ρ(x)` is whatever's left over. The fraction of held-out
output variance the linear part explains is the block's **linear recoverability**, which
we write `R²_lin`. The rest, `1 − R²_lin`, is its **residual nonlinearity**.

The key move is that `W*` can be solved in **closed form** — exact least squares, no
training loop, no learning rate to tune. That makes `R²_lin` an exact, reproducible
*property of the block*, not an artefact of how long you trained a probe.

In PolyWeave this is two steps: capture the block's `(input, output)` pairs as it runs,
then fit the closed-form linear map.

```python
import torch
import torch.nn as nn
from polyweave.distill import fit_closed_form_linear

# A toy "block": a linear map plus a genuinely nonlinear residual. The (X**2 - 1)
# term is centred, so no linear map can absorb it — it shows up as residual.
torch.manual_seed(0)
d, N = 64, 8_000
X = torch.randn(N, d)
W_true = torch.randn(d, d) / d**0.5
M2 = torch.randn(d, d) / d**0.5
Y = X @ W_true.T + 0.25 * (X**2 - 1.0) @ M2

linear = nn.Linear(d, d)
result = fit_closed_form_linear(linear, X, Y)
print(f"R2_lin = {result.val_r2:.3f}")   # 0.882 — mostly linear, ~12% genuine residual
```

`fit_closed_form_linear` solves the affine map on a training split and reports held-out
metrics on a fixed tail split, so the number is honest about generalisation, not just
training fit.

To run it on a *real* FFN, capture activations with `IOCapture` — a forward hook that
flattens every token into one regression row:

```python
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from polyweave.distill import IOCapture, fit_closed_form_linear

model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
tok = AutoTokenizer.from_pretrained("gpt2")

block = model.transformer.h[1].mlp                      # GPT-2's early FFN sub-block
with IOCapture(block, max_rows=30_000) as cap:
    for text in corpus:                                 # any iterable of strings
        ids = tok(text, return_tensors="pt").input_ids[:, :128]
        with torch.no_grad():
            model(ids)                                  # forward passes drive the hook
X, Y = cap.pairs()                                      # [N, 768], [N, 768]

linear = nn.Linear(X.shape[1], Y.shape[1])
print(fit_closed_form_linear(linear, X, Y).val_r2)      # ≈ 0.95 for GPT-2 block 1
```

!!! note "Install the distillation extra"
    The real-model snippet needs `transformers`: `pip install -e ".[distill]"`. The toy
    example above needs only the core library.

## Why *closed-form*, and not a trained probe

This is the part that nearly fooled us, and it's the reason the measurement is
trustworthy. Transformer activations are **ill-conditioned**: a handful of "outlier"
feature directions carry enormous variance compared to the rest. On data like that, a
plain `nn.Linear` trained with Adam can sit badly under-converged for *tens of thousands*
of steps — and an under-fit linear baseline makes the block look far more nonlinear than
it is.

Our own first runs did exactly this. A GPT-2 block that a short-trained linear probe
scored at **R² ≈ 0.25** is recovered by the *exact* closed-form map at **R² ≈ 0.95**. The
entire "this block is nonlinear" story was an optimisation artefact.

| Linear baseline | What it measures | Failure mode |
|---|---|---|
| Trained probe (SGD/Adam) | How good a linear map you found *in your budget* | Under-converges on outlier-heavy activations; overstates nonlinearity by tens of R² points |
| **Closed-form least squares** | The **exact** best linear map | None — it's the true ceiling, deterministic and reproducible |

The lesson generalises: if you're probing representations with a *trained* linear model,
make sure it has actually converged, or solve it in closed form and skip the doubt.

## What we found

Across three pretrained decoders — GPT-2, Pythia-160m, and llama-160m — and all twelve
blocks of each, three things stood out.

**1. Linear recoverability is jagged across depth.** There's no tidy "early layers
nonlinear, late layers linear" trend. GPT-2's per-block `R²_lin` runs roughly:

```
block:   1     2     3     4     5     6     7     8   ...
R²_lin: 0.79  0.95  0.996 0.65  0.58  0.40  0.30  0.25 ...
```

Near-perfectly linear blocks sit right next to strongly nonlinear ones.

**2. It's not predicted by the activation function — it's learned.** This is the clean
result. GPT-2 and Pythia-160m are the same size with the same GELU activation, yet their
depth profiles are *opposite*:

| Early block (1) | GPT-2 (GELU) | Pythia-160m (GELU) | llama-160m (SwiGLU) |
|---|---|---|---|
| Linear recoverability `R²_lin` | **0.95** | **0.45** | **0.47** |

Same architecture, same nonlinearity, opposite answer. Whether a block is linearly
recoverable is a property of the *trained weights*, not the design.

**3. The residual is mostly *not* a simple product.** Having measured the linear part
exactly, we probed the leftover `ρ(x)` with an explicit low-rank bilinear layer
(PolyWeave's `PolyLinear`) to ask: is the residual low-order *multiplicative* — the kind
of thing a single position-wise product could capture? Mostly, no. The bilinear probe
recovers only a few points of R², and — the clean negative result — its gain **does not
scale** with how nonlinear the block is (Pearson `r ≈ 0` across 36 blocks). The
unrecovered computation is genuinely high-order, not one missed multiplication.

!!! tip "Make the depth plot"
    The per-block recoverability curve is the figure to look at. Once you have an
    `R²_lin` per block, it's a few lines of matplotlib:

    ```python
    import matplotlib.pyplot as plt

    blocks = range(1, 13)
    plt.plot(blocks, gpt2_r2lin,   "o-", label="GPT-2 (GELU)")
    plt.plot(blocks, pythia_r2lin, "s-", label="Pythia-160m (GELU)")
    plt.xlabel("FFN block"); plt.ylabel("linear recoverability  R²_lin")
    plt.legend(); plt.tight_layout(); plt.savefig("recoverability.png", dpi=150)
    ```

    The two GELU curves crossing each other is the whole paper in one image.

## The payoff: a compression criterion

This isn't just diagnostics. Linear recoverability is directly **actionable for
compression**: where a block scores high, you can *replace* its whole FFN with a single
linear layer and barely move the model.

The standard GPT-2 FFN (`768 → 3072 → 768`) is about 4.7M parameters. Its early block is
95% linearly recoverable — so a single closed-form linear map stands in for it at:

| Replacement for GPT-2 early FFN | Parameters | Compression | Held-out R² | ΔPerplexity |
|---|---|---|---|---|
| Closed-form linear map | 590,592 | **×8.0** | 0.954 | **+0.77** |
| Low-rank bilinear (`PolyLinear`) | 628,224 | ×7.5 | 0.956 | +0.50 |

An ×8 parameter cut on that block for **+0.77 perplexity** — and the recoverability map
tells you *which* blocks admit this and which don't, before you spend any compute.

!!! warning "High R²_lin is necessary, not sufficient"
    One honest subtlety: a high recoverability score doesn't *guarantee* a cheap swap.
    A block can be 96% linearly recoverable in variance yet still cost tens of perplexity
    points when replaced, because the small residual lands on a direction the rest of the
    model is sensitive to. Measure `R²_lin` to find candidates; confirm with a perplexity
    check before trusting the swap.

## What's next

The tools used here — `fit_closed_form_linear`, `IOCapture`, `PolyLinear`, and the
metrics in `polyweave.distill` — are all in the library; see [Getting
Started](../getting-started.md) to install and the [Concepts](../concepts.md) page for the
multiplicative layers.

The bilinear probe in this post is a descendant of an earlier question: *when* does a
neural network actually recruit multiplication, and can you measure it as it trains? That's
the subject of the next post, on the pi-scale recruitment diagnostic — including an honest
account of what that diagnostic does and doesn't tell you, which is part of what motivated
the exact-measurement approach here.

## Try it out yourself

Run every snippet in this post in your browser — no install required:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Notabot123/polyweave-notebooks/HEAD?labpath=notebooks/how-linear-is-a-transformer-ffn.ipynb)

!!! note "Notebook repo coming soon"
    The companion `polyweave-notebooks` repository isn't published yet — the badge above
    points at its intended home. Until it lands, the snippets run as-is against an editable
    install (`pip install -e ".[distill]"`).
