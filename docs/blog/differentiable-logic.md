# Logic, in Products: Gates, Chaining, and Rule Induction

Most of a neural network is *addition* — weighted sums of features. But logic is
*multiplication*: a conjunction `A ∧ B` is true only when **both** hold, which is exactly
what a product `A · B` computes. PolyWeave leans into that correspondence, and it gives
you a small toolkit of **differentiable logic** built on the same multiplicative
primitive as its Sigma-Pi layers.

This post walks the toolkit from a single gate up to learning interpretable rules — every
number below was produced by running the code.

**On this page:**

- [Conjunction is multiplication](#conjunction-is-multiplication)
- [XOR in a single neuron](#xor-in-a-single-neuron)
- [Reasoning as a differentiable layer](#reasoning-as-a-differentiable-layer)
- [Inducing rules — with negation](#inducing-rules-with-negation)
- [What's next](#whats-next)
- [Try it out yourself](#try-it-out-yourself)

## Conjunction is multiplication

`polyweave.logic` gives you the Boolean operators as differentiable fuzzy gates over
truth values in `[0, 1]`. With the default **product t-norm**, a fuzzy AND is literally a
product neuron:

```python
import torch
from polyweave.logic import fuzzy_and, fuzzy_or, fuzzy_xor

a, b = torch.tensor(1.0), torch.tensor(0.0)
fuzzy_and(a, b)   # 0.0   (= a * b)
fuzzy_or(a, b)    # 1.0   (= a + b - a*b)
fuzzy_xor(a, b)   # 1.0   (= a + b - 2*a*b)
```

They're exact on the Boolean corners and interpolate smoothly between, so you can train
through them. The full set (`fuzzy_not/and/or/nand/nor/xor/xnor`, plus `nn.Module`
versions) lives in [the logic example](../examples/logic-gates.md).

## XOR in a single neuron

XOR is the textbook problem a single *linear* neuron can't solve. But `fuzzy_xor`
already hints at the fix: `a + b − 2ab` is a linear term **plus a product** — a degree-2
neuron. PolyWeave's [`PolyLinear`](../api/layers.md) (linear + low-rank bilinear) has
exactly that, so one rank-1 unit learns XOR while `nn.Linear` cannot:

```python
import torch, torch.nn.functional as F
from polyweave import PolyLinear

X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
Y = torch.tensor([[0.],[1.],[1.],[0.]])
# ... train each on the four points ...
```

```text
PolyLinear(2,1,rank=1):  MSE 0.000   preds [0, 1, 1, 0]   ✓
nn.Linear(2,1):          MSE 0.250   preds [0.5, 0.5, 0.5, 0.5]   ✗ stuck at the mean
```

One bilinear term is the whole difference. (`radbas` gives a second, radial-basis route
to the same problem — see the [logic example](../examples/logic-gates.md).)

## Reasoning as a differentiable layer

Scale conjunction up and you get inference. `polyweave.reasoning` runs **forward
chaining** over a propositional knowledge base — repeatedly firing rules (product-AND of
their premises) until the facts reach a fixpoint:

```python
from polyweave.reasoning import PropKB, ForwardChainer

kb = PropKB()
kb.add_rule(["raining"], "wet_grass")
kb.add_rule(["wet_grass"], "slippery")
kb.add_rule(["wet_grass", "sunny"], "rainbow")   # a conjunction

chainer = ForwardChainer(kb)
chainer.entails(kb.initial_facts(["raining"]), "slippery")   # (True, 1.0)
chainer.entails(kb.initial_facts(["raining"]), "rainbow")    # (False, 0.0)  -- needs sunny
```

Because every step is products and maxes, it's **differentiable in the facts** — the
derivative of `slippery` w.r.t. `raining`, computed *through* the two-hop proof, is
`1.000`. So this is a reasoning *layer* you can embed in a network, not just a solver.
For Horn clauses, chaining to the fixpoint is sound and complete for entailment — details
in the [forward-chaining example](../examples/forward-chaining.md).

## Inducing rules — with negation

A plain product-AND can only build *monotone* conjunctions of positive premises. But
attach one **signed exponent** per premise and the product becomes a learnable rule body
that the optimiser *induces* — negation included:

```
contribution_i = t_i ** [w_i]+ · (1 − t_i) ** [w_i]−
    w_i > 0  →  required     w_i = 0  →  ignored     w_i < 0  →  inhibitory (negated)
```

`SoftSignedLiteral` learns "fly ← bird ∧ ¬penguin" and you read the rule straight off the
exponents:

```python
from polyweave.logic import SoftSignedLiteral
# train on  fly = bird AND NOT penguin  ...
layer.literals(["bird", "penguin", "d2", "d3"])
# [('bird', 'required', +1.00), ('penguin', 'inhibitory', -0.99)]   distractors ~0
```

`SoftRuleLayer` ORs several of these into a soft DNF. On a 2-rule, *non-linearly-separable*
target `(bird ∧ ¬penguin) ∨ (bat ∧ ¬broken)`, it recovers both rules — where a linear
model can't represent the disjunction at all:

```text
nn.Linear                0.871     ← can't: DNF isn't linearly separable
soft rule layer (ours)   1.000  (16 params)   rules: "bird & not penguin", "bat & not broken"
MLP (hidden 32)          1.000  (321 params)  ← solves it, but a black box
```

Same accuracy as the MLP at ~5% of the parameters, and *interpretable*. The recruitment
diagnostic `exponent_abs_mean()` (the same one [Paper 1/2's layers expose](../concepts.md))
reads how much structured rule the layer has recruited.

!!! note "Honest placement"
    These are interpretable rule-learning layers in the lineage of Logical Neural Networks
    and RL-Net/DR-Net — PolyWeave offers them in its geometric-product / recruitment
    framing rather than claiming a new capability. They're handy, composable building
    blocks, not a research result.

## What's next

The pieces here — gates, `radbas`, forward chaining, soft-literal rule induction — make
PolyWeave a small *differentiable AI-maths* toolkit. See [Getting
Started](../getting-started.md) to install and the [Concepts](../concepts.md) page for the
multiplicative layers underneath.

## Try it out yourself

Run every snippet in your browser, no install required:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Notabot123/polyweave-notebooks/HEAD?labpath=notebooks/differentiable-logic.ipynb)

!!! note "Notebook repo coming soon"
    The companion `polyweave-notebooks` repo isn't published yet — the badge points at its
    intended home. Until then, the snippets run against an editable install (`pip install -e .`).
