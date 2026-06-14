# Logic Gates as Neurons

PolyWeave's `logic` module gives you the Boolean operators as **differentiable fuzzy
gates** — they reduce to exact truth tables on `{0, 1}` and interpolate smoothly in
between, so you can drop them into a network and train through them. The default
**product t-norm** makes a fuzzy AND a literal product neuron, which is the bridge
between logic and the multiplicative computation this library is built around.

**On this page:**

- [The gates](#the-gates)
- [AND is a product — and XOR needs one](#and-is-a-product-and-xor-needs-one)
- [One multiplicative neuron learns XOR](#one-multiplicative-neuron-learns-xor)
- [The radial-basis route](#the-radial-basis-route)

## The gates

Truth values are tensors in `[0, 1]`. Every gate works elementwise:

```python
import torch
from polyweave.logic import fuzzy_and, fuzzy_or, fuzzy_xor

for a in (0.0, 1.0):
    for b in (0.0, 1.0):
        ta, tb = torch.tensor(a), torch.tensor(b)
        print(int(a), int(b), "|",
              f"AND={fuzzy_and(ta, tb):.0f}",
              f"OR={fuzzy_or(ta, tb):.0f}",
              f"XOR={fuzzy_xor(ta, tb):.0f}")
```

```text
0 0 | AND=0 OR=0 XOR=0
0 1 | AND=0 OR=1 XOR=1
1 0 | AND=0 OR=1 XOR=1
1 1 | AND=1 OR=1 XOR=0
```

The full set is `fuzzy_not`, `fuzzy_and`, `fuzzy_or`, `fuzzy_nand`, `fuzzy_nor`,
`fuzzy_xor`, `fuzzy_xnor` — plus parameter-free `nn.Module` versions (`FuzzyAnd`,
`FuzzyOr`, …) for use in `nn.Sequential`. Each takes a `t_norm` of `"product"`
(default) or `"min"`.

Because they're smooth, *graded* inputs give graded answers:

```python
h = torch.tensor(0.5)
fuzzy_and(h, h).item()   # 0.25  (= 0.5 * 0.5)
fuzzy_xor(h, h).item()   # 0.5
```

## AND is a product — and XOR needs one

With the product t-norm, `fuzzy_and(a, b) = a * b`. That's not an analogy — it's
literally a product (Pi) neuron, the multiplicative primitive at the core of
PolyWeave. From it, XOR falls out as:

```
xor(a, b) = or(a, b) − and(a, b) = (a + b − ab) − ab = a + b − 2ab
```

A linear term **plus a bilinear product** — i.e. exactly a degree-2 (Sigma-Pi)
neuron. This is why XOR is the textbook example a single linear unit *cannot* solve
but a single multiplicative one can.

## One multiplicative neuron learns XOR

[`PolyLinear`](../api/layers.md) is a linear branch plus a low-rank bilinear branch —
so a single rank-1 `PolyLinear(2, 1)` has exactly the `ab` term XOR requires. Train it
against a plain `nn.Linear(2, 1)` on the four XOR points:

```python
import torch, torch.nn.functional as F
from polyweave import PolyLinear

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([[0.], [1.], [1.], [0.]])

def fit(model, steps=4000, lr=0.05):
    torch.manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad(); F.mse_loss(model(X), Y).backward(); opt.step()
    return F.mse_loss(model(X), Y).item()

print("PolyLinear:", fit(PolyLinear(2, 1, rank=1)))   # -> 0.00000
print("nn.Linear: ", fit(torch.nn.Linear(2, 1)))      # -> 0.25000
```

```text
PolyLinear: 0.0        preds = [0.0, 1.0, 1.0, 0.0]   ✓ solved
nn.Linear:  0.25       preds = [0.5, 0.5, 0.5, 0.5]   ✗ stuck at the mean
```

The linear neuron collapses to predicting `0.5` everywhere (MSE `0.25` is the best a
hyperplane can do on XOR); the multiplicative neuron nails it. That single bilinear
term is the whole difference — the same multiplicative capacity the
[Concepts](../concepts.md) page describes, shown on the smallest possible problem.

## The radial-basis route

There's a second, non-multiplicative way to crack XOR: the radial-basis activation
[`radbas`](../api/ops.md), `exp(-(εx)²)`, which peaks when its input is near zero. Feed
it `a − b` and it fires when the inputs *agree* (XNOR), so `1 − radbas(a − b)` is XOR:

```python
from polyweave import radbas

for a in (0., 1.):
    for b in (0., 1.):
        xor = 1 - radbas(torch.tensor(a - b), epsilon=10.0)
        print(int(a), int(b), round(xor.item(), 3))
```

```text
0 0 0.0
0 1 1.0
1 0 1.0
1 1 0.0
```

Two routes to the same non-linearly-separable problem: an explicit **product**
(Sigma-Pi / poly) or a **distance-to-a-prototype** bump (radial basis). PolyWeave
gives you both as small, composable, differentiable pieces.
