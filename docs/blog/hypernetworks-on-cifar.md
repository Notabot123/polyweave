# Generating a Network's Weights with a Hypernetwork

Normally you get a classifier's weights one way: initialise them randomly and grind
gradient descent until they're good. But there's another option that feels almost like
cheating — train a *second* network to **look at a handful of labelled examples and write
the weights directly**, with no backprop on the target at all.

That second network is a **hypernetwork**. This post builds one with PolyWeave on CIFAR-10
and shows the three things a generated set of weights buys you: an **instant zero-shot
classifier**, a **faster-converging warm start** for fine-tuning, and a **cheap ensemble**.
No multiplicative-layer theory here — just the plain additive teacher and what you can do
with it.

**On this page:**

- [What's a hypernetwork, and why](#whats-a-hypernetwork-and-why)
- [The setup: a CNN student and an FC target](#the-setup-a-cnn-student-and-an-fc-target)
- [Leg 1: zero-shot weight generation](#leg-1-zero-shot-weight-generation)
- [Leg 2: a faster warm start](#leg-2-a-faster-warm-start)
- [Leg 3: a cheap ensemble](#leg-3-a-cheap-ensemble)
- [When is this worth it?](#when-is-this-worth-it)
- [What's next](#whats-next)
- [Try it out yourself](#try-it-out-yourself)

## What's a hypernetwork, and why

A hypernetwork is a network that outputs the *parameters* of another network. In PolyWeave's
setup, a **teacher** `g_w` reads a compact summary of a few-shot support set — the "prototype"
— and emits the weights of one **target layer** in a frozen **student**:

> `weights = g_w(prototype(support_set))`

The teacher is trained once over a distribution of tasks. At inference, you hand it a new
support set and it produces a usable layer — *no gradient steps on the student*. That's
useful three ways, which map exactly to the three legs below:

- **Zero-shot:** a working classifier in one forward pass.
- **Warm start:** a much better-than-random initialisation to fine-tune from.
- **Ensemble:** generate several heads cheaply and vote.

One honesty note up front, because it shapes how to read the results: for a simple *linear*
classification head, the parameter-free **nearest-centroid classifier (NCC)** is already a
strong baseline — class means make a good linear head on their own. So the bar isn't "beat
random," it's "match NCC instantly, then give me things NCC can't" (a fine-tunable init, and
diverse ensemble members). That's the story this post tells.

## The setup: a CNN student and an FC target

The student is a small CNN with a frozen convolutional trunk and a final fully-connected
(`fc`) classification head — the head is the layer the teacher generates. PolyWeave gives you
the pieces:

```python
import torch
from polyweave.students import make_cnn_student
from polyweave.prototypes import feature_class_stats
from polyweave.hypernets import FCMapTeacher

NUM_CLASSES, FEATURE_DIM = 10, 256

# A CNN student: frozen trunk + a generatable fc head.
student = make_cnn_student("A", feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES, in_ch=3)

# The teacher that writes the fc head's {weight, bias}. sigma_pi=False = the plain
# additive teacher (a small conv net over the prototype).
teacher = FCMapTeacher(NUM_CLASSES, FEATURE_DIM, proto_channels=4, width=64, sigma_pi=False)
```

The **prototype** is how the teacher "sees" a support set: per-class statistics of the
student's features. The student exposes `extract_features`, and `feature_class_stats` turns
features + labels into the prototype tensor the teacher consumes:

```python
def build_prototype(student, batch):
    x, y = batch
    feats = student.extract_features(x)            # [B, FEATURE_DIM]
    return feature_class_stats(feats, y, NUM_CLASSES)

# The forward callback ties a generated head into the student's forward pass.
def forward(student, batch, gen):
    x, y = batch
    return student(x, generated_fc=gen), y
```

Training the teacher is a single call over a population of students (`train_teacher` samples
tasks, builds prototypes, and supervises the generated head against real labels):

```python
from polyweave.training import train_teacher

result = train_teacher(
    teacher, train_students,
    sample_batch=lambda: next_support_batch(),     # your data sampler
    build_prototype=build_prototype, forward=forward,
    steps=5000, lr=1e-3,
)
```

!!! note "The full CIFAR population is wired for you"
    Building a *population* of student architectures (train several, freeze trunks,
    warm-restart heads) is the fiddly part. PolyWeave ships it as a runnable experiment —
    `python -m polyweave.experiments.cifar_fc` does the end-to-end CIFAR-10 version, and the
    companion notebook (bottom of the post) runs it cell by cell. The snippets here are the
    public API that experiment is built from.

## Leg 1: zero-shot weight generation

With a trained teacher, generating a head is one call. `generate_averaged` builds the
prototype from a few support batches, averages to reduce variance, and returns
`{"weight", "bias"}`. Then `evaluate_accuracy` runs the student with that head — no training:

```python
from polyweave.evaluation import (
    generate_averaged, evaluate_accuracy,
    class_centroids, centroids_to_fc, random_like,
)

gen = generate_averaged(teacher, student, support_batches, build_prototype)
acc = evaluate_accuracy(student, eval_batches, gen, forward)

# Two reference points, computed from the same support set:
centroids = class_centroids(support_feats, support_labels, NUM_CLASSES)
ncc  = evaluate_accuracy(student, eval_batches, centroids_to_fc(centroids), forward)  # strong
rand = evaluate_accuracy(student, eval_batches, random_like(gen), forward)            # floor
```

What you'll see: the generated head lands **far above the random floor** and **about level
with NCC**. That's the honest result — the teacher has *learned to reproduce a strong
baseline from feature statistics*, instantly, for architectures it was trained on. The payoff
over NCC isn't raw zero-shot accuracy; it's the next two legs.

## Leg 2: a faster warm start

A generated head is also a great *initialisation*. `recovery_curve` installs a set of weights,
then fine-tunes the head and records accuracy as training proceeds — so you can compare how
fast different initialisations converge:

```python
from polyweave.evaluation import recovery_curve

curve = recovery_curve(
    model,                                   # student with the generated head installed
    init=lambda m: [m.fc.weight, m.fc.bias], # parameters to fine-tune
    sample_batch=next_support_batch,
    forward=forward,
    eval_fn=lambda m: evaluate_accuracy(m, eval_batches, None, forward),
    steps=300, lr=1e-3, eval_every=20,
)                                            # -> [(step, accuracy), ...]
```

Run it from a generated init, an NCC init, and a random init, and plot the three curves. The
generated and NCC curves **start high and climb to their ceiling in a fraction of the steps**
the random init needs — the warm start is the win. This is the same `recovery_curve` the
paper experiments use, so the curves you get match the published methodology.

!!! tip "Plot the recovery curves"
    `polyweave.viz` has publication-quality plot helpers, but a recovery curve is just
    accuracy vs step:

    ```python
    import matplotlib.pyplot as plt
    for name, curve in {"generated": gen_c, "ncc": ncc_c, "random": rand_c}.items():
        steps, accs = zip(*curve)
        plt.plot(steps, accs, "o-", label=name)
    plt.xlabel("fine-tune step"); plt.ylabel("test accuracy"); plt.legend()
    ```

    The gap between the random curve and the other two at step 0 — and how many steps it
    takes random to close it — *is* the value of the generated initialisation.

## Leg 3: a cheap ensemble

Because generating a head is cheap, you can make *several* — from different support draws or
different warm-restart students — and ensemble them. The `evaluation.ensemble` helpers work on
a member-probability stack `probs` of shape `[M, N, C]` (M members, N examples, C classes):

```python
import torch
from polyweave.evaluation import ensemble_accuracy, ensemble_gain, pairwise_disagreement

members = []
for support in support_draws:                       # a few different support sets
    gen = generate_averaged(teacher, student, support, build_prototype)
    logits = torch.cat([forward(student, b, gen)[0] for b in eval_batches])
    members.append(logits.softmax(dim=-1))
probs = torch.stack(members)                         # [M, N, C]

print("ensemble acc :", ensemble_accuracy(probs, labels))
print("ensemble gain:", ensemble_gain(probs, labels))          # ensemble − mean member
print("diversity    :", pairwise_disagreement(probs))          # how differently members err
```

`ensemble_gain` is the quantity an ensemble exists to buy: the soft vote minus the average
single-member accuracy. It's only positive when members make *different* mistakes, which
`pairwise_disagreement` measures directly. Generated members drawn from different support sets
are naturally a bit diverse, so you get a few points of gain for almost no extra cost — no
retraining, just more forward passes through the same teacher.

## When is this worth it?

Here's the honest side-by-side for getting a classification head:

| Approach | Gradient steps on student | Uses a support set | Instant? | Fine-tunes further? |
|---|---|---|---|---|
| Random init | many | no | no | yes |
| NCC (nearest-centroid) | 0 | yes | yes | it's just a baseline |
| **Hypernetwork — zero-shot** | **0** | yes | **yes** | yes |
| **Hypernetwork — warm start** | few | yes | starts strong | **converges fast** |

A hypernetwork isn't magic accuracy — for a linear head it matches a strong parameter-free
baseline rather than beating it. Its value is *flexibility*: the same generated weights give
you an instant classifier, a fast-converging fine-tune start, and cheap ensemble members, all
from one trained teacher and a few examples.

## What's next

This post used the plain additive teacher. PolyWeave's real subject is *multiplicative*
(Sigma-Pi) computation — layers that form genuine products of their inputs — and the question
of when that structure actually helps. The [Concepts](../concepts.md) page covers the
multiplicative layers, and the other blog post measures
[how linear a transformer's feed-forward block really is](how-linear-is-a-transformer-ffn.md).
To build your own teacher, see [Getting Started](../getting-started.md).

## Try it out yourself

Run the whole thing — train the teacher, generate heads, plot recovery curves and ensemble
gains — in your browser, no install required:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Notabot123/polyweave-notebooks/HEAD?labpath=notebooks/hypernetworks-on-cifar.ipynb)

!!! note "Notebook repo coming soon"
    The companion `polyweave-notebooks` repository isn't published yet — the badge points at
    its intended home. Until then, run the full experiment locally with
    `python -m polyweave.experiments.cifar_fc` (needs `pip install -e ".[experiments]"`).
