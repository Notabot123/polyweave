# Paper 3 — Scope (working draft)

**Working titles**
- *Reasoning Recoverability: An Explicit Product-AND Chaining Bias Generalizes Where Transformers Don't*
- *How Deep Can It Reason? Depth Generalization of Deductive Inference, Measured*
- *A Parameter-Free Reasoning Layer: Systematic Generalization vs Learned Transformers*

> Status: scope only. No experiments run yet. Decisions marked **[OPEN]** need a call
> before the relevant phase.

---

## 1. Contribution & positioning

**The honest constraint.** Differentiable forward chaining over Horn clauses is prior art
(∂ILP, Neural LP, TensorLog, DeepProbLog, Logic Tensor Networks, NTP). The *chainer is not
the contribution.* The novelty is the **measurement** and the **throughline**, not the
mechanism.

**Throughline (why this author, this library).** This is the third instalment of a single
research programme — *measuring when explicit structure is recruited/recoverable*:
- Paper 1: pi-scale **recruitment** (when does the product branch fire?).
- Paper 2: **linear recoverability** (how much of an FFN is just a linear map?).
- Paper 3: **reasoning recoverability** — how much of a deductive task is captured (and how
  far does it generalize) when reasoning is done by an explicit, parameter-tiny product-AND
  chaining layer vs learned by a transformer.

**Headline claim (the result we must demonstrate).** An explicit product-AND chaining bias
generalizes to proof depths and KB sizes *far beyond training*, where learned transformers
(and CoT-prompted small LMs) degrade sharply — at orders of magnitude fewer parameters, and
exactly (sound & complete for Horn). The novel framing: deductive generalization is a
**recruitment/recoverability** question, and a single product (Sigma-Pi) primitive — the same
one Papers 1–2 study — is what the task needs.

**Why it's arXiv-suitable and defensible.** Controlled, measurement-first, runs on a 6 GB
GPU, honest about scope. Distinct angle (the recoverability lens + the product-AND
throughline) in an otherwise crowded field.

**Candid novelty/risk note.** "Transformers fail to length/depth-generalize" is partly known
(length-generalization literature, CLUTRR). So the *measurement design* and the *product-AND /
recruitment framing* must carry the novelty, not the bare failure. Mitigation: (a) the
cross-source controlled depth curve as a clean instrument; (b) the recruitment measurement (§4
Measurement C) tying reasoning to Sigma-Pi, which no prior work does; (c) framing the chaining
layer as a *drop-in inductive bias* with a recoverability metric, paralleling Paper 2.

---

## 2. The task & the construct

**Task.** Propositional Horn entailment. Input = a set of facts + rules + a query atom;
output = entailed truth value in [0, 1] (thresholded for accuracy). Each instance has a known
**minimal proof depth** d (the controlled axis).

**The reasoning layer.** `polyweave.reasoning.ForwardChainer` (already shipped): product
t-norm AND, max OR, iterate to fixpoint. Parameter count ≈ 0 when rules are given; tiny if
rules are learned. Sound & complete for Horn ⇒ exact at any depth by construction.

**"Recoverability" = the metric.** Primary axis is **accuracy as a function of test proof
depth**, with models trained only on shallow proofs (depth ≤ D_train) and evaluated to
D_test ≫ D_train. The chaining layer should stay flat at ~100%; learned models should decay.
The *gap* and *where it appears* is the figure.

---

## 3. Datasets (3 sources, distinct roles)

| Source | Role | Notes |
|---|---|---|
| **Synthetic Horn-KB** (we generate) | Controlled core | We own depth/branching/#facts/#rules/#distractors. Essential for the depth-generalization claim. Built on `PropKB`. |
| **ProntoQA** | Established + CoT-LM home | Controllable proof depth; natural fit for the small-LM CoT reference. |
| **ProofWriter** | Harder external validity | Closed-world assumption + negation + depth; stress test. |

**Scoping nuance (important).** ProntoQA/ProofWriter are natural language but ship the
underlying symbolic facts/rules. For the *core reasoning* claim, consume the **symbolic form**
(measure reasoning, not parsing). Raw NL is reserved for Further Work (§7).

**Synthetic generator (Phase 0 deliverable).** Random Horn KBs with knobs: target minimal
proof depth `d`, branching factor, #atoms, #rules, #distractor rules (irrelevant to the
query), and a balanced True/False query split (include unprovable queries so it's not all
positives). Emit `(facts, rules, query, label, depth)`. Train on `d ≤ D_train`, test on a
spread up to `D_test`.

---

## 4. Measurements

- **A — Depth/length generalization (headline).** Accuracy vs test proof depth, per model,
  per dataset. Train shallow, test deep. *Figure 1.*
- **B — Parameter & compute efficiency.** Params vs OOD accuracy; the chaining layer is ~0
  params and exact. *Figure 2.* One honest table on tokens/compute.
- **C — Recruitment lens (the throughline; secondary).** Train a free-form model that *can*
  use products (e.g. a `PolyLinear`/Sigma-Pi-augmented reasoner) on the logical task and
  measure whether it **recruits multiplicative structure** (the library's recruitment
  diagnostics: `exponent_abs_mean` / `quad_scale_mean`). Question: does logical conjunction
  induce product recruitment? Ties reasoning directly to Papers 1–2. **[OPEN]** include now
  or hold for a follow-up.

---

## 5. Baselines (answering "what do we compare against?")

The established chainer is the **ceiling**, not the headline — the result needs a *learned
foil that fails to generalize*.

- **Symbolic forward/backward chainer** — ceiling/sanity (≈100%; confirms solvability and
  that our differentiable version matches it).
- **Learned transformer** — trained on the task; its depth-generalization collapse *is* the
  result. **[OPEN]** size/architecture (small decoder vs encoder).
- **GNN over the rule graph** — optional second learned foil (a stronger structured baseline).
  **[OPEN]** include?
- **CoT-prompted small LM (GPT-2 / Llama-160m)** — reference for "can a small LM do this
  without the module / without scale," and the bridge to Further Work.
- **Cite, don't reimplement:** RuleTaker/ProofWriter transformers, an NTP, ∂ILP.

---

## 6. Key figures (target)

1. **Accuracy vs proof depth** (per model, per dataset) — the money figure: chaining flat,
   transformer/CoT decaying past D_train.
2. **Params vs OOD accuracy** — efficiency frontier.
3. **Method schematic** — facts/rules → product-AND chaining layer → entailment, with the
   product-AND highlighted as a Pi neuron.
4. *(if Measurement C)* recruitment diagnostic on logical vs non-logical tasks.

---

## 7. Scope boundaries → Further Work (and library roadmap)

Explicitly out of scope for this paper, framed as the horizon:
- **Language I/O grounding** (NL→KB→reason→NL). The crux is grounding, not reasoning; a big,
  crowded systems effort. Genuine future work *and* a future library module.
- **Differentiable backward chainer / NTP** (radbas soft-unification + cumsum/radbas
  addressing). Our highest-novelty mechanism; deferred deliberately. Future library module.
- **First-order logic** (variables/unification) beyond propositional.
- **Perceptual grounding** (facts extracted from images, DeepProbLog-style).

---

## 8. Related work to position against

Differentiable logic (∂ILP — Evans & Grefenstette; Neural LP — Yang et al.; TensorLog —
Cohen; DeepProbLog — Manhaeve et al.; Logic Tensor Networks — Serafini & Garcez; NTP —
Rocktäschel & Riedel). Systematic/length generalization (CLUTRR — Sinha et al.; the
length-generalization literature). LM reasoning benchmarks (RuleTaker — Clark et al.;
ProofWriter — Tafjord et al.; ProntoQA — Saparov & He; chain-of-thought). **Our delta:** the
recoverability/recruitment measurement framing + the product-AND throughline + a parameter-tiny
differentiable layer studied as a drop-in inductive bias across controlled + established
sources.

---

## 9. Phased plan

- **Phase 0** — Synthetic Horn-KB generator with depth/size knobs + balanced labels (on
  `PropKB`). *Gate: can we dial minimal proof depth reliably and verify with the symbolic
  chainer?*
- **Phase 1** — Experiment harness: train/eval loop, the depth-generalization protocol, the
  transformer baseline, metrics + Figure 1 plumbing.
- **Phase 2** — Run **synthetic**; produce the depth curve. **Decision gate: does the story
  hold?** (chaining flat, transformer decays). If not, reconsider before investing in the
  real datasets.
- **Phase 3** — ProntoQA + ProofWriter (symbolic form); + the CoT-LM reference.
- **Phase 4** — *(optional)* Measurement C (recruitment on logical tasks).
- **Phase 5** — Write-up (NeurIPS-style .tex, arXiv first), figures, related-work.

---

## 10. Decisions (resolved 2026-06-13)

1. **Measurement C — IN.** The recruitment study (does a logical task recruit product
   structure?) is the throughline that makes this *our* paper; worth the extra work.
2. **Learned baseline = small decoder transformer; GNN deferred.** Keep the foil simple/honest
   for v1. A GNN over the rule graph is a strong *future* addition if the transformer foil
   looks too weak — slot in without restructuring.
3. **CoT reference = Llama-160m** (SwiGLU; harness already exists from Paper 2).
4. **Title** — defer until Phase 2 confirms the result shape.
5. **Symbolic-only core, NL strictly in Further Work** — confirmed (isolate reasoning from
   parsing for the main claim).

*Pivots expected.* The Phase 2 gate exists precisely so we can change course (or add the GNN
foil, extend depth, swap datasets) cheaply after seeing the first synthetic curve.
