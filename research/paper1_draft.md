# When Does the Pi Branch Fire? Multiplicative Hypernetworks for Few-Shot Weight Initialisation

**Stuart Whipp**  
Independent Research  
`swhipp87@gmail.com`

---

> **Draft status:** Full draft. Experiments 1–3 (FC, conv1, attention Q/K) and the §4.4 conv1 extended-training + prototype-noise ablation are complete; all result tables are populated. The graded pi-scale ordering FC < conv1 < Q/K is now confirmed across **three paired random seeds (42, 43, 44)** — the ordering holds in every individual seed, not merely on average. pi-scale diagnostics and zero-shot accuracies below are reported as mean ± std over the three seeds. (Recovery-curve tables remain to be reconciled against the multi-seed run — see note in §4.1.)

---

## Abstract

Hypernetworks — networks that generate the weights of a separate target network — offer a principled route to few-shot weight initialisation: given a compact description of a new task, a trained hypernetwork can produce a useful parameter initialisation without any gradient steps on the target network. A natural question is whether the weight-generation mapping benefits from *multiplicative* (Sigma-Pi) structure, or whether an additive convolutional teacher is sufficient. We investigate this question across three target layers that differ in the degree of nonlinearity of the mapping to be learned. When the target layer is a final linear classifier, the mapping is approximately linear and is solved near-optimally by a parameter-free nearest-centroid classifier (NCC), with neither teacher adding value. When the target layer is a convolutional filter bank, the mapping from class-conditional image statistics to useful filters is substantially more nonlinear than the corresponding fc-weight mapping, the pi branch is recruited more strongly, and a Sigma-Pi teacher with a signed-log pi branch and BatchNorm regularisation leads a vanilla convolutional teacher at zero-shot generation on seen architectures. When the target layer is the query/key projection of a self-attention block — whose token-token compatibility is computed by an intrinsically bilinear (multiplicative) product — the pi branch activates most strongly of all. We track `exp(pi_scale).mean()` as a run-time diagnostic and find that its growth is *graded*: smallest for fc generation (+0.009), larger for conv1 (+0.021), and largest for attention Q/K generation (+0.027), tracking the degree of multiplicative structure in the target mapping. Averaged over three paired random seeds, this ordering is preserved in every individual seed. This constitutes a controlled demonstration that Sigma-Pi layers activate in proportion to the intrinsic multiplicative structure of the target mapping, and provides practical guidance for choosing between additive and multiplicative hypernetwork teachers.

---

## 1. Introduction

Meta-learning and few-shot learning research has produced a rich family of methods for rapid adaptation: gradient-based approaches such as MAML [Finn et al., 2017] seek a shared initialisation from which fine-tuning is fast; metric-learning methods [Snell et al., 2017; Vinyals et al., 2016] directly embed examples into a space suited for comparison; and hypernetwork-based approaches [Ha et al., 2017; Bertinetto et al., 2016] train a secondary network to generate the parameters of a primary one.

Hypernetwork approaches are appealing because they decouple *what to learn* (the teacher) from *how to learn it* (the generated parameters). The teacher can be trained once on a distribution of tasks and then applied at inference time to a support set drawn from a new task, producing a weight initialisation without any gradient computation. This has practical value both as a zero-shot classifier and as a warm start that accelerates subsequent fine-tuning.

A key design choice in hypernetworks is the *architecture of the teacher*: how should the support statistics be mapped to parameters? Prior work has largely used standard feedforward or convolutional teachers [Ha et al., 2017; Ravi and Larochelle, 2017]. An alternative is to use *Sigma-Pi* or higher-order layers [Sigaud et al., 2011; Memisevic and Hinton, 2010], in which the output is a sum of products of input features. In the log domain, such layers compute weighted geometric products — the appropriate inductive bias when the target mapping involves multiplicative interactions between input features.

However, multiplicative layers are harder to train: the product pathway provides a weaker gradient signal than the additive pathway when the loss is already low, and without careful initialisation the product branch remains dormant. It is therefore unclear *when* such layers are worth the added complexity.

In this paper we investigate this question empirically across three target layers of increasing inductive complexity: (1) generating the weights of a final fully-connected (fc) classification layer from class-conditional feature statistics; (2) generating the weights of a first convolutional layer (conv1) from class-conditional raw-image statistics; and (3) generating the query/key (Q/K) projection weights of a self-attention block in a small transformer solving a synthetic relational-lookup task. We argue, and confirm experimentally, that:

- The fc → weight mapping is approximately linear (it reduces to nearest-centroid classification), and neither an additive nor a multiplicative teacher improves on the parameter-free NCC baseline on unseen architectures.
- The conv1 → filter mapping is substantially more nonlinear than the fc-weight mapping, recruiting the pi branch more strongly, and the Sigma-Pi teacher leads at zero-shot generation on seen architectures.
- The attention Q/K mapping is multiplicative by construction (token-token compatibility is a bilinear product), and the pi branch activates more strongly here than in either of the other two regimes.
- The pi-scale of our Sigma-Pi teacher grows during training in proportion to the degree of multiplicative structure in the target mapping — smallest for fc, larger for conv1, largest for attention Q/K — providing a graded diagnostic.

Our contributions are:
1. A controlled comparison of additive and multiplicative hypernetwork teachers across three target layers with different inductive structures.
2. A novel signed-log Sigma-Pi formulation that preserves the sign of inputs in log-space, enabling the pi branch to distinguish excitation from inhibition.
3. Empirical evidence that the pi branch is activated by target mappings with multiplicative structure in a *graded* fashion, with pi-scale serving as a practical run-time diagnostic.
4. An analysis of prototype statistics (mean, variance, kurtosis, inter-class contrast; and an embedding cross-moment for the attention task) sufficient to condition hypernetwork teachers.

---

## 2. Related Work

### 2.1 Hypernetworks

Ha et al. [2017] introduced the term *hypernetwork* for a network that generates the weights of a main network. Their work focused on weight compression: a small hypernetwork generates the large weight matrices of an LSTM or ResNet, sharing structure across layers. Bertinetto et al. [2016] applied a one-shot hypernetwork to few-shot learning, training a learnet to predict the parameters of a siamese network from a single example image. More recently, hypernetworks have been used for neural architecture search [Brock et al., 2018], continual learning [von Oswald et al., 2019], and implicit neural representations [Sitzmann et al., 2020].

Our setting is closest to Bertinetto et al. [2016] and to the task-adaptive prediction (TADAM) line of work [Oreshkin et al., 2018], in which a conditioning network modulates a base learner. We differ in focusing on the teacher architecture — additive vs. multiplicative — and in using population statistics (moments) rather than raw examples as the conditioning input.

### 2.2 Meta-Learning and Few-Shot Initialisation

MAML [Finn et al., 2017] and its variants [Nichol et al., 2018; Rajeswaran et al., 2019] seek an initialisation that can be fine-tuned quickly with few examples. Prototypical networks [Snell et al., 2017] and relation networks [Sung et al., 2018] instead learn an embedding space in which class means are discriminative. Our NCC baseline is equivalent to the Prototypical Network classifier applied to frozen features, providing a strong, parameter-free reference point.

### 2.3 Higher-Order and Multiplicative Networks

The theoretical motivation for multiplicative layers dates to early work on higher-order Boltzmann machines [Sejnowski, 1986] and sigma-pi units [Rumelhart and McClelland, 1986]. Memisevic and Hinton [2010] introduced gated RBMs in which three-way interactions model the mapping between image pairs, demonstrating that multiplicative structure is natural for transformations. Sigaud et al. [2011] provide a unified treatment of higher-order networks.

In the context of attention mechanisms, the scaled dot-product attention of Transformers [Vaswani et al., 2017] is multiplicative: query-key compatibility is computed as an inner product, which is a bilinear (multiplicative) operation. This connection motivates our Experiment 3, in which the teacher generates the query/key projection weights of every self-attention block in a small transformer.

Polynomial neural networks [Chrysos et al., 2021] and π-nets provide a framework for networks whose activations are high-order polynomials of the input. Our Sigma-Pi teacher is a lightweight special case: a single product pathway operating in log-space, added to an additive pathway.

Higher-order Sigma-Pi units [Rumelhart and McClelland, 1986; Shin and Ghosh, 1991] are long established, and hypernetworks themselves can be cast as a form of multiplicative interaction [Jayakumar et al., 2020]. To our knowledge, however, using a Sigma-Pi convolutional block as the *internal computation of a weight-generating hypernetwork* — and measuring how strongly different target layers recruit that multiplicative pathway through a learnable per-channel gate — has not been previously studied. It is this combination, rather than the Sigma-Pi primitive itself, that we claim as novel.

### 2.4 Log-Space and Geometric Processing

Processing in log-space to perform multiplicative operations has a long history in signal processing (cepstrum analysis), probabilistic models (log-probabilities), and normalising flows. In neural networks, the key challenge is handling negative inputs: log is undefined for negative real numbers. Our signed-log formulation, `z = sign(x) * log(|x| + ε)`, is an odd extension of the log to the full real line. It is related to the log-modulus transformation used in statistics [John, 1980] and the `asinh` transformation, and preserves both the magnitude (in log-scale) and the sign of the input.

---

## 3. Method

<!-- FIGURE PLACEHOLDER (Overleaf) — methodology overview; redraw hypernet.png as vector/SVG -->
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/hypernet.pdf}
  \caption{Method overview. A teacher (hypernetwork) $g_w$ reads a compact
  prototype $P(S)$ of a few-shot support set and emits the parameters $\psi$ of a
  frozen-trunk student's target layer, $\psi = g_w(P(S))$. We compare an additive
  teacher against a signed-log Sigma-Pi teacher across three target layers (FC,
  conv1, attention Q/K) of increasing multiplicative structure.}
  \label{fig:method-overview}
\end{figure}
```

### 3.1 Problem Setting

We consider a student network `f_θ` partitioned into a frozen trunk `f_ϕ` and a target layer `f_ψ`. The trunk processes inputs and produces intermediate representations; the target layer maps these to outputs. Given a small support set `S = {(x_i, y_i)}_{i=1}^N` drawn from the task at hand, we wish to generate good parameters `ψ` without gradient descent on the student.

A hypernetwork teacher `g_w` takes a compact summary `P(S)` of the support set (the *prototype map*) and outputs `ψ`:

    ψ = g_w(P(S))

The teacher is trained to minimise the cross-entropy of `f_{ϕ,ψ}` on the support set, sampled over a distribution of student architectures and support sets.

### 3.2 Prototype Maps

We consider two prototype representations, appropriate to the two experimental regimes.

**Feature-space prototype (Experiment 2: fc generation).** After applying the frozen trunk to the support set, we compute per-class statistics in the feature space: mean `μ_k`, variance `σ²_k`, excess kurtosis `κ_k`, and inter-class contrast `δ_k = |μ_k − μ_global|`. These are stacked to form a tensor `P ∈ ℝ^{1×4×K×D}` where `K` is the number of classes and `D` is the feature dimension. Each channel is normalised to zero mean and unit variance across the spatial dimensions `(K, D)`.

**Input-space prototype (Experiment 2: conv1 generation).** Since conv1 operates on raw inputs, the prototype must live in input space. We partition each input image into a `G×G` spatial grid and compute the same four statistics per class per cell, yielding `P ∈ ℝ^{1×4×K×(G²·C_in)}` where `C_in` is the number of input channels. Kurtosis and inter-class contrast capture distributional structure (multi-modality, between-class separability) that mean and variance cannot express.

The kurtosis channel is particularly motivated for CIFAR-10: classes such as "automobile" and "ship" have strongly non-Gaussian intensity distributions (heavy tails from sharp edges), while "frog" and "deer" are smoother. Kurtosis provides a lightweight proxy for this.

**Embedding cross-moment prototype (Experiment 3: attention Q/K generation).** For the relational-lookup task the relevant signal is which query token matches which key token. We summarise a support batch by second-moment matrices in the (shared, frozen) `D`-dimensional embedding space, yielding `P ∈ ℝ^{1×4×D×D}` with four `D×D` channels: the query↔matched-key cross-moment `R_qk = E[e_q ⊗ e_{k*}]` (the relation signal), the query auto-covariance `C_qq`, the key auto-covariance `C_kk`, and the query↔mean-key context `R_qctx`. Each channel is normalised to zero mean and unit variance over its `(D, D)` support. The optimal Q/K bilinear form is, to first order, a whitened version of `R_qk`, so the prototype's `D×D` spatial structure *is* the relation; the teacher must preserve it end-to-end (see §3.5).

### 3.3 NCC Baseline

The nearest-centroid classifier assigns a test point `h` to the class `k` minimising `‖h − μ_k‖²`. Expanding:

    argmin_k ‖h − μ_k‖² = argmax_k [ h·μ_k − ½‖μ_k‖² ]

This is exactly a linear layer with `W = [μ_k]` and `b_k = −½‖μ_k‖²`. NCC is therefore a zero-parameter fc initialisation computed directly from support centroids, and serves as a theoretical ceiling for any teacher trying to generate fc weights from feature statistics alone.

### 3.4 Additive Teacher (ConvHyperTeacher)

The additive teacher is a small convolutional network:

    h₁ = BN(ReLU(Conv(P)))        # width w, kernel 3×3
    h₂ = BN(ReLU(Conv(h₁)))
    ψ  = Conv(h₂, 1×1)            # collapses to target shape

BatchNorm and Dropout(0.1) are applied after each hidden layer. For fc generation the output is reshaped to `[K, D]` (weights) plus a pooled projection to `[K]` (biases). For conv1 generation the output is flattened and reshaped to `[C_out, C_in, k, k]`.

### 3.5 Sigma-Pi Teacher (ConvSigmaPiHyperTeacher)

The Sigma-Pi teacher replaces the middle convolutional layer with a ConvSigmaPi2d block. The block computes:

**Sigma branch (additive):**

    σ = Conv_σ(x − mean(x))

The zero-centring removes the DC component, focusing the additive branch on differences and spatial structure.

**Pi branch (multiplicative via signed log-space):**

    z   = sign(x) · log(|x| + ε)
    π   = exp(s) · tanh(Conv_π(z))

where `s` is a per-channel learnable scale parameter, initialised to `−2` so that `exp(s) ≈ 0.135` keeps the pi branch initially subdominant.

The signed-log `z` is an odd (antisymmetric) function: positive inputs map to positive log-space, negative inputs to negative log-space. This contrasts with `log(softplus(x))`, which maps all negative inputs to small positive values, discarding their sign. Preserving the sign allows the pi branch to distinguish excitation (`x > 0`) from inhibition (`x < 0`) at each spatial location.

The combined output is:

    out = ReLU(BN(σ + π))

BatchNorm after the sum normalises the scale relationship between branches, ensuring that gradient flow to the pi branch is not suppressed by a dominant sigma branch.

The pi scale `s` is monitored throughout training as a diagnostic. If the target mapping has multiplicative structure, `s` should grow from its initial value; if not, it should remain near `−2`.

### 3.6 Warm Restarts and Student Diversity

To train the teacher on a rich distribution of target parameters, we generate multiple valid `ψ` for each student trunk. We first train the full student for `E_base` epochs, then freeze the trunk. We then perform `R` warm restarts: at each restart, we reinitialise `ψ` with Kaiming normal and fine-tune for `E_restart` epochs with Adam, yielding `R` distinct parameter snapshots per architecture. This provides a diverse training set of `(P, ψ)` pairs for the teacher.

Cross-architecture generalisation is tested by holding out one student architecture entirely during teacher training (the "unseen" group) and evaluating on it at test time.

### 3.7 BN Statistics Reset (Conv1 Generation Only)

After installing generated conv1 weights into a student, the BatchNorm layer immediately following conv1 has running statistics calibrated to the *original* conv1 output distribution. With generated weights, the activation distribution changes, making the stored BN statistics inaccurate and degrading accuracy. We address this by running `B_reset` forward passes on the support set with BN in training mode, re-estimating the running mean and variance before evaluating or fine-tuning.

---

## 4. Experiments

We describe three experiments in increasing order of target-layer complexity: an fc classification head (§4.1), a first convolutional layer (§4.2), and the query/key projections of a self-attention block (§4.3). A conv1 ablation (extended training + prototype noise) is reported in §4.4. All three core experiments use a matched teacher regime (5,000 teacher steps, width 64, Adam lr 1e-3) so that the pi-scale diagnostic is comparable across target layers.

### 4.1 Experiment 1 — CIFAR-10, FC Layer

**Setup.** Three student architectures (StudentA: 3×3 conv stack; StudentB: wider with 5×5 first kernel; StudentC: VGG-style double conv) are trained on CIFAR-10 with data augmentation. The teacher generates the fc layer from 4-channel feature-space prototypes. Students A and B are seen during teacher training; Student C is the unseen test architecture. The Sigma-Pi teacher uses the signed-log pi branch with BN after sigma+pi (v3 formulation).

**NCC dominance.** The parameter-free NCC baseline outperforms both teachers on the unseen architecture. This is expected: the fc → centroid mapping is approximately linear (it is exactly NCC), so any teacher can at best approximate what NCC computes directly. Teacher generalisation to unseen architectures is limited because the teacher was calibrated to the feature geometry of weaker training architectures.

| Method | Seen (zero-shot) | Unseen (zero-shot) |
|---|---|---|
| Random | 0.113 ± 0.015 | 0.085 ± 0.020 |
| NCC | 0.754 ± 0.004 | **0.800 ± 0.004** |
| Conv | 0.700 ± 0.026 | 0.739 ± 0.024 |
| Conv-SigmaPi | **0.731 ± 0.017** | **0.753 ± 0.030** |

*pi_scale (exp-mean) during training: 0.1353 → 0.1445 (Δ = +0.0092 ± 0.0016, mean ± std over 3 seeds; slow monotonic growth).*

<!-- FIGURE PLACEHOLDER (Overleaf) -->
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.8\linewidth]{figures/polyweave_cifar_fc_zeroshot_multiseed.pdf}
  \caption{Experiment 1 (FC head): zero-shot accuracy on seen and unseen
  architectures, mean $\pm$ std over seeds 42/43/44. The parameter-free NCC
  baseline dominates; Conv-SigmaPi edges the additive Conv teacher.}
  \label{fig:fc-zeroshot}
\end{figure}
```

Conv-SigmaPi outperforms the additive Conv teacher at zero-shot on both seen (0.731 vs 0.700) and unseen (0.753 vs 0.739) students in the three-seed mean. With the v3 formulation (signed-log + BN), the pi branch is no longer fully dormant — it grows slowly, consistent with there being a small degree of nonlinearity in the feature→centroid mapping that NCC's linear form cannot capture.

**Recovery curves.** Recovery finals (3-seed mean over unseen students) are: random **0.832**, Conv-SigmaPi 0.828, Conv 0.814, NCC 0.804 — all within a ~2.8-point band. The fc layer is tiny (2,570 parameters) and 300 Adam steps fully converge from any starting point, so even a fresh random init reaches the top of the band; among the teacher inits, Conv-SigmaPi remains best (above Conv in every seed). For this target layer, zero-shot accuracy is the informative metric; recovery barely distinguishes methods once a short fine-tuning budget is spent.

**Pi-scale as a graded diagnostic.** The growth of pi-scale here (+0.0092 ± 0.0016) is meaningfully smaller than in Experiment 2 (conv1 generation, +0.0213 ± 0.0015) and Experiment 3 (attention Q/K, +0.0267 ± 0.0049). Rather than a binary dormant/active switch, pi-scale responds in proportion to the degree of multiplicative structure in the target mapping. This graduated response strengthens its value as a diagnostic.

### 4.2 Experiment 2 — CIFAR-10, Conv1 Layer

**Setup.** The same three student architectures share an identical conv1 layer (`Conv2d(3, 32, 3, padding=1)`, 896 parameters). The teacher generates all conv1 weights from a 4-channel, 4×4 spatial grid raw-image prototype (`P ∈ ℝ^{1×4×10×48}`). There is no NCC analogue. The Sigma-Pi teacher uses signed-log pi branch + BN.

**Pi branch activates.** In contrast to Experiment 1 (fc generation), the pi-scale grows monotonically from 0.1353 to 0.1566 (Δ = +0.0213 ± 0.0015, mean ± std over 3 seeds) during the 5,000-step training run, consistent with the hypothesis that the conv1 generation task has substantially more nonlinear (and plausibly multiplicative) structure than the fc mapping.

**Zero-shot results.** The teacher successfully generates conv1 weights that achieve far above chance for *seen* architectures. Cross-architecture zero-shot fails (near chance), consistent with the conv1 feature geometry being architecture-specific.

| Method | Seen (zero-shot, mean) | Unseen (zero-shot, mean) |
|---|---|---|
| Random | 0.107 ± 0.004 | 0.118 ± 0.003 |
| Conv | 0.605 ± 0.020 | 0.157 ± 0.026 |
| Conv-SigmaPi | **0.634 ± 0.030** | 0.155 ± 0.033 |

<!-- FIGURE PLACEHOLDER (Overleaf) -->
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.8\linewidth]{figures/polyweave_cifar_conv1_zeroshot_multiseed.pdf}
  \caption{Experiment 2 (conv1 layer): zero-shot accuracy on seen and unseen
  architectures, mean $\pm$ std over seeds 42/43/44. Both teachers generate
  conv1 weights far above chance for seen students (Conv-SigmaPi ahead); cross-
  architecture transfer collapses to near chance, exposing the conv1 feature
  geometry's architecture-specificity.}
  \label{fig:conv1-zeroshot}
\end{figure}
```

**Recovery curves.** After fine-tuning conv1 only on the support set (300 steps, three seeds), the zero-shot ranking reverses. Sigma-Pi's zero-shot lead does not survive fine-tuning: the additive Conv teacher overtakes it on unseen recovery in every seed, and a fresh random init — which 300 Adam steps carry a long way for this small layer — is competitive with both.

| Method | Unseen recovery (mean final, 3 seeds) |
|---|---|
| Random | 0.578 ± 0.019 |
| Conv | **0.593 ± 0.010** |
| Conv-SigmaPi | 0.563 ± 0.041 |

The reading echoes the §4.4 ablation and the §5.6 limitation: **Sigma-Pi's zero-shot accuracy is higher (seen 0.634 vs 0.605), but the additive Conv teacher overtakes once a fine-tuning budget is spent.** We did not tune the recovery optimiser per method; further work could sweep learning rates and other hyperparameters — for instance with random search or Bayesian optimisation, neither of which we used here — to deduce a truer per-method optimum. The contribution of this paper is the pi-scale *diagnostic* rather than a head-to-head end-task comparison, and that diagnostic result — pi-scale growth Δ = +0.0213 ± 0.0015 — is robust and seed-consistent.

### 4.3 Experiment 3 — Synthetic Attention, Q/K Generation

**Setup.** We move to the most directly multiplicative target: the query/key projection weights of self-attention. Students are tiny transformers (`d_model=64`, `n_heads=4`, `n_layers=2`, attention-only blocks) that share a single frozen token+positional embedding space, so all students inhabit a common matching geometry. Architectural diversity comes from a per-architecture pointwise activation — tanh, ReLU, or Swish — applied inside each block; a pointwise nonlinearity cannot perform cross-token matching, so it diversifies the trunk without changing the task. tanh and ReLU students are *seen* during teacher training; Swish is the *unseen* test architecture.

The task is a **relational lookup**. Each episode draws a fresh random relation `π: vocab → vocab` (a permutation). A sequence presents `K=5` key slots, distractors, and a query token `q` at the last position; the label is the slot whose token equals `π(q)`. Chance is `1/K = 0.20`. Because the relation is resampled every episode, the prototype→Q/K mapping is non-trivial: the teacher must read the query↔key cross-moment, infer the relation, and emit projections that implement it.

**Q/K for *every* layer; no leak.** The teacher generates the Q/K weights and biases of *all* attention layers (16,640 parameters across the two blocks). This is essential: if only the first block's Q/K were generated, the remaining base-trained blocks would solve the lookup unaided, making the generated weights irrelevant. We verify the absence of this leak with a random-Q/K baseline, which scores at chance (0.20) — confirming the generated weights are the sole cross-token routing mechanism. The teacher is *image-to-image*: it preserves the `D×D` resolution of the embedding cross-moment prototype throughout (no global pooling before the weight head, which would destroy the relation signal), emitting each Q/K matrix as a `D×D` output map and the biases from a small pooled branch.

**Pi branch activates most strongly.** The pi-scale grows monotonically from 0.1353 to 0.1620 over the 5,000-step run (Δ = **+0.0267 ± 0.0049**, mean ± std over 3 seeds), the largest growth of the three target layers — consistent with the bilinear query-key product being the most explicitly multiplicative mapping studied. Crucially, every individual seed's Q/K growth exceeds every conv1 seed's (min Q/K Δ = +0.0227 > max conv1 Δ = +0.0225), so the conv1 < Q/K ordering is rank-consistent across seeds despite the wider Q/K spread.

**Zero-shot and recovery.** Both teachers generate Q/K well above chance at zero-shot (conv 0.43 / sigma-pi 0.42 on seen students vs the 0.20 chance level), the clearest separator of methods. Recovery (fine-tuning Q/K only on the support relation, 200 steps) converges to near-ceiling from a teacher init in the healthy seeds (42 and 44: both teachers reach ~0.996, effectively tied, with random a little lower at ~0.91); seed 43 collapses for all three inits alike (an episode-sampling instability, not a property of any one method), so recovery does not separate the teachers and we read no Sigma-Pi recovery advantage into it. The decisive evidence that the generated weights are load-bearing (not a leaked shortcut) is the random-Q/K *zero-shot* score sitting exactly at chance.

| Method | Seen (zero-shot) | Unseen (zero-shot) |
|---|---|---|
| Random | 0.200 ± 0.001 | 0.202 ± 0.001 |
| Conv | **0.427 ± 0.029** | **0.262 ± 0.087** |
| Conv-SigmaPi | 0.421 ± 0.014 | 0.260 ± 0.096 |

*pi_scale (exp-mean) during training: 0.1353 → 0.1620 (Δ = +0.0267 ± 0.0049, mean ± std over 3 seeds; the largest of the three experiments). Matched regime: prototype noise = 0, as in Experiments 1–2.*

<!-- FIGURE PLACEHOLDER (Overleaf) -->
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.8\linewidth]{figures/polyweave_synthetic_attention_zeroshot_multiseed.pdf}
  \caption{Experiment 3 (attention Q/K): zero-shot accuracy on seen and unseen
  architectures, mean $\pm$ std over seeds 42/43/44, against the $1/K = 0.20$
  chance level (dashed). Both teachers sit well above chance; the additive Conv
  and Conv-SigmaPi teachers are level within seed-to-seed std, while the
  random-Q/K baseline pins to chance, confirming the generated weights are the
  sole cross-token routing mechanism.}
  \label{fig:qk-zeroshot}
\end{figure}
```

Notably, in this regime the additive Conv teacher is at best *level* with the Sigma-Pi teacher on end-task accuracy (seen 0.427 vs 0.421, unseen 0.262 vs 0.260 — differences well within the seed-to-seed std), even though the pi branch activates most strongly here. We read this as follows: pi-scale growth measures the extent to which the Sigma-Pi teacher *recruits* its multiplicative pathway to fit the mapping — a property of the mapping's structure — not a guarantee that multiplicative computation yields higher accuracy than a sufficiently-trained additive approximator at this scale. The diagnostic claim (pi fires in proportion to multiplicative structure) is supported; the accuracy comparison between teachers is discussed as a limitation in §5.6.

### 4.4 Ablation — Conv1 with Extended Training and Prototype Noise

**Setup.** To probe the robustness of the Experiment 2 conv1 result, we re-run it with `teacher_steps = 10,000` (double), `warm_restart_epochs = 10` (deeper target distribution), and `proto_noise_std = 0.05` (Gaussian noise added to the prototype each step, as data augmentation over support statistics). Because these settings differ from the matched regime used in §4.1–4.3, this run is reported as an **ablation**, not as a source of the comparison-table numbers: pi-scale grows with training length, so its value here is not directly comparable to the 5,000-step diagnostics above.

**Motivation.** The Experiment 2 teacher failed to generalise cross-architecture at zero-shot; prototype noise tests whether training on perturbed prototypes improves robustness to the seen→unseen distributional shift.

**Results.** Zero-shot accuracy is averaged over 10 seen and 5 unseen students; unseen recovery is the mean final accuracy after 300 conv1-only fine-tuning steps.

| Method | Seen (zero-shot) | Unseen (zero-shot) | Unseen recovery |
|---|---|---|---|
| Random | 0.1195 | 0.1302 | 0.548 |
| Conv | 0.6617 | 0.1869 | **0.653** |
| Conv-SigmaPi | **0.6662** | **0.1902** | 0.633 |

*pi_scale trajectory (10,000 steps): 0.1375 (step 500) → 0.1691 (step 10,000), Δ ≈ +0.032 — larger than the matched 5,000-step conv1 diagnostic (+0.0213 ± 0.0015), consistent with longer training recruiting more multiplicative computation. This extended run is single-seed and is not part of the three-seed matched comparison.*

**Reading.** Two observations carry over from the matched run, and one new nuance emerges. (i) Prototype noise does **not** repair cross-architecture generalisation: unseen zero-shot accuracy stays near chance (~0.19 vs 0.10 chance) for both teachers, confirming that the seen→unseen failure of Experiment 2 is a representational mismatch the teacher cannot augment its way out of. (ii) The Sigma-Pi teacher edges out the additive teacher on *zero-shot* accuracy for both seen (0.6662 vs 0.6617) and unseen (0.1902 vs 0.1869) students, and its pi-scale grows further under the longer schedule. (iii) However, after 300 recovery steps the additive teacher's initialisation reaches slightly higher final accuracy on unseen students (0.653 vs 0.633). This mirrors the attention-experiment caveat (§5.6): the pi branch being more active does not guarantee a downstream accuracy advantage once a sufficient fine-tuning budget is spent. The Sigma-Pi advantage is concentrated at the zero-shot and short-budget end.

---

### 4.5 Ensembling Teacher-Generated Populations

**Setup.** A warm-restarted student population is *diverse by construction*: its members share a target architecture but reach their conv1 layer by different optimisation routes. We ask whether the teacher that generates their conv1 filters affects that diversity, and whether it buys a better ensemble. For each teacher (additive Conv and Conv-SigmaPi), and *without any retraining*, we generate and install conv1 for all 10 seen-architecture students (arch A×5 + B×5) under the pure zero-shot protocol of §4.2 (generated filters through the student's original bn1), collect each member's softmax over a fixed CIFAR-10 test set, and form a soft-vote ensemble (mean softmax). We report mean single-member accuracy, ensemble accuracy, the ensemble gain (ensemble − mean member), and the mean pairwise prediction disagreement between members (an error-diversity measure; Kuncheva and Whitaker, 2003).

```{=latex}
\begin{figure}[t]
  \centering
  \includegraphics[width=0.49\linewidth]{figures/polyweave_cifar_conv1_ensemble_accuracy_seed42.pdf}
  \hfill
  \includegraphics[width=0.49\linewidth]{figures/polyweave_cifar_conv1_ensemble_diversity_seed42.pdf}
  \caption{Ensembling teacher-generated conv1 populations (seed 42, 10 seen-architecture
  students, pure zero-shot). \textbf{Left:} mean single-member accuracy ($\pm$ std error
  bars) versus soft-vote ensemble accuracy for each teacher. Both teachers' ensembles
  improve substantially over their average member ($+0.119$ additive, $+0.131$ Sigma-Pi).
  \textbf{Right:} distribution of pairwise member disagreement; the Sigma-Pi population is
  marginally more diverse (mean $0.441$ vs $0.438$, dashed lines). Single seed --- suggestive,
  not conclusive.}
  \label{fig:ensemble}
\end{figure}
```

**Results.** Both teachers produce populations whose ensemble materially beats the average member. The additive teacher's members average $0.5838 \pm 0.0533$ and ensemble to $0.7031$ (gain $+0.1193$); the Sigma-Pi teacher's members average $0.5964 \pm 0.0514$ and ensemble to $0.7273$ (gain $+0.1309$). Mean pairwise disagreement is $0.4377$ (additive) versus $0.4414$ (Sigma-Pi).

**Reading.** Single-member accuracy is slightly ahead for the Sigma-Pi population ($0.5964$ vs $0.5838$), consistent with §4.2's zero-shot lead. The Sigma-Pi population also shows marginally higher prediction diversity and a marginally larger ensemble gain — the latter two are coupled, since more diverse members make more decorrelated errors and therefore ensemble better. These results suggest Sigma-Pi might lead to greater diversity in a generated student population; however, the effect is small and measured at a single seed, so more extensive research with repeated experiments would be needed to verify this. We report it as a suggestive observation rather than a claim.

---

## 5. Analysis and Discussion

### 5.1 Why NCC Wins on FC Generation

The task of generating fc weights from class centroids is, to first order, solved by NCC. The teacher must learn an approximation to a function that is already available in closed form: `W_k = μ_k, b_k = −½‖μ_k‖²`. Any teacher can at best match this, and in practice falls short because (a) it is calibrated to the feature geometry of the training architectures, and (b) it only sees a noisy, finite-sample estimate of the centroids.

This result has a clean theoretical interpretation: the optimal fc layer for an L2-nearest-centroid task is a linear function of the class means. There is no multiplicative structure for the pi branch to exploit. The pi-scale diagnostic confirms this: it remains flat throughout training.

This is not a failure of hypernetworks — it is a signal that the teacher adds no value over a parameter-free baseline in this regime. The right lesson is that hypernetworks should be applied to target layers where the optimal weight mapping is *not* available in closed form.

### 5.2 Why Conv1 Generation Is Harder and Why Pi Helps

The conv1 → filter mapping is fundamentally different from the fc case. What our experiments directly support is that **the mapping from class-conditional image statistics to useful conv1 filters appears substantially more nonlinear than the corresponding fc-weight mapping**: the fc mapping is solved in closed form by NCC (§5.1), whereas no closed-form analogue exists for conv1, the teacher's pi branch is recruited during training, and both teachers generate conv1 weights far above chance at zero-shot on seen architectures (a random init stays near chance).

As *intuition* for why multiplicative structure might be present — though we stress this is motivation rather than proof — edge- and texture-selective filters can be thought of in terms of spatial-frequency and orientation tuning, and a Gabor-like response is naturally expressed as an interaction (product) of a frequency term and an orientation term. We do not claim the filters are *literally* generated through multiplicative interactions in the parameter space we predict; we infer the presence of exploitable multiplicative structure from two indirect signals — the growing pi-scale and the well-above-chance zero-shot generation on seen architectures — and offer the frequency × orientation picture only as an interpretive aid.

The monotonic growth of pi-scale in Experiment 2 — from 0.1353 to 0.1566 (Δ = +0.0213 ± 0.0015 over 3 seeds) over 5,000 steps — is consistent with this reading. The pi branch is not growing dramatically, but it grows robustly and seed-consistently. (The downstream recovery comparison between teachers is less settled: as noted in §4.2, the additive teacher overtakes Sigma-Pi once a fine-tuning budget is spent, so we rest the diagnostic claim on pi-scale growth rather than on the recovery endpoint.)

### 5.3 Cross-Architecture Generalisation

The zero-shot cross-architecture drop in Experiment 2 is expected: the teacher was trained on StudentA and StudentB features, and StudentC has a different conv1 output distribution due to its deeper stem. At the recovery endpoint the picture is more equivocal: for the small conv1 layer, 300 fine-tuning steps carry even a random init most of the way, so the teacher inits retain only a slim margin (the additive teacher over random, with Sigma-Pi falling back to roughly random; §4.2). The attention experiment (Experiment 3) shows the same convergence in a milder form: zero-shot accuracy drops from seen to unseen (Swish) architectures, but recovery converges to ~0.99 regardless, indicating the generated Q/K provides a useful warm start that survives the activation change.

Prototype noise (the §4.4 ablation) is an attempt to make the conv1 teacher more robust to this distributional shift by training it on a family of perturbed prototypes rather than the exact training-set statistics — analogous to input augmentation in standard supervised learning. Because it changes the training regime, we keep it out of the matched cross-experiment comparison.

### 5.4 Pi-Scale as a Graded Diagnostic

A practical contribution of this work is the use of `exp(pi_scale).mean()` as a run-time diagnostic for the degree of multiplicative structure in the target mapping. Across experiments the response is graded:

| Target layer | pi-scale Δ (mean ± std, 3 seeds) | Teacher advantage |
|---|---|---|
| FC (classification head) | +0.0092 ± 0.0016 | Conv-SigmaPi > Conv at zero-shot, but NCC dominates |
| Conv1 (filter generation) | +0.0213 ± 0.0015 | Conv-SigmaPi > Conv at seen zero-shot; Conv overtakes at recovery (§4.2) |
| Attention Q/K (bilinear) | **+0.0267 ± 0.0049** | Conv ≈ Conv-SigmaPi on accuracy (pi most active) |

The ordering FC < conv1 < attention Q/K matches the *a priori* ordering of multiplicative structure in the three target mappings: a near-linear centroid map, a more nonlinear spatial-filter map, and an explicitly bilinear query-key product. This graduated response is more useful than a binary dormant/active indicator: it provides a continuous measure of how much the pi branch is contributing. A teacher designer can use pi-scale growth as a cheap, post-hoc signal for whether the more complex Sigma-Pi architecture was warranted — while bearing in mind (Experiment 3) that strong pi activation indicates the teacher is *using* multiplicative computation, not that it necessarily beats a well-trained additive teacher on end-task accuracy.

The ordering is now established across **three paired random seeds (42, 43, 44)** and is rank-consistent in *every* seed: the FC band (+0.0092 ± 0.0016) is cleanly separated from conv1 (+0.0213 ± 0.0015), and although the Q/K band (+0.0267 ± 0.0049) is wider, its smallest per-seed growth (+0.0227) still exceeds conv1's largest (+0.0225). The previously-flagged "modest single-seed margin" between conv1 and Q/K therefore holds up under repetition, with the caveat that Q/K recruitment is more variable run-to-run.

<!-- FIGURE PLACEHOLDER (Overleaf) — headline result -->
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.7\linewidth]{figures/polyweave_pi_ordering.pdf}
  \caption{Headline result: Sigma-Pi recruitment ($\Delta$ pi-scale, mean $\pm$ std
  over seeds 42/43/44) increases with the multiplicative structure of the target
  mapping, FC $<$ conv1 $<$ Q/K. The ordering is rank-consistent in every seed.}
  \label{fig:pi-ordering}
\end{figure}
```

### 5.5 Occlusion Sensitivity: A Mechanistic Confirmation

The pi-scale diagnostic measures multiplicative recruitment *from inside* the teacher (the magnitude of a learnable gate). It is natural to ask whether the features the teacher computes actually behave multiplicatively under an *external, model-agnostic* probe. Occlusion sensitivity (Zeiler & Fergus, 2014) — masking part of an input and measuring the resulting drop in a response — provides exactly such a probe, and it draws a sharp, quantitative line between additive and multiplicative computation.

The key observation is interaction. For an additive feature `r = f(A) + g(B)` over two disjoint input groups `A`, `B`, occluding either group removes only that group's contribution, so the joint drop equals the sum of the single drops and the interaction `drop(A&B) − drop(A) − drop(B)` is zero. For a multiplicative feature `r = f(A)·g(B)`, occluding *either* factor collapses the entire response, so each single drop already equals the joint drop and the interaction is strongly negative (sub-additive). We summarise this in a **conjunction index** `(drop_A + drop_B − drop_{A&B}) / |drop_{A&B}|`, clamped to `[0, 1]`: 0 for a purely additive feature, 1 for a purely multiplicative (conjunctive "AND") one.

This probe reproduces the paper's central theme — that multiplicativity is *graded*, not binary — along an independent axis. On a controlled feature that interpolates between additive and multiplicative, `r = (1−α)(p+q) + α·(p·q)`, the conjunction index rises monotonically with the multiplicative fraction `α` (0.00, 0.25, 0.49, 0.74, 1.00 at α = 0, ¼, ½, ¾, 1; Figure&nbsp;{fig:conjunction-index}), echoing the FC < conv1 < Q/K ordering recovered by pi-scale. The spatial form of the probe makes the conjunctive signature visually unmistakable (Figure&nbsp;{fig:occlusion-heatmaps}): for a detector responding to the *product* of two image patches, occluding either patch removes ~100% of the response (each factor alone is critical — the AND-gate), whereas an additive detector loses only ~50% per patch.

We note one subtlety specific to our formulation: the signed-log pi branch `z = sign(x)·log(|x|+ε)` is *additive in log-space*, so the cleanest linear-space AND-signature is exhibited by genuine products such as the attention query–key score — which is precisely the Experiment 3 target where pi-scale grows most. This prediction is borne out on real data: applying the spatial probe to a trained CIFAR student whose `conv1` filters were generated by the additive teacher versus the Sigma-Pi teacher (Figure&nbsp;{fig:student-occlusion}) yields input-space sensitivity maps that are *qualitatively similar*, rather than the strong divergence one would see for a genuinely product-form feature — consistent with the signed-log branch behaving additively in the pixel domain. Occlusion sensitivity is therefore a complementary, post-hoc confirmation of recruitment rather than a replacement for the pi-scale diagnostic, and it is provided as a reusable probe in the accompanying library. *(Implementation: `polyweave.interpretability.occlusion`; the synthetic figures are reproduced by `python -m polyweave.experiments.occlusion_demo` and the CIFAR-student overlay by `python -m polyweave.experiments.student_occlusion`.)*

<!-- FIGURE PLACEHOLDER (Overleaf) -->
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.7\linewidth]{figures/polyweave_occlusion_conjunction_index.pdf}
  \caption{Occlusion \emph{conjunction index} for a feature interpolating between
  additive and multiplicative, $r = (1-\alpha)(p+q) + \alpha\,(p\cdot q)$. The index
  rises monotonically with the multiplicative fraction $\alpha$ (0.00, 0.25, 0.49,
  0.74, 1.00), an external confirmation of the graded recruitment recovered by
  pi-scale (FC $<$ conv1 $<$ Q/K). Error bars are $\pm1$ std over inputs.}
  \label{fig:conjunction-index}
\end{figure}
```

<!-- FIGURE PLACEHOLDER (Overleaf) -->
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/polyweave_occlusion_heatmaps.pdf}
  \caption{Spatial occlusion sensitivity (fraction of response lost) for an
  additive vs.\ a multiplicative two-patch detector, on a shared colour scale.
  Both localise the two informative patches, but the multiplicative detector loses
  $\sim$100\% of its response when \emph{either} patch is occluded (the
  conjunctive AND-signature), while the additive detector loses only $\sim$50\%.}
  \label{fig:occlusion-heatmaps}
\end{figure}
```

<!-- FIGURE PLACEHOLDER (Overleaf) -->
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/polyweave_cifar_conv1_student_occlusion_seed42.pdf}
  \caption{Occlusion sensitivity on a real CIFAR-10 image for a trained (unseen-
  architecture) student whose \texttt{conv1} filters were generated by the additive
  teacher (centre) vs.\ the $\Sigma\Pi$ teacher (right); brighter = larger relative
  drop in the predicted-class logit when that region is occluded. The two maps are
  qualitatively similar, as expected: the signed-log pi branch is additive in
  log-space, so it leaves no strong product-form AND-signature in the pixel domain
  (cf.\ the synthetic product detector of Figure~\ref{fig:occlusion-heatmaps}).
  Generated by \texttt{polyweave.experiments.student\_occlusion} with no retraining.}
  \label{fig:student-occlusion}
\end{figure}
```

### 5.6 Limitations

- **Scale of experiments.** The convolutional experiments use CIFAR-10 with three student architectures; the attention experiment uses a synthetic relational-lookup task with tiny transformers. The findings may not generalise to larger datasets (ImageNet), real language/vision transformers, more diverse architecture families, or other target layers (value/output projections, normalisation parameters).
- **Pi activation ≠ accuracy advantage.** In the attention Q/K experiment the additive teacher slightly outperforms the Sigma-Pi teacher on end-task accuracy despite the pi branch being most active. Pi-scale growth diagnoses that the teacher *recruits* multiplicative computation in proportion to the mapping's structure; it does not by itself imply the multiplicative teacher achieves higher accuracy than a sufficiently-trained additive one. The diagnostic and the accuracy comparison are distinct claims, and only the former is the central contribution.
- **Seed count and the recovery endpoint.** Pi-scale growth is reported over three paired seeds (42, 43, 44) and the FC < conv1 < Q/K ordering is rank-consistent in every seed, so the recruitment ordering is not a single-seed result; three seeds is still a modest sample, and Q/K growth in particular is variable run-to-run (σ ≈ 0.005). The recovery-accuracy comparisons are the least settled part of the study: across three seeds the additive teacher overtakes Sigma-Pi at the conv1 recovery endpoint (§4.2), random recovery is competitive for the tiny fc layer (§4.1), and Q/K recovery finals are high-variance owing to a single collapsing seed (§4.3). We therefore rest the paper's claims on the recruitment diagnostic and the zero-shot accuracies — which are seed-consistent — rather than on the recovery endpoints.
- **Regime consistency.** The three core experiments are matched on teacher steps, width, learning rate, and prototype noise (set to zero everywhere) so that the pi-scale diagnostic is comparable across target layers. The §4.4 extended run deliberately departs from the matched regime (longer training and σ=0.05 prototype noise) and is reported only as a robustness ablation.
- **Prototype design.** The prototype statistics (moments; embedding cross-moments for attention) are hand-crafted. A learned encoder would likely outperform them, at the cost of conflating the prototype representation with the teacher architecture in ablation studies.
- **Recovery steps.** Recovery is measured at a fixed budget. The advantage of teacher-generated initialisations may narrow or widen at different budgets.

---

## 6. Conclusion

We have presented a controlled study of additive versus multiplicative hypernetwork teachers for few-shot weight initialisation, varying the target layer across three regimes to change the inductive structure of the weight-generation mapping. Our main findings are:

1. When the target mapping is approximately linear (fc weight generation), neither teacher outperforms the parameter-free NCC baseline, and the pi branch grows only slightly (Δ = +0.0092 ± 0.0016 over 3 seeds).
2. When the target mapping is substantially more nonlinear (conv1 filter generation), the Sigma-Pi teacher recruits its pi branch markedly more (Δ = +0.0213 ± 0.0015) and leads at seen zero-shot, though the additive teacher overtakes it at the recovery endpoint once a fine-tuning budget is spent (§4.2).
3. When the target mapping is explicitly bilinear (attention query/key generation), the pi branch activates most strongly of all (Δ = +0.0267 ± 0.0049) — the ordering FC < conv1 < Q/K mirroring the *a priori* ordering of multiplicative structure in the three mappings, and holding in every one of three paired seeds.
4. The signed-log formulation `z = sign(x) · log(|x| + ε)` is an important ingredient: the earlier softplus-log formulation discards input sign and yields a less useful pi branch.
5. Pi-scale (`exp(pi_scale).mean()`) serves as a cheap, *graded* run-time diagnostic for the degree of multiplicative structure in the target mapping.

These findings suggest that higher-order / Sigma-Pi layers are not universally beneficial but are recruited in proportion to the multiplicative structure of the weight-generation task. We caution that strong pi activation diagnoses the teacher's *use* of multiplicative computation rather than guaranteeing an accuracy advantage over a well-trained additive teacher (§5.6). The conv1 < Q/K margin, modest in a single seed, is confirmed rank-consistent across three paired seeds. Promising future directions include scaling the attention experiment to real transformers and language/vision tasks, and generating value/output projections and normalisation parameters.

**Sigma-Pi students.** Throughout this work the weight-generator is multiplicative but its targets are conventional, additive layers. A natural next step is to close the loop: equip the *student* with Sigma-Pi layers and task the teacher with generating their multiplicative weights. Our recruitment hypothesis predicts the pi branch should engage most strongly in this setting, since the target mapping is then explicitly higher-order — extending the graded ordering observed across FC, convolutional, and attention targets. Crucially, this design disentangles two effects our current experiments conflate: whether the pi branch fires because the *generation map* is multiplicative, or because the *target layer* is. We expect the strongest activation when both hold.

**Shared versus separate log-space pathways.** Our pi branch applies a single weight matrix to the signed-log representation `z = sign(x) · log(|x| + ε)`, which folds the positive and negative log-space regimes into one set of parameters. An informative ablation is to instead learn *two* weight matrices — one for the positive and one for the negative branch of the signed-log — versus *sharing* a single matrix as we do now. The separate-matrix variant offers strictly more expressive capacity; the shared variant is more compute- and parameter-efficient. Should the shared variant match the separate one, that would be a particularly satisfying result: it would suggest the product pathway's *inductive bias* matters more than the extra capacity, reinforcing the view that multiplicative structure — not parameter count — is what the pi branch contributes.

---

## Code and Reproducibility

The signed-log Sigma-Pi primitive (`ConvSigmaPi2d`), the additive and multiplicative hypernetwork teachers, the prototype encoders, the NCC and random baselines, and the full experiment harness used in this paper are packaged as **PolyWeave**, a small open-source PyTorch library (MIT-licensed). All three experiments are reproducible via a single multi-seed driver that writes the figures and the per-seed `multiseed_results.json` underlying the tables above; the graded pi-scale ordering reported here was produced by this driver across seeds 42, 43, and 44. PolyWeave additionally ships a factorised polynomial layer (`PolyLinear`) and an activation-space distillation harness developed for follow-on work. *(Release: PyPI `polyweave` and accompanying documentation — repository URL to be added on publication.)*

## Acknowledgements

This research was conducted independently. Experiments were run on a consumer GPU with 6 GB VRAM.

---

## References

Bertinetto, L., Henriques, J. F., Valmadre, J., Torr, P. H. S., and Vedaldi, A. (2016). Learning feed-forward one-shot learners. *NeurIPS 2016*.

Brock, A., Lim, T., Ritchie, J. M., and Weston, N. (2018). SMASH: One-shot model architecture search through hypernetworks. *ICLR 2018*.

Chrysos, G. G., Moschoglou, S., Bouritsas, G., Deng, J., Panagakis, Y., and Zafeiriou, S. (2021). Deep polynomial neural networks. *IEEE TPAMI*.

Finn, C., Abbeel, P., and Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. *ICML 2017*.

Ha, D., Dai, A., and Le, Q. V. (2017). HyperNetworks. *ICLR 2017*.

Jayakumar, S. M., Czarnecki, W. M., Menick, J., Schwarz, J., Rae, J., Osindero, S., Teh, Y. W., Harley, T., and Pascanu, R. (2020). Multiplicative interactions and where to find them. *ICLR 2020*.

John, J. A. (1980). Outliers in factorial experiments. *Journal of the Royal Statistical Society, Series C*.

Kuncheva, L. I. and Whitaker, C. J. (2003). Measures of diversity in classifier ensembles and their relationship with the ensemble accuracy. *Machine Learning*, 51(2), 181–207.

Memisevic, R. and Hinton, G. E. (2010). Learning to represent spatial transformations with factored higher-order Boltzmann machines. *Neural Computation*.

Nichol, A., Achiam, J., and Schulman, J. (2018). On first-order meta-learning algorithms. *arXiv:1803.02999*.

Oreshkin, B. N., López, P. R., and Lacoste, A. (2018). TADAM: Task dependent adaptive metric for improved few-shot learning. *NeurIPS 2018*.

Rajeswaran, A., Finn, C., Kakade, S., and Levine, S. (2019). Meta-learning with implicit gradients. *NeurIPS 2019*.

Rumelhart, D. E. and McClelland, J. L. (1986). *Parallel Distributed Processing*, Vol. 1. MIT Press.

Sejnowski, T. J. (1986). Higher-order Boltzmann machines. *AIP Conference Proceedings*.

Shin, Y. and Ghosh, J. (1991). The pi-sigma network: an efficient higher-order neural network for pattern classification and function approximation. *IJCNN 1991*.

Sigaud, O., Salaün, C., and Padois, V. (2011). On-line regression algorithms for learning mechanical models of robots: a survey. *Robotics and Autonomous Systems*.

Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B., and Wetzstein, G. (2020). Implicit neural representations with periodic activation functions. *NeurIPS 2020*.

Snell, J., Swersky, K., and Zemel, R. (2017). Prototypical networks for few-shot learning. *NeurIPS 2017*.

Sung, F., Yang, Y., Zhang, L., Xiang, T., Torr, P. H. S., and Hospedales, T. M. (2018). Learning to compare: Relation network for few-shot learning. *CVPR 2018*.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention is all you need. *NeurIPS 2017*.

Vinyals, O., Blundell, C., Lillicrap, T., Wierstra, D., and Kavukcuoglu, K. (2016). Matching networks for one-shot learning. *NeurIPS 2016*.

von Oswald, J., Henning, C., Sacramento, J., and Grewe, B. F. (2019). Continual learning with hypernetworks. *ICLR 2020*.

---

## Appendix A — Architecture Details

### Student Architectures

**StudentA:** `Conv2d(3,32,3,p=1)–BN–ReLU–Pool → Conv2d(32,64,3,p=1)–BN–ReLU–Pool → Conv2d(64,128,3,p=1)–BN–ReLU–Pool → Flatten → Linear(2048,256)–ReLU → Linear(256,10)`

**StudentB:** As StudentA but first conv is `Conv2d(3,48,5,p=2)` and second is `Conv2d(48,96,3)`, wider with a larger first receptive field.

**StudentC (unseen):** VGG-style double conv before each pooling. Deeper stem, different activation geometry — the key test of cross-architecture generalisation.

### Teacher Architectures (v3 formulation)

**ConvHyperTeacher:** `Conv–BN–ReLU–Drop → Conv–BN–ReLU–Drop → Conv(1×1)` with width `w=64`. Separate bias head via AdaptiveAvgPool.

**ConvSigmaPiHyperTeacher:** `Conv–BN–ReLU–Drop → ConvSigmaPi2d(w) → Conv–BN–ReLU–Drop → Conv(1×1)`. The ConvSigmaPi2d block has learnable per-channel `pi_scale` initialised to −2.

### ConvSigmaPi2d Block

```
class ConvSigmaPi2d(nn.Module):
    def __init__(self, channels):
        self.sigma_conv = Conv2d(ch, ch, 3, padding=1)
        self.pi_conv    = Conv2d(ch, ch, 3, padding=1)
        self.pi_scale   = Parameter(full((ch, 1, 1), -2.0))
        self.bn         = BatchNorm2d(ch)

    def forward(self, x):
        sigma = self.sigma_conv(x - x.mean((-2,-1), keepdim=True))
        z     = sign(x) * log(abs(x) + 1e-8)
        pi    = exp(self.pi_scale) * tanh(self.pi_conv(z))
        return relu(self.bn(sigma + pi))
```

### Attention Experiment (Experiment 3)

**Students.** Tiny transformers with `d_model=64`, `n_heads=4`, `n_layers=2`, attention-only blocks (no feed-forward sublayer), `LayerNorm(x + act(attn(x)))`. Token and positional embeddings are drawn once from a fixed seed, shared across all students, and frozen, so every student inhabits the same matching geometry. Per-architecture diversity is a single pointwise activation: tanh and ReLU (seen), Swish (unseen). Base students are trained on the identity relation; warm restarts reinitialise and fine-tune Q/K only.

**Task.** Relational lookup with `vocab=64`, `seq_len=10`, `K=5` key slots (chance = 0.20). Each episode samples a fresh permutation `π: vocab → vocab`; the label is the slot whose token equals `π(q)` for query token `q`.

**Prototype.** Embedding cross-moment tensor `P ∈ ℝ^{1×4×64×64}`: channels `R_qk = E[e_q ⊗ e_{k*}]`, `C_qq`, `C_kk`, `R_qctx`, each normalised over its `(D, D)` support.

**Teachers (image-to-image).** The teacher preserves the `D×D` prototype resolution end-to-end and emits each Q/K weight matrix as a `D×D` output map (a `Conv2d(w, 2·n_layers, 3, p=1)` head) with biases from a pooled branch and a learnable output scale (init 0.1). The Sigma-Pi variant inserts a `ConvSigmaPi2d(w)` block (identical to the CIFAR formulation) between the input conv and the weight head. The teacher generates Q/K for **all** attention layers (16,640 parameters); a random-Q/K baseline scores at chance, confirming the generated weights are the sole cross-token routing mechanism.

| Hyperparameter | Value |
|---|---|
| d_model / n_heads / n_layers | 64 / 4 / 2 |
| vocab / seq_len / key slots K | 64 / 10 / 5 (chance 0.20) |
| Student activations | tanh, ReLU (seen); Swish (unseen) |
| Student base steps | 2,000 (converges by ~300) |
| Warm restarts | per architecture, Q/K-only fine-tuning |
| Teacher steps / lr / width | 5,000 / 1e-3 / 64 |
| Prototype | embedding cross-moment, `1×4×64×64` |
| Prototype noise | 0 (matched to Experiments 1–2) |
| Recovery steps | 200, Q/K-only, Adam lr=1e-3 |

## Appendix B — Training Details

| Hyperparameter | Value |
|---|---|
| Dataset | CIFAR-10 (50k train / 10k test) |
| Augmentation | RandomCrop(32, pad=4), RandomHorizontalFlip |
| Normalisation | mean=(0.491,0.482,0.447), std=(0.247,0.244,0.262) |
| Student base training | 15 epochs, Adam lr=1e-3, CosineAnnealingLR |
| Warm restarts (v3) | 5 per arch (base), 10 per arch (extended) |
| Teacher optimiser | Adam, lr=1e-3, CosineAnnealingLR |
| Teacher steps | 5,000 (base), 10,000 (extended) |
| Gradient clipping | max norm 1.0 |
| Teacher width | 64 |
| Proto channels | 4 (mean, variance, kurtosis, contrast) |
| Proto noise (extended) | σ=0.05, added to full proto tensor |
| BN reset batches | 10 (conv1 generation only) |
| Recovery steps | 300, Adam lr=1e-3 |
| Eval support batches | 5 |
| Hardware | Single GPU, 6 GB VRAM |
| DataLoader workers | 0 (Windows paging file constraint) |
