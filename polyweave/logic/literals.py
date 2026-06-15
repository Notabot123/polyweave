"""Soft signed literals — differentiable rule induction with native negation.

A plain product t-norm AND (:func:`fuzzy_and`) builds *monotone* conjunctions of
positive literals. A :class:`SoftSignedLiteral` attaches one **signed log-space
exponent** per input, turning the product into a learnable conjunction whose body the
optimiser *induces* from data — negation included:

    contribution_i = t_i ** [w_i]+  ·  (1 - t_i) ** [w_i]-          ( [w]+ = max(w, 0) )
    fire           = prod_i contribution_i

    w_i > 0  ->  premise REQUIRED   (positive literal)
    w_i = 0  ->  premise IGNORED    (t ** 0 = 1, drops out of the rule)
    w_i < 0  ->  premise INHIBITORY (negated literal)

In log space this is a Sigma-Pi geometric product over the features ``[log t, log(1-t)]``,
so ``mean|w_i|`` (:meth:`exponent_abs_mean`) is the same product-recruitment diagnostic
used by :class:`~polyweave.layers.SigmaPiLinear` / :class:`~polyweave.layers.PolyLinear`:
it reads how much structured conjunction the layer has recruited.

:class:`SoftRuleLayer` ORs several literals into a soft DNF (disjunction of
conjunctions), so it can induce multi-rule theories like
``(bird & not penguin) or (bat & not broken)`` and you can read the rules straight off
the exponents via :meth:`SoftRuleLayer.rules_text`.

These are interpretable rule-learning layers in the lineage of Logical Neural Networks,
RL-Net and DR-Net — offered here in PolyWeave's geometric-product / recruitment framing.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

DEFAULT_EPS: float = 1e-6


class SoftSignedLiteral(nn.Module):
    """A single learnable conjunction with signed log-space exponents.

    Args:
        n_features: size of the input truth vector's last dimension.
        signed: if ``True`` (default) each premise can be positive, ignored, or
            *negated*; if ``False`` only positive literals are possible (a plain
            monotone product AND that cannot represent an exception).
        eps: truth values are clamped to ``[eps, 1 - eps]`` for numerical safety.
    """

    def __init__(self, n_features: int, *, signed: bool = True, eps: float = DEFAULT_EPS) -> None:
        super().__init__()
        self.signed = signed
        self.eps = eps
        self.w = nn.Parameter(torch.randn(n_features) * 0.1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """``t`` in ``[0, 1]`` (``[..., n_features]``) -> rule firing in ``(0, 1]``."""
        t = t.clamp(self.eps, 1.0 - self.eps)
        logc = self.w.clamp(min=0.0) * torch.log(t)
        if self.signed:
            logc = logc + (-self.w).clamp(min=0.0) * torch.log1p(-t)
        return torch.exp(logc.sum(-1, keepdim=True))

    @torch.no_grad()
    def exponent_abs_mean(self) -> float:
        """Recruitment metric A — ``mean(|w_i|)``; ~0 = no rule structure recruited."""
        return self.w.abs().mean().item()

    @torch.no_grad()
    def literals(self, feature_names: Optional[List[str]] = None,
                 threshold: float = 0.3) -> List[Tuple[str, str, float]]:
        """The induced literals as ``(feature, role, weight)`` where role is one of
        ``"required"`` / ``"inhibitory"`` (premises with ``|w| <= threshold`` are
        treated as ignored and omitted)."""
        names = feature_names or [f"x{i}" for i in range(self.w.numel())]
        out = []
        for name, wv in zip(names, self.w.tolist()):
            if wv > threshold:
                out.append((name, "required", wv))
            elif wv < -threshold:
                out.append((name, "inhibitory", wv))
        return out

    def extra_repr(self) -> str:
        return f"n_features={self.w.numel()}, signed={self.signed}"


class SoftRuleLayer(nn.Module):
    """``n_rules`` signed-literal conjunctions combined by a probabilistic OR (a soft DNF).

    Args:
        n_features: input truth-vector width.
        n_rules: number of conjunctions (disjuncts) to induce.
        signed: whether premises may be negated (default ``True``).
        eps: numerical clamp passed to each literal.
    """

    def __init__(self, n_features: int, n_rules: int, *, signed: bool = True,
                 eps: float = DEFAULT_EPS) -> None:
        super().__init__()
        self.rules = nn.ModuleList(
            SoftSignedLiteral(n_features, signed=signed, eps=eps) for _ in range(n_rules)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Probabilistic OR over the rules: ``1 - prod_r (1 - fire_r)`` in ``(0, 1)``."""
        not_fire = torch.cat([1.0 - rule(t) for rule in self.rules], dim=-1)
        return 1.0 - not_fire.prod(dim=-1, keepdim=True)

    @torch.no_grad()
    def rules_text(self, feature_names: Optional[List[str]] = None,
                   threshold: float = 0.3) -> List[str]:
        """Read each induced rule as a string, e.g. ``"bird & not penguin"``."""
        names = feature_names or [f"x{i}" for i in range(self.rules[0].w.numel())]
        texts = []
        for rule in self.rules:
            parts = [(n if r == "required" else f"not {n}")
                     for n, r, _ in rule.literals(names, threshold)]
            texts.append(" & ".join(parts) if parts else "(empty)")
        return texts
