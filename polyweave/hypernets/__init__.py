"""Hypernetwork teachers — modules that generate a student's target weights.

A teacher reads a prototype ``[1, C, H, W]`` and emits the weights for a target
layer. Two head topologies, both available in vanilla and Sigma-Pi variants
(toggle ``sigma_pi=True``):

* :class:`ConvFilterTeacher` — *vector head*: encode, global-average-pool, then a
  linear layer emits a flat parameter vector that a :class:`~polyweave.targets.
  TargetSpec` unpacks. Used for conv-filter generation (Experiment 2).
* :class:`FCMapTeacher` / :class:`QKMapTeacher` — *spatial map head*: the encoder
  preserves the prototype's spatial resolution and a conv weight-head emits
  weight maps aligned with that resolution, with a small pooled bias head. Used
  for FC head generation (Experiment 1) and Q/K generation (Experiment 3), where
  the prototype's spatial structure *is* the signal and must not be pooled away.

Every Sigma-Pi teacher exposes :meth:`pi_scale_mean` returning the diagnostic
``exp(pi_scale).mean()`` (``None`` for vanilla teachers).
"""

from __future__ import annotations

from .teachers import ConvFilterTeacher, FCMapTeacher, QKMapTeacher

__all__ = ["ConvFilterTeacher", "FCMapTeacher", "QKMapTeacher"]
