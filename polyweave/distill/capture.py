"""Collect a submodule's ``(input, output)`` activation pairs via a forward hook.

Model-agnostic: works on any ``nn.Module`` whose ``forward`` takes the tensor to
be transformed as its first positional argument and returns a tensor — which is
exactly the shape of a transformer feed-forward sub-block (``mlp(x) -> y``). The
captured pairs are the regression dataset for :func:`fit_layer`.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn


class IOCapture:
    """Context manager that records inputs/outputs of ``module`` during forward.

    Usage::

        with IOCapture(model.transformer.h[6].mlp) as cap:
            for batch in loader:
                model(batch)          # forward passes drive the hook
        X, Y = cap.pairs()            # [N, in], [N, out]

    Leading dims (batch, sequence, ...) are flattened so each *token* becomes one
    independent training example, matching the position-wise nature of an FFN.

    Args:
        module: the submodule to tap.
        max_rows: stop accumulating once this many flattened rows are collected
            (``None`` = unbounded). Useful to cap a large activation cache.
        device: where to store captured tensors (default ``"cpu"`` to spare GPU
            memory during a long caching pass).
        flatten_leading: flatten all but the last dim into rows (default ``True``).
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        max_rows: Optional[int] = None,
        device: str = "cpu",
        flatten_leading: bool = True,
    ) -> None:
        self.module = module
        self.max_rows = max_rows
        self.device = device
        self.flatten_leading = flatten_leading
        self._xs: List[torch.Tensor] = []
        self._ys: List[torch.Tensor] = []
        self._rows = 0
        self._handle = None

    def _hook(self, _module, inputs, output):
        if self.max_rows is not None and self._rows >= self.max_rows:
            return
        x = inputs[0]
        y = output[0] if isinstance(output, (tuple, list)) else output
        x = x.detach()
        y = y.detach()
        if self.flatten_leading:
            x = x.reshape(-1, x.shape[-1])
            y = y.reshape(-1, y.shape[-1])
        if self.max_rows is not None:
            room = self.max_rows - self._rows
            x, y = x[:room], y[:room]
        self._xs.append(x.to(self.device))
        self._ys.append(y.to(self.device))
        self._rows += x.shape[0]

    def __enter__(self) -> "IOCapture":
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_exc) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @property
    def num_rows(self) -> int:
        return self._rows

    def pairs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the concatenated ``(X, Y)`` activation pairs."""
        if not self._xs:
            raise RuntimeError("no activations captured; run a forward pass inside the context")
        return torch.cat(self._xs, dim=0), torch.cat(self._ys, dim=0)


def collect_io(
    module: nn.Module,
    run_fn: Callable[[], None],
    *,
    max_rows: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper: run ``run_fn`` under an :class:`IOCapture` and return pairs.

    ``run_fn`` should drive whatever forward passes exercise ``module`` (e.g. a
    loop feeding text batches through a language model).
    """
    with IOCapture(module, max_rows=max_rows, device=device) as cap:
        run_fn()
    return cap.pairs()
