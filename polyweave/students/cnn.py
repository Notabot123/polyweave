"""CIFAR-style convolutional students with a factored-out first conv layer.

The first convolution (``conv1`` + ``bn1`` + ``pool1``) is kept separate from the
rest of the trunk so a hypernetwork teacher can surgically replace *just* the
first-layer filters (Experiment 2). The forward pass also accepts a generated
linear head (Experiment 1). When no generated weights are supplied the student
behaves like an ordinary classifier.

All architectures share an identical ``conv1`` specification
(``Conv2d(in_ch, conv1_out, k, padding=k//2)``) so the generated conv1 target is
a fixed shape regardless of which student is being repaired. Architectural
diversity lives entirely in ``trunk_rest``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

GeneratedConv1 = Dict[str, torch.Tensor]  # {"weight": [O,I,k,k], "bias": [O]}
GeneratedFC = Dict[str, torch.Tensor]     # {"weight": [C,F], "bias": [C]}


class CNNStudent(nn.Module):
    """A convolutional classifier with a replaceable first conv layer.

    Args:
        trunk_rest: the body of the network mapping the pooled ``conv1`` output
            ``[B, conv1_out, H/2, W/2]`` to a flat feature vector of size
            ``feature_dim`` (typically ending in ``Flatten``/``Linear``/``ReLU``).
        feature_dim: width of the penultimate feature vector.
        num_classes: number of output classes.
        in_ch: input image channels (3 for CIFAR).
        conv1_out: output channels of ``conv1`` (the generated target width).
        kernel_size: ``conv1`` kernel size.
    """

    def __init__(
        self,
        trunk_rest: nn.Module,
        feature_dim: int = 256,
        num_classes: int = 10,
        in_ch: int = 3,
        conv1_out: int = 32,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.conv1_out = conv1_out
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_ch, conv1_out, kernel_size, padding=self.padding)
        self.bn1 = nn.BatchNorm2d(conv1_out)
        self.pool1 = nn.MaxPool2d(2)
        self.trunk_rest = trunk_rest
        self.fc = nn.Linear(feature_dim, num_classes)

    # -- conv1 stage (optionally with generated weights) --------------------
    def _apply_conv1(self, x: torch.Tensor, gen_conv1: Optional[GeneratedConv1]) -> torch.Tensor:
        if gen_conv1 is not None:
            h = F.conv2d(x, gen_conv1["weight"], gen_conv1["bias"], padding=self.padding)
        else:
            h = self.conv1(x)
        return self.pool1(F.relu(self.bn1(h)))

    def extract_features(
        self, x: torch.Tensor, gen_conv1: Optional[GeneratedConv1] = None
    ) -> torch.Tensor:
        return self.trunk_rest(self._apply_conv1(x, gen_conv1))

    def forward(
        self,
        x: torch.Tensor,
        gen_conv1: Optional[GeneratedConv1] = None,
        generated_fc: Optional[GeneratedFC] = None,
    ) -> torch.Tensor:
        h = self.extract_features(x, gen_conv1)
        if generated_fc is not None:
            return F.linear(h, generated_fc["weight"], generated_fc["bias"])
        return self.fc(h)


# ---------------------------------------------------------------------------
# Architecture zoo
# ---------------------------------------------------------------------------

def _trunk_a(conv1_out: int, feature_dim: int) -> nn.Module:
    """Plain widening trunk: conv1_out -> 64 -> 128 -> feature_dim."""
    return nn.Sequential(
        nn.Conv2d(conv1_out, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(128 * 4 * 4, feature_dim), nn.ReLU(),
    )


def _trunk_b(conv1_out: int, feature_dim: int) -> nn.Module:
    """Wider trunk: conv1_out -> 96 -> 128 -> feature_dim."""
    return nn.Sequential(
        nn.Conv2d(conv1_out, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(96, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(128 * 4 * 4, feature_dim), nn.ReLU(),
    )


def _trunk_c(conv1_out: int, feature_dim: int) -> nn.Module:
    """VGG-style double-conv trunk: conv1_out -> 64 -> 128 -> 128 -> feature_dim."""
    return nn.Sequential(
        nn.Conv2d(conv1_out, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(128 * 4 * 4, feature_dim), nn.ReLU(),
    )


_TRUNKS: Dict[str, Callable[[int, int], nn.Module]] = {
    "A": _trunk_a,
    "B": _trunk_b,
    "C": _trunk_c,
}


def make_cnn_student(
    arch: str = "A",
    *,
    feature_dim: int = 256,
    num_classes: int = 10,
    in_ch: int = 3,
    conv1_out: int = 32,
    kernel_size: int = 3,
) -> CNNStudent:
    """Build one CNN student of architecture ``arch`` in ``{"A", "B", "C"}``.

    The trunks differ in width/depth but all consume the same fixed ``conv1``
    output (assuming a 32x32 input that has been pooled to 16x16 then reduced to
    4x4 by the trunk's two pooling stages).
    """
    key = arch.upper()
    if key not in _TRUNKS:
        raise ValueError(f"unknown arch {arch!r}; choose from {sorted(_TRUNKS)}")
    trunk = _TRUNKS[key](conv1_out, feature_dim)
    return CNNStudent(
        trunk, feature_dim=feature_dim, num_classes=num_classes,
        in_ch=in_ch, conv1_out=conv1_out, kernel_size=kernel_size,
    )


def make_cnn_students(
    archs: Optional[List[str]] = None, **kwargs
) -> List[CNNStudent]:
    """Build one student per architecture name (defaults to A, B, C)."""
    if archs is None:
        archs = ["A", "B", "C"]
    return [make_cnn_student(a, **kwargs) for a in archs]
