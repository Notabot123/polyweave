"""Smoke test for the student occlusion-overlay driver.

Exercises the testable core (``student_occlusion_maps``) and the overlay plot on
tiny fabricated CIFAR-shaped data — a real CNN student + two real (untrained)
ConvFilterTeachers — without torchvision or the saved-models payload. Shapes and
file outputs are checked; numeric values are not.
"""

from __future__ import annotations

import torch

from polyweave.experiments.student_occlusion import (
    _denormalize,
    student_occlusion_maps,
)
from polyweave.experiments.cifar_conv1 import CONV1_IN, CONV1_KERNEL, CONV1_OUT
from polyweave.hypernets import ConvFilterTeacher
from polyweave.students import make_cnn_student
from polyweave.targets import Conv2dTargetSpec
from polyweave.utils import set_seed
from polyweave.viz import plot_occlusion_overlay

NUM_CLASSES = 3
PROTO_GRID = 4


def _tiny_teachers():
    spec = Conv2dTargetSpec(CONV1_OUT, CONV1_IN, CONV1_KERNEL)
    return {
        "conv": ConvFilterTeacher(spec, proto_channels=4, width=8, sigma_pi=False),
        "conv_sigmapi": ConvFilterTeacher(spec, proto_channels=4, width=8, sigma_pi=True),
    }


def _support(n_batches=2, bs=8):
    return [(torch.randn(bs, 3, 32, 32), torch.randint(0, NUM_CLASSES, (bs,)))
            for _ in range(n_batches)]


def test_student_occlusion_maps_shapes_and_keys():
    set_seed(0)
    student = make_cnn_student(
        "A", feature_dim=16, num_classes=NUM_CLASSES,
        in_ch=CONV1_IN, conv1_out=CONV1_OUT, kernel_size=CONV1_KERNEL,
    )
    teachers = _tiny_teachers()
    image = torch.randn(1, 3, 32, 32)

    maps, classes = student_occlusion_maps(
        student, teachers, _support(), image,
        num_classes=NUM_CLASSES, proto_grid=PROTO_GRID,
        window=8, stride=4, bn_reset_batches=2,
    )

    assert set(maps) == {"conv", "conv_sigmapi"}
    assert set(classes) == {"conv", "conv_sigmapi"}
    for m in maps.values():
        assert m.ndim == 2                       # [Hout, Wout]
        assert m.shape[0] > 0 and m.shape[1] > 0
        assert torch.isfinite(m).all()
    for c in classes.values():
        assert 0 <= c < NUM_CLASSES


def test_fixed_target_class_is_honoured():
    set_seed(1)
    student = make_cnn_student(
        "A", feature_dim=16, num_classes=NUM_CLASSES,
        in_ch=CONV1_IN, conv1_out=CONV1_OUT, kernel_size=CONV1_KERNEL,
    )
    _, classes = student_occlusion_maps(
        student, _tiny_teachers(), _support(), torch.randn(1, 3, 32, 32),
        num_classes=NUM_CLASSES, proto_grid=PROTO_GRID,
        window=8, stride=4, bn_reset_batches=1, target_class=2,
    )
    assert classes == {"conv": 2, "conv_sigmapi": 2}


def test_denormalize_recenters_into_unit_range():
    # A zero (normalised) image maps to the CIFAR mean, which is inside [0, 1].
    img = _denormalize(torch.zeros(1, 3, 32, 32))
    assert img.shape == (3, 32, 32)
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0


def test_overlay_plot_writes_pdf(tmp_path):
    maps = {"additive teacher": torch.rand(8, 8), r"$\Sigma\Pi$ teacher": torch.rand(8, 8)}
    image = torch.rand(3, 32, 32)
    written = plot_occlusion_overlay(
        image, maps, name="overlay_test", plots_dir=tmp_path,
    )
    assert (tmp_path / "overlay_test.pdf").exists()
    assert any(p.suffix == ".pdf" for p in written)
