# tests/test_generator_shape.py
from __future__ import annotations

import pytest
import torch

from src.model.generator import UNet3DGenerator


def _tiny_net(in_channels: int = 1, output: str = "tanh") -> UNet3DGenerator:
    return UNet3DGenerator(
        in_channels=in_channels,
        enc_channels=[4, 8],
        dec_channels=[8, 4],
        noise_channels=4,
        output=output,
    )


def test_output_matches_input_shape_train():
    net = _tiny_net()
    sparse = torch.zeros(2, 1, 8, 8, 8)
    mask = torch.zeros(2, 1, 8, 8, 8)
    out = net(sparse, mask)
    assert out.shape == (2, 1, 8, 8, 8)


def test_accepts_2x_train_shape():
    net = _tiny_net()
    sparse = torch.zeros(1, 1, 16, 16, 16)
    mask = torch.zeros(1, 1, 16, 16, 16)
    out = net(sparse, mask)
    assert out.shape == (1, 1, 16, 16, 16)


def test_rejects_non_stride_divisible_shape():
    net = _tiny_net()
    sparse = torch.zeros(1, 1, 9, 8, 8)
    mask = torch.zeros(1, 1, 9, 8, 8)
    with pytest.raises(ValueError):
        net(sparse, mask)


def test_empty_input_runs():
    net = _tiny_net()
    sparse = torch.zeros(1, 1, 8, 8, 8)
    mask = torch.zeros(1, 1, 8, 8, 8)
    out = net(sparse, mask)
    assert out.shape == (1, 1, 8, 8, 8)
    # tanh keeps outputs in [-1, 1]
    assert out.abs().max() <= 1.0


def test_rgb_output():
    net = _tiny_net(in_channels=3)
    sparse = torch.zeros(1, 3, 8, 8, 8)
    mask = torch.zeros(1, 1, 8, 8, 8)
    out = net(sparse, mask)
    assert out.shape == (1, 3, 8, 8, 8)


def test_softmax_output_sums_to_one():
    # For softmax mode in_channels (=classes) > 1 is typical; use C=3.
    net = _tiny_net(in_channels=3, output="softmax")
    sparse = torch.zeros(1, 3, 8, 8, 8)
    mask = torch.zeros(1, 1, 8, 8, 8)
    out = net(sparse, mask)
    assert torch.allclose(out.sum(dim=1), torch.ones_like(out.sum(dim=1)), atol=1e-5)


def test_total_stride_property():
    net = _tiny_net()
    assert net.total_stride == 4  # 2 downsampling stages × stride 2
