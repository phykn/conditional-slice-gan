from __future__ import annotations

import torch

from src.model.generator import Generator3D


def _make(output: str = "tanh", channels=(8, 4, 1)) -> Generator3D:
    return Generator3D(
        latent_shape=[8, 2, 2, 2],
        channels=list(channels),
        kernels=[4, 4],
        strides=[2, 2],
        paddings=[1, 1],
        output=output,
    )


def test_sample_noise_shape():
    g = _make()
    noise = g.sample_noise(3)
    assert noise.shape == (3, 8, 2, 2, 2)


def test_forward_returns_tensor_with_tanh_range():
    g = _make(output="tanh")
    noise = g.sample_noise(2)
    out = g(noise)
    assert out is not None  # regression: forward used to drop `return`
    assert out.shape[0] == 2
    assert out.shape[1] == 1
    assert out.min() >= -1.0 - 1e-5
    assert out.max() <= 1.0 + 1e-5


def test_forward_softmax_channels_sum_to_one():
    g = _make(output="softmax", channels=(8, 4, 2))
    out = g(g.sample_noise(1))
    sums = out.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_sample_runs():
    g = _make()
    out = g.sample(2)
    assert out is not None
    assert out.shape[0] == 2


def test_bn_weight_initialized_to_one():
    g = _make()
    for m in g.modules():
        if isinstance(m, torch.nn.BatchNorm3d):
            assert torch.allclose(m.weight.data, torch.ones_like(m.weight.data))
            assert torch.allclose(m.bias.data, torch.zeros_like(m.bias.data))
