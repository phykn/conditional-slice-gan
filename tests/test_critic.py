import torch
import torch.nn as nn

from src.model.critic import Critic2D


def _make(in_channels: int = 1) -> Critic2D:
    return Critic2D(
        channels=[in_channels, 4, 8, 1],
        kernels=[4, 4, 4],
        strides=[2, 2, 2],
        paddings=[1, 1, 0],
    )


def test_forward_shape_gray():
    c = _make(in_channels=1)
    x = torch.randn(2, 1, 16, 16)
    out = c(x)
    assert out.dim() == 4
    assert out.shape[0] == 2


def test_forward_shape_rgb():
    c = _make(in_channels=3)
    x = torch.randn(2, 3, 16, 16)
    out = c(x)
    assert out.shape[0] == 2


def test_conv2d_weights_have_init_std_around_0_02():
    c = _make()
    convs = [m for m in c.modules() if isinstance(m, nn.Conv2d)]
    assert convs, "expected Conv2d layers"
    for conv in convs:
        std = conv.weight.data.std().item()
        assert 0.005 < std < 0.05, f"unexpected std {std}"
