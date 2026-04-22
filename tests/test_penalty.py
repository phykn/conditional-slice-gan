import torch

from src.model.critic import Critic2D
from src.training.penalty import gradient_penalty


def _critic():
    return Critic2D(
        channels=[1, 4, 8, 1],
        kernels=[4, 4, 4],
        strides=[2, 2, 2],
        paddings=[1, 1, 0],
    )


def test_gradient_penalty_scalar_and_positive():
    netC = _critic()
    real = torch.randn(2, 1, 16, 16)
    fake = torch.randn(10, 1, 16, 16)
    gp = gradient_penalty(netC, real, fake, gp_lambda=10.0)
    assert gp.dim() == 0
    assert gp.item() >= 0.0


def test_gradient_penalty_backward_populates_grads():
    netC = _critic()
    real = torch.randn(2, 1, 16, 16)
    fake = torch.randn(4, 1, 16, 16)
    gp = gradient_penalty(netC, real, fake)
    gp.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in netC.parameters())


def test_gradient_penalty_handles_more_reals_than_fakes():
    netC = _critic()
    real = torch.randn(10, 1, 16, 16)
    fake = torch.randn(2, 1, 16, 16)
    gp = gradient_penalty(netC, real, fake, gp_lambda=10.0)
    assert gp.dim() == 0
    assert gp.item() >= 0.0
