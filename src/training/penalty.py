import torch
from einops import rearrange

from ..model.critic import Critic2D


def gradient_penalty(
    netC: Critic2D,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    gp_lambda: float = 10.0,
) -> torch.Tensor:
    """WGAN-GP penalty. Interpolation needs matching batch sizes, so whichever
    side is larger is randomly subsampled down to ``min(n_real, n_fake)``."""
    n_real = real_data.size(0)
    n_fake = fake_data.size(0)
    n = min(n_real, n_fake)
    assert n > 0, "real and fake batches must be non-empty"
    device = real_data.device

    real = real_data
    if n_real > n:
        real = real_data[torch.randperm(n_real, device=device)[:n]]

    fake = fake_data
    if n_fake > n:
        fake = fake_data[torch.randperm(n_fake, device=device)[:n]]

    alpha = torch.rand(n, 1, 1, 1, device=device)
    inputs = alpha * real.detach() + (1.0 - alpha) * fake.detach()
    inputs.requires_grad_(True)

    outputs = netC(inputs)
    grad = torch.autograd.grad(
        inputs=inputs,
        outputs=outputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
    )[0]
    grad = rearrange(grad, "b c h w -> b (c h w)")
    norm = grad.norm(p=2, dim=1)
    return gp_lambda * ((norm - 1.0) ** 2).mean()
