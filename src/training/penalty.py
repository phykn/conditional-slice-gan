import torch
from einops import rearrange

from ..model.critic import Critic2D


def gradient_penalty(
    netC: Critic2D,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    gp_lambda: float = 10.0,
) -> torch.Tensor:
    """WGAN-GP penalty. Fake batch is subsampled to match real batch size.

    With the current trainer `n_real == n_fake` always (real count is ``B * S_axis``
    and fake is slice-expanded to the same size), so the randperm below is a no-op.
    The subsample is kept as a defensive path: if callers ever pass asymmetric
    batches (e.g. slice-expanding across multiple axes), interpolation still requires
    matching sizes.
    """
    n_real = real_data.size(0)
    n_fake = fake_data.size(0)
    assert n_real <= n_fake, "fake batch must be at least as large as real batch"
    device = real_data.device

    idx = torch.randperm(n_fake, device=device)[:n_real]
    fake = fake_data[idx]

    alpha = torch.rand(n_real, 1, 1, 1, device=device)
    inputs = alpha * real_data.detach() + (1.0 - alpha) * fake.detach()
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
