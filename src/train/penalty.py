import torch
from einops import rearrange
from ..model.critic import Critic


def cal_gp(
    netC: Critic,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    gp_lambda: float = 10.0,
) -> torch.Tensor:

    real_data_size = real_data.size(0)
    fake_data_size = fake_data.size(0)
    assert real_data_size <= fake_data_size

    device = real_data.device

    index = torch.randperm(fake_data_size)[:real_data_size]
    fake_data = fake_data[index]

    alpha = torch.rand(real_data.shape[0], device=device)
    alpha = alpha[:, None, None, None] * torch.ones_like(real_data)

    inputs = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    inputs.requires_grad_(True)

    outputs = netC(inputs.float())
    grad_outputs = torch.ones(outputs.size(), device=device)

    grad = torch.autograd.grad(
        inputs=inputs,
        outputs=outputs,
        grad_outputs=grad_outputs,
        create_graph=True,
    )[0]

    grad = rearrange(grad, "b c h w -> b (c h w)")
    norm = grad.norm(p=2, dim=1)
    return gp_lambda * ((norm - 1) ** 2).mean()
