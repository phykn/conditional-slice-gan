import random

import torch


def sample_anchors(
    sub_volume: torch.Tensor,
    anchor_axis: int,
    empty_prob: float,
    full_prob: float,
    sparse_min: int,
    sparse_max: int | None,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[torch.Tensor]]:
    assert sub_volume.ndim == 4, "expected (C, D, H, W)"
    assert anchor_axis in (0, 1, 2), f"anchor_axis must be 0, 1, or 2; got {anchor_axis}"
    assert empty_prob >= 0.0 and full_prob >= 0.0
    assert empty_prob + full_prob <= 1.0

    spatial = sub_volume.shape[1:]  # (D, H, W)
    D_axis = spatial[anchor_axis]

    smax = D_axis - 1 if sparse_max is None else sparse_max
    assert 1 <= sparse_min <= smax <= D_axis - 1

    r = random.random()
    if r < empty_prob:
        K = 0
    elif r < empty_prob + full_prob:
        K = D_axis
    else:
        K = random.randint(sparse_min, smax)

    if K == 0:
        indices: list[int] = []
    elif K == D_axis:
        indices = list(range(D_axis))
    else:
        indices = random.sample(range(D_axis), K)

    sparse = torch.zeros_like(sub_volume)
    mask = torch.zeros((1, *spatial), dtype=sub_volume.dtype, device=sub_volume.device)
    images: list[torch.Tensor] = []

    for k in indices:
        if anchor_axis == 0:
            img = sub_volume[:, k, :, :]
            sparse[:, k, :, :] = img
            mask[:, k, :, :] = 1
        elif anchor_axis == 1:
            img = sub_volume[:, :, k, :]
            sparse[:, :, k, :] = img
            mask[:, :, k, :] = 1
        else:
            img = sub_volume[:, :, :, k]
            sparse[:, :, :, k] = img
            mask[:, :, :, k] = 1
        images.append(img.clone())

    return sparse, mask, indices, images
