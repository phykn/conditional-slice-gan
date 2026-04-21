import random

import torch


def choose_anchor_count(
    D_axis: int,
    empty_prob: float,
    full_prob: float,
    sparse_min: int,
    sparse_max: int | None,
) -> int:
    assert empty_prob >= 0.0 and full_prob >= 0.0
    assert empty_prob + full_prob <= 1.0
    smax = D_axis - 1 if sparse_max is None else sparse_max
    assert 1 <= sparse_min <= smax <= D_axis - 1

    r = random.random()
    if r < empty_prob:
        return 0
    if r < empty_prob + full_prob:
        return D_axis
    return random.randint(sparse_min, smax)


def place_anchor_slices(
    sub_volume: torch.Tensor,
    anchor_axis: int,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[torch.Tensor]]:
    assert sub_volume.ndim == 4, "expected (C, D, H, W)"
    assert anchor_axis in (0, 1, 2), f"anchor_axis must be 0, 1, or 2; got {anchor_axis}"

    spatial = sub_volume.shape[1:]  # (D, H, W)
    D_axis = spatial[anchor_axis]
    assert 0 <= K <= D_axis, f"K={K} out of range [0, {D_axis}]"

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
    D_axis = sub_volume.shape[1 + anchor_axis]
    K = choose_anchor_count(D_axis, empty_prob, full_prob, sparse_min, sparse_max)
    return place_anchor_slices(sub_volume, anchor_axis, K)
