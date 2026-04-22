from collections.abc import Sequence

import numpy as np
import torch
from tqdm import tqdm

from .predictor import Predictor


def predictor_output_to_uint8(out_float: np.ndarray) -> np.ndarray:
    """Convert predictor float output in [-1, 1] to uint8 in [0, 255]."""
    return (((out_float + 1.0) / 2.0) * 255.0).clip(0, 255).astype(np.uint8)


def volume_to_axis_batches(
    volume: np.ndarray,
    in_channels: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """(D, H, W) or (D, H, W, C) uint8 volume -> list of three per-axis uint8
    tensors shaped (N_i, 3, H_i, W_i) on ``device``, ready for
    FrechetInceptionDistance.update(). Grayscale channels are replicated 1 -> 3."""
    if volume.dtype != np.uint8:
        raise TypeError(f"volume must be uint8, got {volume.dtype}")
    if in_channels == 1:
        if volume.ndim != 3:
            raise ValueError(f"grayscale volume must be (D, H, W); got {volume.shape}")
        vol4 = volume[..., None]  # (D, H, W, 1)
    elif in_channels == 3:
        if volume.ndim != 4 or volume.shape[-1] != 3:
            raise ValueError(f"rgb volume must be (D, H, W, 3); got {volume.shape}")
        vol4 = volume
    else:
        raise ValueError(f"unsupported in_channels={in_channels}")

    batches: list[torch.Tensor] = []
    # axis 0: iterate along D -> slices are (H, W, C)
    # axis 1: iterate along H -> slices are (D, W, C)
    # axis 2: iterate along W -> slices are (D, H, C)
    for axis in (0, 1, 2):
        slices = np.moveaxis(vol4, axis, 0)           # (N, h, w, C)
        t = torch.from_numpy(np.ascontiguousarray(slices))
        t = t.permute(0, 3, 1, 2).contiguous()        # (N, C, h, w)
        if in_channels == 1:
            t = t.expand(-1, 3, -1, -1).contiguous()  # replicate channels 1 -> 3
        batches.append(t.to(device))
    return batches


def generate_fake_volume(
    predictor: Predictor,
    gt_volume: np.ndarray,
    k: int,
    seed: int,
) -> np.ndarray:
    """Generate one fake volume conditioned on ``k`` GT axis-0 slices picked
    at random index positions. k=0 -> unconditioned. Returns uint8 volume
    shaped (D, H, W) for grayscale or (D, H, W, C) for multi-channel."""
    D = gt_volume.shape[0]
    if not 0 <= k <= D:
        raise ValueError(f"k={k} out of range [0, {D}]")
    rng = np.random.default_rng(seed)
    if k == 0:
        anchor_images: list[np.ndarray] = []
        anchor_indices: list[int] = []
    else:
        idx = rng.choice(D, size=k, replace=False)
        anchor_images = [gt_volume[int(i)] for i in idx]
        anchor_indices = [int(i) for i in idx]

    out_float = predictor.predict(
        anchor_images=anchor_images,
        anchor_indices=anchor_indices,
        seed=seed,
    )
    out_uint8 = predictor_output_to_uint8(out_float)
    if predictor.in_channels == 1:
        return out_uint8[..., 0]
    return out_uint8


def sweep_fid_vs_anchor_count(
    predictor: Predictor,
    gt_volume: np.ndarray,
    k_list: Sequence[int],
    n_per_k: int,
    seed_base: int = 0,
    device: torch.device | str = "cuda",
    progress: bool = True,
) -> list[tuple[int, float]]:
    """Sweep K across ``k_list``. For each K, generate ``n_per_k`` volumes,
    slice GT + all fakes along all three axes, feed them to a freshly
    instantiated FrechetInceptionDistance, and record the resulting FID.

    Seed per generation: ``seed_base + k_index * n_per_k + volume_index``
    (``k_index`` is the list position, not K itself)."""
    from torchmetrics.image.fid import FrechetInceptionDistance

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    results: list[tuple[int, float]] = []
    real_batches = volume_to_axis_batches(gt_volume, predictor.in_channels, dev)

    iterator = enumerate(k_list)
    if progress:
        iterator = tqdm(list(iterator), desc="FID sweep", total=len(k_list))

    for k_index, k in iterator:
        fid = FrechetInceptionDistance(feature=2048, normalize=False).to(dev)
        for batch in real_batches:
            fid.update(batch, real=True)
        for v in range(n_per_k):
            seed = seed_base + k_index * n_per_k + v
            fake_vol = generate_fake_volume(predictor, gt_volume, k=k, seed=seed)
            for batch in volume_to_axis_batches(fake_vol, predictor.in_channels, dev):
                fid.update(batch, real=False)
        value = float(fid.compute().item())
        results.append((int(k), value))
    return results
