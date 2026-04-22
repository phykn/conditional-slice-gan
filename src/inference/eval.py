import numpy as np
import torch


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
