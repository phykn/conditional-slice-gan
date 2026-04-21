import os

import numpy as np
import torch


def _load_volume(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(path)
    raise ValueError(f"unsupported volume extension: {ext}")


class VoxelDataset:
    def __init__(
        self,
        voxel_path: str,
        train_shape: list[int],
        in_channels: int = 1,
        steps_per_epoch: int = 512,
    ) -> None:
        assert in_channels in (1, 3), "in_channels must be 1 or 3"
        assert len(train_shape) == 3, "train_shape must be (D, H, W)"

        self.voxel_path = voxel_path
        self.train_shape = tuple(train_shape)
        self.in_channels = in_channels
        self.steps_per_epoch = steps_per_epoch

        vol = _load_volume(voxel_path)
        if vol.ndim == 3:
            if in_channels != 1:
                raise ValueError(
                    f"volume has no channel axis but in_channels={in_channels}"
                )
            vol = vol[..., None]  # (D, H, W, 1)
        elif vol.ndim == 4:
            if vol.shape[-1] != in_channels:
                raise ValueError(
                    f"volume has {vol.shape[-1]} channels, expected {in_channels}"
                )
        else:
            raise ValueError(f"volume must be 3D or 4D; got shape {vol.shape}")

        for i, (d, t) in enumerate(zip(vol.shape[:3], train_shape)):
            if d < t:
                raise ValueError(
                    f"volume dim {i} ({d}) smaller than train_shape ({t})"
                )

        self.volume = (vol.astype(np.float32) / 127.5) - 1.0  # (D, H, W, C) in [-1, 1]

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, idx: int) -> torch.Tensor:
        D, H, W = self.volume.shape[:3]
        td, th, tw = self.train_shape
        d0 = np.random.randint(0, D - td + 1)
        h0 = np.random.randint(0, H - th + 1)
        w0 = np.random.randint(0, W - tw + 1)
        crop = self.volume[d0:d0+td, h0:h0+th, w0:w0+tw, :]  # (td, th, tw, C)

        # Random H-W flips (axes 1 and 2). No D-axis transforms.
        if np.random.rand() < 0.5:
            crop = crop[:, ::-1, :, :]
        if np.random.rand() < 0.5:
            crop = crop[:, :, ::-1, :]
        # Random 90° rotation in H-W plane.
        k = np.random.randint(0, 4)
        if k:
            crop = np.rot90(crop, k=k, axes=(1, 2))

        crop = np.ascontiguousarray(np.transpose(crop, (3, 0, 1, 2)))  # (C, D, H, W)
        return torch.from_numpy(crop).float()
