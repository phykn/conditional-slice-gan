import os

import numpy as np
import torch
from omegaconf import OmegaConf

from ..builder import build_generator
from ..data.anchor_sampling import axis_index
from ..data.image_dataset import normalize_image


class Predictor:
    def __init__(self, run_dir: str, device: str = "cuda") -> None:
        self.device = torch.device(device)
        self.cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
        self.train_shape = tuple(self.cfg.data.train_shape)
        self.anchor_axis = int(self.cfg.anchor.axis)
        self.in_channels = int(self.cfg.data.in_channels)

        self.netG = build_generator(self.cfg).to(self.device)
        state = torch.load(
            os.path.join(run_dir, "weights", "generator.pth"),
            map_location=self.device,
        )
        self.netG.load_state_dict(state)
        self.netG.eval()

    def _validate_shape(self, shape: tuple[int, int, int]) -> None:
        stride = self.netG.total_stride
        for i, (s, t) in enumerate(zip(shape, self.train_shape)):
            if s > 2 * t:
                raise ValueError(f"shape[{i}]={s} exceeds 2× train_shape[{i}]={t}")
            if s % stride != 0:
                raise ValueError(
                    f"shape[{i}]={s} not divisible by total stride {stride}"
                )

    def _prepare_anchor(self, img: np.ndarray) -> np.ndarray:
        """uint8 (H, W) / (H, W, C) anchor -> (C, H, W) float32 in [-1, 1]."""
        normed = normalize_image(img, self.in_channels)
        if normed.ndim == 2:
            return normed[None, :, :]
        if normed.ndim == 3 and normed.shape[-1] == self.in_channels:
            return np.transpose(normed, (2, 0, 1))
        raise ValueError(
            f"anchor image shape {img.shape} not interpretable; "
            f"expected (H, W) or (H, W, {self.in_channels})"
        )

    @torch.no_grad()
    def predict(
        self,
        anchor_images: list[np.ndarray] = [],
        anchor_indices: list[int] = [],
        shape: tuple[int, int, int] | None = None,
        axis: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        if len(anchor_images) != len(anchor_indices):
            raise ValueError("anchor_images and anchor_indices must have same length")

        shape = tuple(shape) if shape is not None else self.train_shape
        if len(shape) != 3:
            raise ValueError(f"shape must be (D, H, W); got {shape}")
        self._validate_shape(shape)
        axis = self.anchor_axis if axis is None else axis
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2; got {axis}")

        D_axis = shape[axis]
        for k in anchor_indices:
            if not 0 <= k < D_axis:
                raise ValueError(f"anchor index {k} out of range [0, {D_axis})")

        sparse_np = np.zeros((self.in_channels,) + shape, dtype=np.float32)
        mask_np = np.zeros((1,) + shape, dtype=np.float32)
        for img, k in zip(anchor_images, anchor_indices):
            slot = axis_index(axis, k)
            sparse_np[slot] = self._prepare_anchor(img)
            mask_np[slot] = 1.0

        sparse = torch.from_numpy(sparse_np).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)

        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(seed)
            bottleneck = [s // self.netG.total_stride for s in shape]
            noise = torch.randn(
                1,
                self.netG.noise_channels,
                *bottleneck,
                device=self.device,
                generator=gen,
            )
        else:
            noise = None

        out = self.netG(sparse, mask, noise=noise).squeeze(0).cpu().numpy()
        # (C, D, H, W) -> (D, H, W, C)
        return out.transpose(1, 2, 3, 0)
