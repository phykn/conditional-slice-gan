# src/inference/predictor.py
from __future__ import annotations

import os

import numpy as np
import torch
from omegaconf import OmegaConf

from ..model.generator import UNet3DGenerator


class Predictor:
    def __init__(self, run_dir: str, device: str = "cuda") -> None:
        self.run_dir = run_dir
        self.device = torch.device(device)
        self.cfg = OmegaConf.load(os.path.join(run_dir, "config.yaml"))
        self.train_shape = tuple(self.cfg.data.train_shape)
        self.anchor_axis = int(self.cfg.anchor.axis)
        self.in_channels = int(self.cfg.data.in_channels)

        self.netG = UNet3DGenerator(
            in_channels=self.in_channels,
            enc_channels=list(self.cfg.generator.enc_channels),
            dec_channels=list(self.cfg.generator.dec_channels),
            noise_channels=self.cfg.generator.noise_channels,
            output=self.cfg.generator.output,
        ).to(self.device)
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
                raise ValueError(
                    f"shape[{i}]={s} exceeds 2× train_shape[{i}]={t} (max {2 * t})"
                )
            if s % stride != 0:
                raise ValueError(
                    f"shape[{i}]={s} not divisible by total stride {stride}"
                )

    def _to_chw(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img[None, :, :]
        if img.ndim == 3 and img.shape[0] == self.in_channels:
            return img
        if img.ndim == 3 and img.shape[-1] == self.in_channels:
            return np.transpose(img, (2, 0, 1))
        raise ValueError(f"anchor image shape {img.shape} not interpretable")

    @torch.no_grad()
    def predict(
        self,
        anchor_images: list[np.ndarray],
        anchor_indices: list[int],
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

        C = self.in_channels
        sparse_np = np.zeros((C,) + shape, dtype=np.float32)
        mask_np = np.zeros((1,) + shape, dtype=np.float32)
        for img, k in zip(anchor_images, anchor_indices):
            img_chw = self._to_chw(img.astype(np.float32))
            if axis == 0:
                sparse_np[:, k, :, :] = img_chw
                mask_np[:, k, :, :] = 1.0
            elif axis == 1:
                sparse_np[:, :, k, :] = img_chw
                mask_np[:, :, k, :] = 1.0
            else:
                sparse_np[:, :, :, k] = img_chw
                mask_np[:, :, :, k] = 1.0

        sparse = torch.from_numpy(sparse_np).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)

        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(seed)
            bottleneck = [s // self.netG.total_stride for s in shape]
            noise = torch.randn(
                1, self.netG.noise_channels, *bottleneck,
                device=self.device, generator=gen,
            )
        else:
            noise = None

        out = self.netG(sparse, mask, noise=noise)
        return out.squeeze(0).cpu().numpy()
