# src/inference/io.py
from __future__ import annotations

import os

import numpy as np
from omegaconf import OmegaConf


def load_anchor_spec(path: str) -> dict:
    spec = OmegaConf.load(path)
    return OmegaConf.to_container(spec, resolve=True)  # type: ignore[return-value]


def load_anchor_image(path: str) -> np.ndarray:
    """Load a 2D image file into a (H, W) or (H, W, C) numpy array in [-1, 1]."""
    from imrw import imread

    img = imread(path).astype(np.float32)
    return img / 127.5 - 1.0


def save_volume(path: str, volume: np.ndarray) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        np.save(path, volume)
        return
    if ext in (".tif", ".tiff"):
        import tifffile
        # volume is (C, D, H, W); TIFF stack convention: (D, H, W) or (D, H, W, C)
        if volume.shape[0] == 1:
            tifffile.imwrite(path, volume[0].astype(np.float32))
        else:
            tifffile.imwrite(path, np.transpose(volume, (1, 2, 3, 0)).astype(np.float32))
        return
    raise ValueError(f"unsupported output extension: {ext}")
