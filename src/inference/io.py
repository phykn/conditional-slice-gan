# src/inference/io.py
import os

import numpy as np
from omegaconf import OmegaConf


def load_anchor_spec(path: str) -> dict:
    spec = OmegaConf.load(path)
    return OmegaConf.to_container(spec, resolve=True)  # type: ignore[return-value]


def load_anchor_image(path: str, in_channels: int) -> np.ndarray:
    """Load a 2D image and return a (C, H, W) float32 array in [-1, 1].

    `imrw.imread` expands grayscale to (H, W, 3), so `in_channels` drives
    the final layout: 1 → convert RGB to grayscale and return (1, H, W);
    3 → transpose HWC to CHW and return (3, H, W).
    """
    from imrw import imread
    import cv2

    img = imread(path)
    if in_channels == 1:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img[None, :, :]
    elif in_channels == 3:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = np.transpose(img, (2, 0, 1))
    else:
        raise ValueError(f"in_channels must be 1 or 3; got {in_channels}")
    return (img.astype(np.float32) / 127.5) - 1.0


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
