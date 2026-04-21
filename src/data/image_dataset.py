import os

import cv2
import numpy as np
import torch


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_images(directory: str) -> list[str]:
    paths = []
    for name in sorted(os.listdir(directory)):
        ext = os.path.splitext(name)[1].lower()
        if ext in _IMAGE_EXTS:
            paths.append(os.path.join(directory, name))
    if not paths:
        raise ValueError(f"no images found in {directory}")
    return paths


def _load_image(path: str, in_channels: int) -> np.ndarray:
    """Load a 2D image and return (C, H, W) float32 in [-1, 1]."""
    from imrw import imread

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


def _axis_crop_hw(axis: int, train_shape: tuple[int, int, int]) -> tuple[int, int]:
    """Return the (H', W') crop size for 2D slices along `axis`."""
    D, H, W = train_shape
    if axis == 0:
        return H, W
    if axis == 1:
        return D, W
    if axis == 2:
        return D, H
    raise ValueError(f"axis must be 0, 1, or 2; got {axis}")


class ImageDataset:
    def __init__(
        self,
        pools: dict[int, str],
        train_shape: tuple[int, int, int],
        in_channels: int,
    ) -> None:
        for a in (0, 1, 2):
            if a not in pools or pools[a] is None:
                raise ValueError(f"missing image pool for axis {a}")
        self.train_shape = tuple(train_shape)
        self.in_channels = in_channels

        self._images: dict[int, list[np.ndarray]] = {}
        for a, directory in pools.items():
            paths = _list_images(directory)
            loaded = [_load_image(p, in_channels) for p in paths]
            _validate_pool_sizes(a, self.train_shape, loaded, paths)
            self._images[a] = loaded

    def sample(self, axis: int, count: int) -> torch.Tensor:
        ch, cw = _axis_crop_hw(axis, self.train_shape)
        pool = self._images[axis]
        out = np.empty((count, self.in_channels, ch, cw), dtype=np.float32)
        for i in range(count):
            img = pool[np.random.randint(len(pool))]
            _, ih, iw = img.shape
            y0 = np.random.randint(0, ih - ch + 1)
            x0 = np.random.randint(0, iw - cw + 1)
            crop = img[:, y0:y0+ch, x0:x0+cw]
            if np.random.rand() < 0.5:
                crop = crop[:, ::-1, :]
            if np.random.rand() < 0.5:
                crop = crop[:, :, ::-1]
            out[i] = np.ascontiguousarray(crop)
        return torch.from_numpy(out)


def _validate_pool_sizes(
    axis: int,
    train_shape: tuple[int, int, int],
    loaded: list[np.ndarray],
    paths: list[str],
) -> None:
    ch, cw = _axis_crop_hw(axis, train_shape)
    for p, img in zip(paths, loaded):
        _, ih, iw = img.shape
        if ih < ch or iw < cw:
            raise ValueError(
                f"image {p} of shape ({ih},{iw}) smaller than "
                f"crop ({ch},{cw}) required for axis {axis}"
            )


def resolve_pools(
    shared: str | None,
    axis0: str | None,
    axis1: str | None,
    axis2: str | None,
) -> dict[int, str]:
    """Resolve per-axis pools, falling back to `shared` when an axis is unset.

    Raises if any axis ends up unresolved.
    """
    candidates = {0: axis0, 1: axis1, 2: axis2}
    resolved: dict[int, str] = {}
    missing: list[int] = []
    for a, v in candidates.items():
        chosen = v if v is not None else shared
        if chosen is None:
            missing.append(a)
        else:
            resolved[a] = chosen
    if missing:
        axes_str = ", ".join(str(a) for a in missing)
        raise ValueError(
            f"no image pool resolved for axis {axes_str}; set data.images.shared "
            f"or the per-axis override"
        )
    return resolved
