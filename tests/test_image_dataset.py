import numpy as np
import pytest
import torch

from src.data.image_dataset import ImageDataset


@pytest.fixture
def img_dir(tmp_path):
    import cv2
    d = tmp_path / "pool"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        img = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        cv2.imwrite(str(d / f"img_{i}.png"), img)
    return str(d)


def test_axis0_crop_shape(img_dir):
    ds = ImageDataset(
        pools={0: img_dir, 1: img_dir, 2: img_dir},
        train_shape=(8, 16, 24),
        in_channels=1,
    )
    batch = ds.sample(axis=0, count=5)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (5, 1, 16, 24)
    assert batch.dtype == torch.float32
    assert batch.min() >= -1.0 and batch.max() <= 1.0


def test_axis1_crop_shape(img_dir):
    ds = ImageDataset(
        pools={0: img_dir, 1: img_dir, 2: img_dir},
        train_shape=(8, 16, 24),
        in_channels=1,
    )
    batch = ds.sample(axis=1, count=3)
    assert batch.shape == (3, 1, 8, 24)


def test_axis2_crop_shape(img_dir):
    ds = ImageDataset(
        pools={0: img_dir, 1: img_dir, 2: img_dir},
        train_shape=(8, 16, 24),
        in_channels=1,
    )
    batch = ds.sample(axis=2, count=3)
    assert batch.shape == (3, 1, 8, 16)


def test_resolve_pools_shared_only(img_dir):
    from src.data.image_dataset import resolve_pools

    pools = resolve_pools(shared=img_dir, axis0=None, axis1=None, axis2=None)
    assert pools == {0: img_dir, 1: img_dir, 2: img_dir}


def test_resolve_pools_axis_override(tmp_path, img_dir):
    from src.data.image_dataset import resolve_pools

    other = tmp_path / "other"
    other.mkdir()
    pools = resolve_pools(shared=img_dir, axis0=str(other), axis1=None, axis2=None)
    assert pools == {0: str(other), 1: img_dir, 2: img_dir}


def test_resolve_pools_requires_coverage(img_dir):
    from src.data.image_dataset import resolve_pools

    with pytest.raises(ValueError, match="axis 1"):
        resolve_pools(shared=None, axis0=img_dir, axis1=None, axis2=img_dir)
