# tests/test_dataset.py
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.dataset import VoxelDataset


@pytest.fixture
def npy_volume_path(tmp_path):
    path = tmp_path / "vol.npy"
    rng = np.random.default_rng(0)
    vol = rng.integers(0, 256, size=(32, 32, 32), dtype=np.uint8)
    np.save(path, vol)
    return str(path)


@pytest.fixture
def tiff_volume_path(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "vol.tif"
    rng = np.random.default_rng(1)
    vol = rng.integers(0, 256, size=(32, 32, 32), dtype=np.uint8)
    tifffile.imwrite(path, vol)
    return str(path)


def test_npy_roundtrip_shape(npy_volume_path):
    ds = VoxelDataset(
        voxel_path=npy_volume_path,
        train_shape=[8, 16, 16],
        in_channels=1,
        steps_per_epoch=4,
    )
    assert len(ds) == 4
    sample = ds[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (1, 8, 16, 16)
    assert sample.dtype == torch.float32
    assert sample.min() >= -1.0 and sample.max() <= 1.0


def test_tiff_load(tiff_volume_path):
    ds = VoxelDataset(
        voxel_path=tiff_volume_path,
        train_shape=[8, 16, 16],
        in_channels=1,
        steps_per_epoch=2,
    )
    s = ds[0]
    assert s.shape == (1, 8, 16, 16)


def test_rgb_volume(tmp_path):
    path = tmp_path / "vol.npy"
    rng = np.random.default_rng(2)
    vol = rng.integers(0, 256, size=(16, 16, 16, 3), dtype=np.uint8)
    np.save(path, vol)
    ds = VoxelDataset(
        voxel_path=str(path),
        train_shape=[8, 8, 8],
        in_channels=3,
        steps_per_epoch=2,
    )
    s = ds[0]
    assert s.shape == (3, 8, 8, 8)


def test_random_crop_varies(npy_volume_path):
    ds = VoxelDataset(
        voxel_path=npy_volume_path,
        train_shape=[8, 16, 16],
        in_channels=1,
        steps_per_epoch=8,
    )
    samples = torch.stack([ds[i] for i in range(8)])
    # at least two of the eight draws should differ
    assert not torch.allclose(samples[0], samples[1]) or not torch.allclose(samples[0], samples[2])


def test_rejects_crop_larger_than_volume(tmp_path):
    path = tmp_path / "vol.npy"
    np.save(path, np.zeros((8, 8, 8), dtype=np.uint8))
    with pytest.raises(ValueError):
        VoxelDataset(
            voxel_path=str(path),
            train_shape=[16, 8, 8],
            in_channels=1,
            steps_per_epoch=1,
        )
