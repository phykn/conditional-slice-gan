from __future__ import annotations

import numpy as np
import pytest

from src.data.dataset import SliceDataset


def test_dataset_len(sample_image_path):
    ds = SliceDataset(
        image_path=sample_image_path,
        image_size=16,
        in_channels=1,
        steps_per_epoch=7,
    )
    assert len(ds) == 7


def test_dataset_gray_shape_and_range(sample_image_path):
    ds = SliceDataset(
        image_path=sample_image_path, image_size=16, in_channels=1
    )
    item = ds[0]
    assert item.dtype == np.float32
    assert item.shape == (1, 16, 16)
    assert item.min() >= -1.0 - 1e-6
    assert item.max() <= 1.0 + 1e-6


def test_dataset_rgb_shape(sample_image_path):
    ds = SliceDataset(
        image_path=sample_image_path, image_size=16, in_channels=3
    )
    item = ds[0]
    assert item.dtype == np.float32
    assert item.shape == (3, 16, 16)


def test_dataset_invalid_channels(sample_image_path):
    with pytest.raises(AssertionError):
        SliceDataset(image_path=sample_image_path, image_size=16, in_channels=2)
