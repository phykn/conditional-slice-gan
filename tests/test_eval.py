import numpy as np
import torch

from src.inference.eval import (
    generate_fake_volume,
    predictor_output_to_uint8,
    volume_to_axis_batches,
)
from src.inference.predictor import Predictor


def test_predictor_output_to_uint8_range():
    arr = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)[..., None]  # (1, 3, 1)
    out = predictor_output_to_uint8(arr)
    assert out.dtype == np.uint8
    assert out.shape == arr.shape
    assert out[0, 0, 0] == 0
    assert out[0, 2, 0] == 255
    assert 126 <= out[0, 1, 0] <= 128


def test_predictor_output_to_uint8_clips_out_of_range():
    arr = np.array([[-2.0, 2.0]], dtype=np.float32)[..., None]
    out = predictor_output_to_uint8(arr)
    assert out[0, 0, 0] == 0
    assert out[0, 1, 0] == 255


def test_volume_to_axis_batches_grayscale_shapes():
    vol = np.zeros((4, 5, 6), dtype=np.uint8)
    batches = volume_to_axis_batches(vol, in_channels=1, device=torch.device("cpu"))
    assert len(batches) == 3
    assert batches[0].shape == (4, 3, 5, 6)  # axis-0: D slices of (H, W)
    assert batches[1].shape == (5, 3, 4, 6)  # axis-1: H slices of (D, W)
    assert batches[2].shape == (6, 3, 4, 5)  # axis-2: W slices of (D, H)
    for b in batches:
        assert b.dtype == torch.uint8


def test_volume_to_axis_batches_grayscale_channel_replicated():
    vol = np.full((2, 3, 4), 200, dtype=np.uint8)
    batches = volume_to_axis_batches(vol, in_channels=1, device=torch.device("cpu"))
    # all three channels should be identical (replicated)
    b0 = batches[0]
    assert torch.equal(b0[:, 0], b0[:, 1])
    assert torch.equal(b0[:, 1], b0[:, 2])
    assert (b0 == 200).all()


def test_volume_to_axis_batches_rgb():
    vol = np.zeros((4, 5, 6, 3), dtype=np.uint8)
    vol[..., 0] = 10
    vol[..., 1] = 20
    vol[..., 2] = 30
    batches = volume_to_axis_batches(vol, in_channels=3, device=torch.device("cpu"))
    assert batches[0].shape == (4, 3, 5, 6)
    # channels preserved (not replicated)
    assert (batches[0][:, 0] == 10).all()
    assert (batches[0][:, 1] == 20).all()
    assert (batches[0][:, 2] == 30).all()


def test_generate_fake_volume_k_zero(mock_run_dir):
    p = Predictor(str(mock_run_dir), device="cpu")
    gt = np.zeros((8, 8, 8), dtype=np.uint8)
    out = generate_fake_volume(p, gt, k=0, seed=0)
    assert out.shape == (8, 8, 8)
    assert out.dtype == np.uint8


def test_generate_fake_volume_k_positive(mock_run_dir):
    p = Predictor(str(mock_run_dir), device="cpu")
    gt = np.arange(8 * 8 * 8, dtype=np.uint8).reshape(8, 8, 8)
    out = generate_fake_volume(p, gt, k=3, seed=0)
    assert out.shape == (8, 8, 8)
    assert out.dtype == np.uint8


def test_generate_fake_volume_reproducible(mock_run_dir):
    p = Predictor(str(mock_run_dir), device="cpu")
    gt = np.zeros((8, 8, 8), dtype=np.uint8)
    a = generate_fake_volume(p, gt, k=2, seed=42)
    b = generate_fake_volume(p, gt, k=2, seed=42)
    assert np.array_equal(a, b)
