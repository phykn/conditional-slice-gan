import numpy as np
import torch

from src.inference.eval import predictor_output_to_uint8


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
