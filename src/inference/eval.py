import numpy as np


def predictor_output_to_uint8(out_float: np.ndarray) -> np.ndarray:
    """Convert predictor float output in [-1, 1] to uint8 in [0, 255]."""
    return (((out_float + 1.0) / 2.0) * 255.0).clip(0, 255).astype(np.uint8)
