import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf


@pytest.fixture
def sample_image_dir(tmp_path):
    import cv2

    d = tmp_path / "images"
    d.mkdir()
    rng = np.random.default_rng(42)
    for i in range(3):
        img = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        cv2.imwrite(str(d / f"img_{i}.png"), img)
    return str(d)


@pytest.fixture
def tiny_cfg(sample_image_dir) -> DictConfig:
    return OmegaConf.create(
        {
            "data": {
                "train_shape": [8, 8, 8],
                "in_channels": 1,
                "images": {
                    "shared": sample_image_dir,
                    "axis0": None,
                    "axis1": None,
                    "axis2": None,
                },
            },
            "anchor": {
                "axis": 0,
                "empty_prob": 0.2,
                "min_gap": 1,
            },
            "dl": {
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
            },
            "generator": {
                "enc_channels": [4, 8],
                "dec_channels": [8, 4],
                "noise_channels": 4,
                "output": "tanh",
            },
            "critic": {
                "channels": [1, 4, 8, 1],
                "kernels": [4, 4, 2],
                "strides": [2, 2, 2],
                "paddings": [1, 1, 0],
            },
            "optimizer": {"lr": 1e-4, "betas": [0.9, 0.99]},
            "trainer": {
                "gp_lambda": 10.0,
                "recon_lambda": 10.0,
                "gen_freq": 2,
                "steps": 2,
                "save_freq": 1,
            },
            "device": "cpu",
        }
    )
