from __future__ import annotations

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf
from PIL import Image


@pytest.fixture
def sample_image_path(tmp_path):
    path = tmp_path / "sample.png"
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    Image.fromarray(image).save(path)
    return str(path)


@pytest.fixture
def tiny_cfg(sample_image_path) -> DictConfig:
    return OmegaConf.create(
        {
            "data": {
                "image_path": sample_image_path,
                "image_size": 16,
                "in_channels": 1,
                "steps_per_epoch": 4,
            },
            "dl": {
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
            },
            "generator": {
                "latent_shape": [8, 2, 2, 2],
                "channels": [8, 4, 1],
                "kernels": [4, 4],
                "strides": [2, 2],
                "paddings": [1, 1],
                "output": "tanh",
            },
            "critic": {
                "channels": [1, 4, 8, 1],
                "kernels": [4, 4, 4],
                "strides": [2, 2, 2],
                "paddings": [1, 1, 0],
            },
            "optimizer": {"lr": 1e-4, "betas": [0.9, 0.99]},
            "trainer": {
                "gp_lambda": 10.0,
                "gen_batch_size": 2,
                "gen_freq": 2,
                "steps": 2,
                "save_freq": 1,
            },
            "device": "cpu",
        }
    )
