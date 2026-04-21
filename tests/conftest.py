# tests/conftest.py
import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf


@pytest.fixture
def sample_voxel_path(tmp_path):
    path = tmp_path / "vol.npy"
    rng = np.random.default_rng(0)
    vol = rng.integers(0, 256, size=(32, 32, 32), dtype=np.uint8)
    np.save(path, vol)
    return str(path)


@pytest.fixture
def tiny_cfg(sample_voxel_path) -> DictConfig:
    return OmegaConf.create(
        {
            "data": {
                "voxel_path": sample_voxel_path,
                "train_shape": [8, 8, 8],
                "in_channels": 1,
                "steps_per_epoch": 4,
            },
            "anchor": {
                "axis": 0,
                "empty_prob": 0.2,
                "full_prob": 0.2,
                "sparse_min": 1,
                "sparse_max": None,
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
