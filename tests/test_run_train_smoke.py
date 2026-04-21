import os
import subprocess
import sys

import numpy as np
from omegaconf import OmegaConf


def test_run_train_end_to_end(tmp_path, monkeypatch):
    # Build a tiny synthetic volume.
    vol_path = tmp_path / "vol.npy"
    rng = np.random.default_rng(0)
    np.save(vol_path, rng.integers(0, 256, size=(16, 16, 16), dtype=np.uint8))

    cfg_path = tmp_path / "config.yaml"
    cfg = {
        "data": {
            "voxel_path": str(vol_path),
            "train_shape": [8, 8, 8],
            "in_channels": 1,
            "steps_per_epoch": 4,
        },
        "anchor": {
            "axis": 0,
            "empty_prob": 0.33,
            "full_prob": 0.33,
            "sparse_min": 1,
            "sparse_max": None,
        },
        "dl": {"batch_size": 2, "num_workers": 0, "pin_memory": False},
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
    OmegaConf.save(OmegaConf.create(cfg), cfg_path)

    monkeypatch.chdir(tmp_path)
    res = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), "..", "run_train.py"),
         "--config", str(cfg_path)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"

    runs = sorted((tmp_path / "run").iterdir())
    assert runs, "no run directory created"
    last = runs[-1]
    assert (last / "config.yaml").exists()
    assert (last / "weights" / "generator.pth").exists()
    for i in range(3):
        assert (last / "weights" / f"critic_{i}.pth").exists()
