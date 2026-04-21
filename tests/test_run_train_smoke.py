import os
import subprocess
import sys

import cv2
import numpy as np
from omegaconf import OmegaConf


def _tiny_cfg(img_dir: str) -> dict:
    return {
        "data": {
            "train_shape": [8, 8, 8],
            "in_channels": 1,
            "images": {
                "shared": str(img_dir),
                "axis0": None,
                "axis1": None,
                "axis2": None,
            },
        },
        "anchor": {
            "axis": 0,
            "empty_prob": 0.33,
            "full_prob": 0.33,
            "sparse_min": 1,
            "sparse_max": None,
            "min_gap": 1,
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


def test_run_train_end_to_end(tmp_path, monkeypatch):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        img = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"img_{i}.png"), img)

    cfg_path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create(_tiny_cfg(str(img_dir))), cfg_path)

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


def test_run_train_unconditional(tmp_path, monkeypatch):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rng = np.random.default_rng(3)
    for i in range(2):
        img = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"img_{i}.png"), img)

    cfg = _tiny_cfg(str(img_dir))
    cfg["anchor"]["empty_prob"] = 1.0
    cfg["anchor"]["full_prob"] = 0.0

    cfg_path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create(cfg), cfg_path)

    monkeypatch.chdir(tmp_path)
    res = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), "..", "run_train.py"),
         "--config", str(cfg_path)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"

    runs = sorted((tmp_path / "run").iterdir())
    last = runs[-1]
    assert (last / "weights" / "generator.pth").exists()
