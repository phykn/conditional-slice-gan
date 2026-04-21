import numpy as np
import torch
from omegaconf import OmegaConf

from src.builder import build_critic, build_generator
from src.inference.predictor import Predictor


def _make_run_dir(tmp_path, tiny_cfg) -> str:
    run_dir = tmp_path / "run_mock"
    (run_dir / "weights").mkdir(parents=True)
    OmegaConf.save(tiny_cfg, run_dir / "config.yaml")

    netG = build_generator(tiny_cfg)
    torch.save(netG.state_dict(), run_dir / "weights" / "generator.pth")
    for i in range(3):
        c = build_critic(tiny_cfg)
        torch.save(c.state_dict(), run_dir / "weights" / f"critic_{i}.pth")
    return str(run_dir)


def test_predict_unconditional(tmp_path, tiny_cfg):
    p = Predictor(_make_run_dir(tmp_path, tiny_cfg), device="cpu")
    out = p.predict(anchor_images=[], anchor_indices=[], seed=0)
    assert out.shape == (1, 8, 8, 8)
    assert out.dtype == np.float32


def test_predict_sparse(tmp_path, tiny_cfg):
    p = Predictor(_make_run_dir(tmp_path, tiny_cfg), device="cpu")
    anchor = np.zeros((1, 8, 8), dtype=np.float32)
    out = p.predict(anchor_images=[anchor], anchor_indices=[0], seed=0)
    assert out.shape == (1, 8, 8, 8)
    # Plumbing only — no pixel-exact assertion (training objective, not inference guarantee).


def test_predict_full_identity(tmp_path, tiny_cfg):
    p = Predictor(_make_run_dir(tmp_path, tiny_cfg), device="cpu")
    D = 8
    anchors = [np.full((1, 8, 8), i / D, dtype=np.float32) for i in range(D)]
    out = p.predict(
        anchor_images=anchors,
        anchor_indices=list(range(D)),
        seed=0,
    )
    assert out.shape == (1, 8, 8, 8)


def test_predict_rejects_shape_over_2x(tmp_path, tiny_cfg):
    p = Predictor(_make_run_dir(tmp_path, tiny_cfg), device="cpu")
    import pytest
    with pytest.raises(ValueError):
        p.predict(anchor_images=[], anchor_indices=[], shape=(32, 8, 8))  # 4× on axis 0


def test_predict_accepts_2x(tmp_path, tiny_cfg):
    p = Predictor(_make_run_dir(tmp_path, tiny_cfg), device="cpu")
    out = p.predict(anchor_images=[], anchor_indices=[], shape=(16, 16, 16), seed=0)
    assert out.shape == (1, 16, 16, 16)


def test_predict_seed_reproducible(tmp_path, tiny_cfg):
    p = Predictor(_make_run_dir(tmp_path, tiny_cfg), device="cpu")
    a = p.predict(anchor_images=[], anchor_indices=[], seed=42)
    b = p.predict(anchor_images=[], anchor_indices=[], seed=42)
    assert np.allclose(a, b)
