import numpy as np
import pytest

from src.inference.predictor import Predictor


def test_predict_unconditional(mock_run_dir):
    p = Predictor(str(mock_run_dir), device="cpu")
    out = p.predict(anchor_images=[], anchor_indices=[], seed=0)
    assert out.shape == (8, 8, 8, 1)
    assert out.dtype == np.float32


def test_predict_sparse(mock_run_dir):
    p = Predictor(str(mock_run_dir), device="cpu")
    anchor = np.zeros((8, 8), dtype=np.uint8)
    out = p.predict(anchor_images=[anchor], anchor_indices=[0], seed=0)
    assert out.shape == (8, 8, 8, 1)
    # Plumbing only — no pixel-exact assertion (training objective, not inference guarantee).


def test_predict_full_identity(mock_run_dir):
    p = Predictor(str(mock_run_dir), device="cpu")
    D = 8
    anchors = [np.full((8, 8), i * 255 // (D - 1), dtype=np.uint8) for i in range(D)]
    out = p.predict(
        anchor_images=anchors,
        anchor_indices=list(range(D)),
        seed=0,
    )
    assert out.shape == (8, 8, 8, 1)


def test_predict_rejects_shape_over_2x(mock_run_dir):
    p = Predictor(str(mock_run_dir), device="cpu")
    with pytest.raises(ValueError):
        p.predict(anchor_images=[], anchor_indices=[], shape=(32, 8, 8))  # 4× on axis 0


def test_predict_accepts_2x(mock_run_dir):
    p = Predictor(str(mock_run_dir), device="cpu")
    out = p.predict(anchor_images=[], anchor_indices=[], shape=(16, 16, 16), seed=0)
    assert out.shape == (16, 16, 16, 1)


def test_predict_seed_reproducible(mock_run_dir):
    p = Predictor(str(mock_run_dir), device="cpu")
    a = p.predict(anchor_images=[], anchor_indices=[], seed=42)
    b = p.predict(anchor_images=[], anchor_indices=[], seed=42)
    assert np.allclose(a, b)
