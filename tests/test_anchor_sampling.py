# tests/test_anchor_sampling.py
from __future__ import annotations

import torch

from src.data.anchor_sampling import sample_anchors


def _sub_volume(C: int = 1, D: int = 8, H: int = 8, W: int = 8) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(C, D, H, W, generator=g)


def test_empty_regime_K_zero():
    sub = _sub_volume()
    sparse, mask, idx, imgs = sample_anchors(
        sub, anchor_axis=0,
        empty_prob=1.0, full_prob=0.0,
        sparse_min=1, sparse_max=7,
    )
    assert sparse.shape == sub.shape
    assert mask.shape == (1, 8, 8, 8)
    assert torch.all(sparse == 0)
    assert torch.all(mask == 0)
    assert idx == []
    assert imgs == []


def test_full_regime_K_equals_D():
    sub = _sub_volume()
    sparse, mask, idx, imgs = sample_anchors(
        sub, anchor_axis=0,
        empty_prob=0.0, full_prob=1.0,
        sparse_min=1, sparse_max=7,
    )
    assert sparse.shape == sub.shape
    assert torch.allclose(sparse, sub)
    assert torch.all(mask == 1)
    assert sorted(idx) == list(range(8))
    assert len(imgs) == 8


def test_sparse_regime_K_in_range():
    sub = _sub_volume()
    torch.manual_seed(1)
    for _ in range(20):
        sparse, mask, idx, imgs = sample_anchors(
            sub, anchor_axis=0,
            empty_prob=0.0, full_prob=0.0,
            sparse_min=3, sparse_max=5,
        )
        assert 3 <= len(idx) <= 5
        assert len(set(idx)) == len(idx)  # unique
        assert all(0 <= k < 8 for k in idx)
        # sparse matches sub at anchor positions
        for k, img in zip(idx, imgs):
            assert torch.allclose(sparse[:, k, :, :], img)
            assert torch.all(mask[:, k, :, :] == 1)
        # zero elsewhere
        non_idx = set(range(8)) - set(idx)
        for k in non_idx:
            assert torch.all(sparse[:, k, :, :] == 0)
            assert torch.all(mask[:, k, :, :] == 0)


def test_axis_1():
    sub = _sub_volume(D=8, H=6, W=8)
    sparse, mask, idx, imgs = sample_anchors(
        sub, anchor_axis=1,
        empty_prob=0.0, full_prob=1.0,
        sparse_min=1, sparse_max=5,
    )
    assert sorted(idx) == list(range(6))
    assert torch.all(mask == 1)
    # each anchor slice is along H
    for k, img in zip(idx, imgs):
        assert img.shape == (1, 8, 8)
        assert torch.allclose(sparse[:, :, k, :], img)


def test_axis_2():
    sub = _sub_volume(D=8, H=8, W=6)
    sparse, mask, idx, imgs = sample_anchors(
        sub, anchor_axis=2,
        empty_prob=0.0, full_prob=1.0,
        sparse_min=1, sparse_max=5,
    )
    assert sorted(idx) == list(range(6))
    for k, img in zip(idx, imgs):
        assert img.shape == (1, 8, 8)
        assert torch.allclose(sparse[:, :, :, k], img)


def test_sparse_max_null_resolves_to_D_minus_1():
    sub = _sub_volume(D=8)
    torch.manual_seed(0)
    for _ in range(20):
        _, _, idx, _ = sample_anchors(
            sub, anchor_axis=0,
            empty_prob=0.0, full_prob=0.0,
            sparse_min=1, sparse_max=None,
        )
        assert 1 <= len(idx) <= 7
