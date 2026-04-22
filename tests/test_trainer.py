import os

import torch

from src.builder import (
    build_critic,
    build_generator,
    build_image_loader,
    build_optimizer,
    build_trainer,
)
from src.training.trainer import _drop_axis0_anchors, _slice_along_axis


def _trainer(tiny_cfg):
    netG = build_generator(tiny_cfg)
    netCs = [build_critic(tiny_cfg) for _ in range(3)]
    optG = build_optimizer(tiny_cfg, netG.parameters())
    optCs = [build_optimizer(tiny_cfg, c.parameters()) for c in netCs]
    image_loader = build_image_loader(tiny_cfg)
    return build_trainer(tiny_cfg, netG, netCs, optG, optCs, image_loader)


def test_step_returns_losses(tiny_cfg):
    t = _trainer(tiny_cfg)
    losses = t.step(global_step=1)
    assert "critic_fake_score" in losses
    assert "critic_real_score" in losses
    assert "gp" in losses
    assert "loss" in losses


def test_step_at_gen_freq_returns_generator_loss(tiny_cfg):
    t = _trainer(tiny_cfg)
    # gen_freq=2: step 2 triggers gen update
    losses = t.step(global_step=2)
    assert "generator_loss" in losses
    assert "adv_loss" in losses
    assert "recon_loss" in losses


def test_empty_regime_skips_recon(tiny_cfg):
    tiny_cfg.anchor.empty_prob = 1.0
    t = _trainer(tiny_cfg)
    losses = t.step(global_step=2)
    assert losses["recon_loss"] == 0.0


def test_sparse_regime_recon_nonzero(tiny_cfg):
    tiny_cfg.anchor.empty_prob = 0.0
    t = _trainer(tiny_cfg)
    losses = t.step(global_step=2)
    assert losses["recon_loss"] > 0.0


def test_train_runs_and_saves(tiny_cfg):
    t = _trainer(tiny_cfg)
    t.train()
    w = os.path.join(t.save_dir, "weights")
    assert os.path.exists(os.path.join(w, "generator.pth"))
    for i in range(3):
        assert os.path.exists(os.path.join(w, f"critic_{i}.pth"))


def test_critic_real_batch_size(tiny_cfg):
    """Real 2D batch size equals B * train_shape[axis] for each critic axis."""
    t = _trainer(tiny_cfg)
    B = tiny_cfg.dl.batch_size
    for axis in range(3):
        real = t._sample_real_2d(axis)
        expected = B * tiny_cfg.data.train_shape[axis]
        assert real.shape[0] == expected, (
            f"axis {axis}: real batch size {real.shape[0]} != B * train_shape[axis] = {expected}"
        )


def test_drop_axis0_anchors_filters_mask_positions():
    B, C, D, H, W = 2, 1, 4, 3, 3
    volume = torch.zeros(B, C, D, H, W)
    for b in range(B):
        for d in range(D):
            volume[b, 0, d] = float(b * 10 + d)

    mask = torch.zeros(B, 1, D, H, W)
    mask[0, 0, 0] = 1.0  # sample 0 anchor at d=0
    mask[0, 0, 2] = 1.0  # sample 0 anchor at d=2
    mask[1, 0, 1] = 1.0  # sample 1 anchor at d=1

    slices = _slice_along_axis(volume, 0)
    kept = _drop_axis0_anchors(slices, mask)

    # sample 0 keeps d=1,3 (vals 1, 3); sample 1 keeps d=0,2,3 (vals 10, 12, 13)
    kept_vals = kept[:, 0, 0, 0].tolist()
    assert kept_vals == [1.0, 3.0, 10.0, 12.0, 13.0]


def test_drop_axis0_anchors_empty_mask_keeps_all():
    B, C, D, H, W = 2, 1, 4, 3, 3
    volume = torch.randn(B, C, D, H, W)
    mask = torch.zeros(B, 1, D, H, W)
    slices = _slice_along_axis(volume, 0)
    kept = _drop_axis0_anchors(slices, mask)
    assert kept.shape[0] == B * D
    assert torch.equal(kept, slices)


def test_sparse_only_along_anchor_axis(tiny_cfg):
    """Sparse/mask should only have nonzero entries at the anchor axis."""
    tiny_cfg.anchor.empty_prob = 0.0
    t = _trainer(tiny_cfg)
    sparse, mask = t._make_anchor_batch()
    # anchor_axis=0, train_shape=(8,8,8). Each planted slice spans full HxW.
    # Check: for each sample, each (h,w) column along axis 0 has the same mask value.
    for b in range(sparse.shape[0]):
        m = mask[b, 0]  # (D, H, W)
        for d in range(m.shape[0]):
            v = m[d, 0, 0].item()
            assert torch.all(m[d] == v), "mask must be constant over H,W at each D"
