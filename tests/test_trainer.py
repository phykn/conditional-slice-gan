import os

import torch

from src.builder import (
    build_critic,
    build_generator,
    build_optimizer,
    build_trainer,
)


def _trainer(tiny_cfg):
    from src.builder import build_image_loader, build_voxel_loader

    netG = build_generator(tiny_cfg)
    netCs = [build_critic(tiny_cfg) for _ in range(3)]
    optG = build_optimizer(tiny_cfg, netG.parameters())
    optCs = [build_optimizer(tiny_cfg, c.parameters()) for c in netCs]
    image_loader = build_image_loader(tiny_cfg)
    voxel_loader = build_voxel_loader(tiny_cfg)
    return build_trainer(tiny_cfg, netG, netCs, optG, optCs, image_loader, voxel_loader)


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
    tiny_cfg.anchor.full_prob = 0.0
    t = _trainer(tiny_cfg)
    losses = t.step(global_step=2)
    assert losses["recon_loss"] == 0.0


def test_full_regime_recon_nonzero(tiny_cfg):
    tiny_cfg.anchor.empty_prob = 0.0
    tiny_cfg.anchor.full_prob = 1.0
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
