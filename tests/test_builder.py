import pytest

from src.builder import (
    build_critic,
    build_generator,
    build_loader,
    build_optimizer,
    build_trainer,
    validate_config,
)
from src.model.generator import UNet3DGenerator
from src.model.critic import Critic2D


def test_validate_config_ok(tiny_cfg):
    validate_config(tiny_cfg)  # should not raise


def test_validate_config_mismatch(tiny_cfg):
    tiny_cfg.critic.channels = [3, 4, 8, 1]  # mismatch
    with pytest.raises(AssertionError):
        validate_config(tiny_cfg)


def test_check_axis_range(tiny_cfg):
    tiny_cfg.anchor.axis = 3
    with pytest.raises(AssertionError):
        validate_config(tiny_cfg)


def test_check_regime_probs(tiny_cfg):
    tiny_cfg.anchor.empty_prob = 0.6
    tiny_cfg.anchor.full_prob = 0.6
    with pytest.raises(AssertionError):
        validate_config(tiny_cfg)


def test_check_train_shape_divisible(tiny_cfg):
    tiny_cfg.data.train_shape = [9, 8, 8]  # not divisible by stride 4
    with pytest.raises(AssertionError):
        validate_config(tiny_cfg)


def test_build_generator(tiny_cfg):
    g = build_generator(tiny_cfg)
    assert isinstance(g, UNet3DGenerator)


def test_build_critic(tiny_cfg):
    c = build_critic(tiny_cfg)
    assert isinstance(c, Critic2D)


def test_build_loader_single(tiny_cfg):
    loader = build_loader(tiny_cfg)
    batch = next(loader)
    assert batch.shape == (2, 1, 8, 8, 8)


def test_build_optimizer(tiny_cfg):
    g = build_generator(tiny_cfg)
    opt = build_optimizer(tiny_cfg, g.parameters())
    assert opt.param_groups[0]["lr"] == 1e-4


def test_build_trainer(tiny_cfg):
    g = build_generator(tiny_cfg)
    cs = [build_critic(tiny_cfg) for _ in range(3)]
    optG = build_optimizer(tiny_cfg, g.parameters())
    optCs = [build_optimizer(tiny_cfg, c.parameters()) for c in cs]
    loader = build_loader(tiny_cfg)
    trainer = build_trainer(tiny_cfg, g, cs, optG, optCs, loader)
    assert trainer.recon_lambda == 10.0


def test_validate_rejects_missing_image_pool(tiny_cfg):
    tiny_cfg.data.images.shared = None
    tiny_cfg.data.images.axis1 = None  # axis 1 has no resolution
    with pytest.raises(ValueError, match="axis"):
        validate_config(tiny_cfg)


def test_validate_accepts_per_axis_images(tiny_cfg, sample_image_dir):
    tiny_cfg.data.images.shared = None
    tiny_cfg.data.images.axis0 = sample_image_dir
    tiny_cfg.data.images.axis1 = sample_image_dir
    tiny_cfg.data.images.axis2 = sample_image_dir
    validate_config(tiny_cfg)  # must not raise


def test_validate_rejects_no_voxel_with_anchors(tiny_cfg):
    tiny_cfg.data.voxel_path = None
    tiny_cfg.anchor.empty_prob = 0.0  # conflicts with voxel=None
    with pytest.raises(ValueError, match="empty_prob"):
        validate_config(tiny_cfg)


def test_validate_accepts_no_voxel_with_empty_prob_one(tiny_cfg):
    tiny_cfg.data.voxel_path = None
    tiny_cfg.anchor.empty_prob = 1.0
    tiny_cfg.anchor.full_prob = 0.0
    validate_config(tiny_cfg)  # must not raise
