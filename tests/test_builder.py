import pytest

from src.builder import (
    build_critic,
    build_generator,
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
    with pytest.raises(ValueError, match="critic.channels"):
        validate_config(tiny_cfg)


def test_check_axis_range(tiny_cfg):
    tiny_cfg.anchor.axis = 3
    with pytest.raises(ValueError, match="anchor.axis"):
        validate_config(tiny_cfg)


def test_check_empty_prob_range(tiny_cfg):
    tiny_cfg.anchor.empty_prob = 1.5
    with pytest.raises(ValueError, match="empty_prob"):
        validate_config(tiny_cfg)


def test_check_train_shape_divisible(tiny_cfg):
    tiny_cfg.data.train_shape = [9, 8, 8]  # not divisible by stride 4
    with pytest.raises(ValueError, match="divisible"):
        validate_config(tiny_cfg)


def test_check_min_gap_positive(tiny_cfg):
    tiny_cfg.anchor.min_gap = 0
    with pytest.raises(ValueError, match="min_gap"):
        validate_config(tiny_cfg)


def test_build_generator(tiny_cfg):
    g = build_generator(tiny_cfg)
    assert isinstance(g, UNet3DGenerator)


def test_build_critic(tiny_cfg):
    c = build_critic(tiny_cfg)
    assert isinstance(c, Critic2D)


def test_build_image_loader(tiny_cfg):
    from src.builder import build_image_loader
    from src.data.image_dataset import ImageDataset

    loader = build_image_loader(tiny_cfg)
    assert isinstance(loader, ImageDataset)
    batch = loader.sample(axis=0, count=2)
    assert batch.shape == (2, 1, 8, 8)


def test_build_optimizer(tiny_cfg):
    g = build_generator(tiny_cfg)
    opt = build_optimizer(tiny_cfg, g.parameters())
    assert opt.param_groups[0]["lr"] == 1e-4


def test_build_trainer(tiny_cfg):
    from src.builder import build_image_loader

    g = build_generator(tiny_cfg)
    cs = [build_critic(tiny_cfg) for _ in range(3)]
    optG = build_optimizer(tiny_cfg, g.parameters())
    optCs = [build_optimizer(tiny_cfg, c.parameters()) for c in cs]
    image_loader = build_image_loader(tiny_cfg)
    trainer = build_trainer(tiny_cfg, g, cs, optG, optCs, image_loader)
    assert trainer.recon_lambda == 10.0
    assert trainer.anchor.min_gap == 1


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
