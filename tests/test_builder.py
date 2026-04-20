from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from src.model.critic import Critic2D
from src.model.generator import Generator3D
from src.builder import (
    build_critic,
    build_generator,
    build_loaders,
    build_optimizer,
    build_trainer,
    check_channel_consistency,
)


def test_check_consistency_ok(tiny_cfg):
    check_channel_consistency(tiny_cfg)


def test_check_consistency_generator_mismatch(tiny_cfg):
    cfg = OmegaConf.create(OmegaConf.to_container(tiny_cfg, resolve=True))
    cfg.generator.channels[-1] = 3
    with pytest.raises(AssertionError):
        check_channel_consistency(cfg)


def test_check_consistency_critic_mismatch(tiny_cfg):
    cfg = OmegaConf.create(OmegaConf.to_container(tiny_cfg, resolve=True))
    cfg.critic.channels[0] = 3
    with pytest.raises(AssertionError):
        check_channel_consistency(cfg)


def test_build_generator_type(tiny_cfg):
    g = build_generator(tiny_cfg)
    assert isinstance(g, Generator3D)


def test_build_critic_type(tiny_cfg):
    c = build_critic(tiny_cfg)
    assert isinstance(c, Critic2D)


def test_build_optimizer_is_adam(tiny_cfg):
    g = build_generator(tiny_cfg)
    opt = build_optimizer(tiny_cfg, g.parameters())
    assert isinstance(opt, torch.optim.Adam)


def test_build_loaders_returns_three_cyclers(tiny_cfg):
    loaders = build_loaders(tiny_cfg)
    assert len(loaders) == 3
    for it in loaders:
        batch = next(it)
        assert batch.shape[0] == tiny_cfg.dl.batch_size


def test_build_trainer_wires_fields(tiny_cfg):
    g = build_generator(tiny_cfg)
    cs = [build_critic(tiny_cfg) for _ in range(3)]
    optG = build_optimizer(tiny_cfg, g.parameters())
    optCs = [build_optimizer(tiny_cfg, c.parameters()) for c in cs]
    loaders = build_loaders(tiny_cfg)
    trainer = build_trainer(tiny_cfg, g, cs, optG, optCs, loaders)
    assert trainer.netG is g
    assert len(trainer.netCs) == 3
    assert trainer.steps == tiny_cfg.trainer.steps
