from __future__ import annotations

import os

import torch

from src.training.builder import (
    build_critic,
    build_generator,
    build_loaders,
    build_optimizer,
    build_trainer,
)
from src.training.trainer import SliceGANTrainer


def _build(tiny_cfg) -> SliceGANTrainer:
    g = build_generator(tiny_cfg)
    cs = [build_critic(tiny_cfg) for _ in range(3)]
    optG = build_optimizer(tiny_cfg, g.parameters())
    optCs = [build_optimizer(tiny_cfg, c.parameters()) for c in cs]
    loaders = build_loaders(tiny_cfg)
    return build_trainer(tiny_cfg, g, cs, optG, optCs, loaders)


def test_slice_along_axis_shapes():
    vol = torch.randn(2, 1, 4, 5, 6)
    assert SliceGANTrainer.slice_along_axis(vol, 0).shape == (2 * 4, 1, 5, 6)
    assert SliceGANTrainer.slice_along_axis(vol, 1).shape == (2 * 5, 1, 4, 6)
    assert SliceGANTrainer.slice_along_axis(vol, 2).shape == (2 * 6, 1, 4, 5)


def test_step_critic_returns_floats(tiny_cfg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trainer = _build(tiny_cfg)
    out = trainer.step_critic(axis=0)
    assert set(out.keys()) >= {
        "critic_fake_score",
        "critic_real_score",
        "wass_dist",
        "gp",
        "loss",
    }
    for v in out.values():
        assert isinstance(v, float)


def test_step_generator_returns_float(tiny_cfg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trainer = _build(tiny_cfg)
    out = trainer.step_generator()
    assert "generator_loss" in out
    assert isinstance(out["generator_loss"], float)


def test_save_writes_weight_files(tiny_cfg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trainer = _build(tiny_cfg)
    trainer.save()
    w = os.path.join(trainer.save_dir, "weights")
    assert os.path.exists(os.path.join(w, "generator.pth"))
    for i in range(3):
        assert os.path.exists(os.path.join(w, f"critic_{i}.pth"))
