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


def test_end_to_end_two_steps(tiny_cfg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    device = torch.device("cpu")

    netG = build_generator(tiny_cfg).to(device)
    netCs = [build_critic(tiny_cfg).to(device) for _ in range(3)]
    optG = build_optimizer(tiny_cfg, netG.parameters())
    optCs = [build_optimizer(tiny_cfg, c.parameters()) for c in netCs]
    loaders = build_loaders(tiny_cfg)

    trainer = build_trainer(tiny_cfg, netG, netCs, optG, optCs, loaders)
    trainer.train()

    assert os.path.exists(os.path.join(trainer.save_dir, "logs"))
    weights = os.path.join(trainer.save_dir, "weights")
    assert os.path.exists(os.path.join(weights, "generator.pth"))
    for i in range(3):
        assert os.path.exists(os.path.join(weights, f"critic_{i}.pth"))
