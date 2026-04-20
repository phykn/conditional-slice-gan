# run_train.py
from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf

from src.builder import (
    build_critic,
    build_generator,
    build_loader,
    build_optimizer,
    build_trainer,
    check_channel_consistency,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config/default.yaml")
    parser.add_argument("--voxel-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.voxel_path is not None:
        cfg.data.voxel_path = args.voxel_path
    check_channel_consistency(cfg)

    device = torch.device(cfg.device)
    netG = build_generator(cfg).to(device)
    netCs = [build_critic(cfg).to(device) for _ in range(3)]
    optG = build_optimizer(cfg, netG.parameters())
    optCs = [build_optimizer(cfg, c.parameters()) for c in netCs]
    train_loader = build_loader(cfg)

    trainer = build_trainer(cfg, netG, netCs, optG, optCs, train_loader)
    OmegaConf.save(cfg, f"{trainer.save_dir}/config.yaml")
    trainer.train()


if __name__ == "__main__":
    main()
