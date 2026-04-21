import argparse

import torch
from omegaconf import OmegaConf

from src.builder import (
    build_critic,
    build_generator,
    build_image_loader,
    build_optimizer,
    build_trainer,
    build_voxel_loader,
    validate_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--voxel-path", default=None,
                        help="Override data.voxel_path. Pass empty string to force null.")
    parser.add_argument("--images-dir", default=None,
                        help="Override data.images.shared.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.voxel_path is not None:
        cfg.data.voxel_path = None if args.voxel_path == "" else args.voxel_path
    if args.images_dir is not None:
        cfg.data.images.shared = args.images_dir
    validate_config(cfg)

    device = torch.device(cfg.device)
    netG = build_generator(cfg).to(device)
    netCs = [build_critic(cfg).to(device) for _ in range(3)]
    optG = build_optimizer(cfg, netG.parameters())
    optCs = [build_optimizer(cfg, c.parameters()) for c in netCs]
    image_loader = build_image_loader(cfg)
    voxel_loader = build_voxel_loader(cfg)

    trainer = build_trainer(cfg, netG, netCs, optG, optCs, image_loader, voxel_loader)
    OmegaConf.save(cfg, f"{trainer.save_dir}/config.yaml")
    trainer.train()


if __name__ == "__main__":
    main()
