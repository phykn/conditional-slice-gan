import argparse
import shutil
from itertools import cycle
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from src.builder import (
    build_critic,
    build_generator,
    build_image_loader,
    build_optimizer,
    build_trainer,
    validate_config,
)

RESET = "\033[0m"
BOLD = "\033[1m"
COLORS = ["\033[96m", "\033[92m", "\033[93m", "\033[95m", "\033[94m", "\033[91m"]


def print_cfg(cfg: DictConfig) -> None:
    def walk(node: Any, path: str = "") -> list[str]:
        if isinstance(node, dict):
            return [
                p
                for k, v in node.items()
                for p in walk(v, f"{path}.{k}" if path else k)
            ]
        return [f"{path}={node}"]

    container = OmegaConf.to_container(cfg, resolve=True)
    groups = [(k, walk(v, k)) for k, v in container.items()]
    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    col_w = max(len(p) for _, pairs in groups for p in pairs) + 2
    ncols = max(1, width // col_w)

    for (name, pairs), color in zip(groups, cycle(COLORS)):
        print(f"{color}{BOLD}[{name}]{RESET}")
        for i in range(0, len(pairs), ncols):
            row = pairs[i : i + ncols]
            print(color + "".join(p.ljust(col_w) for p in row) + RESET)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--images-dir", default=None, help="Override data.images.shared."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.images_dir is not None:
        cfg.data.images.shared = args.images_dir
    validate_config(cfg)

    print_cfg(cfg)

    device = torch.device(cfg.device)
    netG = build_generator(cfg).to(device)
    netCs = [build_critic(cfg).to(device) for _ in range(3)]
    optG = build_optimizer(cfg, netG.parameters())
    optCs = [build_optimizer(cfg, c.parameters()) for c in netCs]
    image_loader = build_image_loader(cfg)

    trainer = build_trainer(cfg, netG, netCs, optG, optCs, image_loader)
    OmegaConf.save(cfg, f"{trainer.save_dir}/config.yaml")
    trainer.train()


if __name__ == "__main__":
    main()
