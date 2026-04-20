import os
import argparse


from src.build import (
    build_config,
    build_dl,
    build_model,
    build_optimizer,
    build_trainer,
)
from src.misc import save_yaml


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    return args


def main(cfg_name, image_path=None):
    cfg = build_config(cfg_name)

    if image_path is not None:
        cfg["data"]["image_path"] = image_path

    dl0, dl1, dl2 = build_dl(cfg)
    netG, netC0, netC1, netC2 = build_model(cfg)
    optG, optC0, optC1, optC2 = build_optimizer(cfg, netG, netC0, netC1, netC2)

    trainer = build_trainer(
        cfg=cfg,
        loaders=[dl0, dl1, dl2],
        netG=netG,
        optG=optG,
        netCs=[netC0, netC1, netC2],
        optCs=[optC0, optC1, optC2],
    )

    save_path = os.path.join(trainer.folder, cfg_name)
    save_yaml(cfg, save_path)

    trainer.run()


if __name__ == "__main__":
    cfg_name = "cfg_gray.yaml"
    args = parse_arguments()

    main(cfg_name, args.path)
