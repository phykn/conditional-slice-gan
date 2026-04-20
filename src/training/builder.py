from __future__ import annotations

from itertools import cycle
from typing import Iterable, Iterator

import torch
from omegaconf import DictConfig
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..data.dataset import SliceDataset
from ..model.critic import Critic2D
from ..model.generator import Generator3D
from .trainer import SliceGANTrainer


def check_channel_consistency(cfg: DictConfig) -> None:
    c = cfg.data.in_channels
    g_in = cfg.generator.channels[0]
    g_out = cfg.generator.channels[-1]
    c_in = cfg.critic.channels[0]
    latent = cfg.generator.latent_shape[0]
    assert g_in == latent, (
        f"generator.channels[0]={g_in} must equal generator.latent_shape[0]={latent}"
    )
    assert g_out == c, (
        f"generator.channels[-1]={g_out} must equal data.in_channels={c}"
    )
    assert c_in == c, (
        f"critic.channels[0]={c_in} must equal data.in_channels={c}"
    )


def build_loaders(cfg: DictConfig) -> list[Iterator]:
    loaders: list[Iterator] = []
    for _ in range(3):
        dataset = SliceDataset(
            image_path=cfg.data.image_path,
            image_size=cfg.data.image_size,
            in_channels=cfg.data.in_channels,
            steps_per_epoch=cfg.data.steps_per_epoch,
        )
        dl = DataLoader(
            dataset=dataset,
            batch_size=cfg.dl.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=cfg.dl.num_workers,
            pin_memory=cfg.dl.pin_memory,
        )
        loaders.append(cycle(dl))
    return loaders


def build_generator(cfg: DictConfig) -> Generator3D:
    return Generator3D(
        latent_shape=list(cfg.generator.latent_shape),
        channels=list(cfg.generator.channels),
        kernels=list(cfg.generator.kernels),
        strides=list(cfg.generator.strides),
        paddings=list(cfg.generator.paddings),
        output=cfg.generator.output,
    )


def build_critic(cfg: DictConfig) -> Critic2D:
    return Critic2D(
        channels=list(cfg.critic.channels),
        kernels=list(cfg.critic.kernels),
        strides=list(cfg.critic.strides),
        paddings=list(cfg.critic.paddings),
    )


def build_optimizer(cfg: DictConfig, params: Iterable[Parameter]) -> Optimizer:
    return torch.optim.Adam(
        params=params,
        lr=cfg.optimizer.lr,
        betas=tuple(cfg.optimizer.betas),
    )


def build_trainer(
    cfg: DictConfig,
    netG: Generator3D,
    netCs: list[Critic2D],
    optG: Optimizer,
    optCs: list[Optimizer],
    train_loaders: list[Iterator],
) -> SliceGANTrainer:
    return SliceGANTrainer(
        netG=netG,
        netCs=netCs,
        optG=optG,
        optCs=optCs,
        train_loaders=train_loaders,
        gp_lambda=cfg.trainer.gp_lambda,
        gen_batch_size=cfg.trainer.gen_batch_size,
        gen_freq=cfg.trainer.gen_freq,
        steps=cfg.trainer.steps,
        save_freq=cfg.trainer.save_freq,
    )
