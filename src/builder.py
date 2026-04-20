# src/builder.py
from __future__ import annotations

from itertools import cycle
from typing import Iterable, Iterator

import torch
from omegaconf import DictConfig
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .data.dataset import VoxelDataset
from .model.critic import Critic2D
from .model.generator import UNet3DGenerator
from .training.trainer import ConditionalSliceGANTrainer


def check_channel_consistency(cfg: DictConfig) -> None:
    c = cfg.data.in_channels
    c_in = cfg.critic.channels[0]

    assert c_in == c, f"critic.channels[0]={c_in} must equal data.in_channels={c}"

    assert len(cfg.generator.enc_channels) == len(cfg.generator.dec_channels), (
        "generator.enc_channels and dec_channels must have same length"
    )

    axis = cfg.anchor.axis
    assert axis in (0, 1, 2), f"anchor.axis must be 0, 1, or 2; got {axis}"
    ep = cfg.anchor.empty_prob
    fp = cfg.anchor.full_prob
    assert 0 <= ep, f"anchor.empty_prob must be >= 0; got {ep}"
    assert 0 <= fp, f"anchor.full_prob must be >= 0; got {fp}"
    assert ep + fp <= 1.0, (
        f"anchor.empty_prob + anchor.full_prob must be <= 1; got {ep + fp}"
    )

    D_axis = cfg.data.train_shape[axis]
    smin = cfg.anchor.sparse_min
    smax = D_axis - 1 if cfg.anchor.sparse_max is None else cfg.anchor.sparse_max
    assert 1 <= smin <= smax <= D_axis - 1, (
        f"anchor sparse range invalid: min={smin} max={smax} D={D_axis}"
    )

    total_stride = 2 ** len(cfg.generator.enc_channels)
    for i, d in enumerate(cfg.data.train_shape):
        assert d % total_stride == 0, (
            f"train_shape[{i}]={d} not divisible by total stride {total_stride}"
        )


def build_loader(cfg: DictConfig) -> Iterator:
    dataset = VoxelDataset(
        voxel_path=cfg.data.voxel_path,
        train_shape=list(cfg.data.train_shape),
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
    return cycle(dl)


def build_generator(cfg: DictConfig) -> UNet3DGenerator:
    return UNet3DGenerator(
        in_channels=cfg.data.in_channels,
        enc_channels=list(cfg.generator.enc_channels),
        dec_channels=list(cfg.generator.dec_channels),
        noise_channels=cfg.generator.noise_channels,
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
    netG: UNet3DGenerator,
    netCs: list[Critic2D],
    optG: Optimizer,
    optCs: list[Optimizer],
    train_loader: Iterator,
) -> ConditionalSliceGANTrainer:
    return ConditionalSliceGANTrainer(
        netG=netG,
        netCs=netCs,
        optG=optG,
        optCs=optCs,
        train_loader=train_loader,
        anchor_axis=cfg.anchor.axis,
        empty_prob=cfg.anchor.empty_prob,
        full_prob=cfg.anchor.full_prob,
        sparse_min=cfg.anchor.sparse_min,
        sparse_max=cfg.anchor.sparse_max,
        gp_lambda=cfg.trainer.gp_lambda,
        recon_lambda=cfg.trainer.recon_lambda,
        gen_freq=cfg.trainer.gen_freq,
        steps=cfg.trainer.steps,
        save_freq=cfg.trainer.save_freq,
    )
