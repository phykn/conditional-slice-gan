from typing import Iterable

import torch
from omegaconf import DictConfig
from torch.nn import Parameter
from torch.optim import Optimizer

from .data.anchor_sampling import AnchorSpec
from .data.image_dataset import ImageDataset, resolve_pools
from .model.critic import Critic2D
from .model.generator import UNet3DGenerator
from .training.trainer import ConditionalSliceGANTrainer


def validate_config(cfg: DictConfig) -> None:
    _validate_channels(cfg)
    _validate_generator_depth(cfg)
    _validate_anchor(cfg)
    _validate_shape_divisibility(cfg)
    _validate_image_pools(cfg)


def _validate_channels(cfg: DictConfig) -> None:
    in_channels = cfg.data.in_channels
    critic_in = cfg.critic.channels[0]
    if critic_in != in_channels:
        raise ValueError(
            f"critic.channels[0]={critic_in} must equal data.in_channels={in_channels}"
        )


def _validate_generator_depth(cfg: DictConfig) -> None:
    if len(cfg.generator.enc_channels) != len(cfg.generator.dec_channels):
        raise ValueError(
            "generator.enc_channels and dec_channels must have same length"
        )


def _validate_anchor(cfg: DictConfig) -> None:
    axis = cfg.anchor.axis
    if axis not in (0, 1, 2):
        raise ValueError(f"anchor.axis must be 0, 1, or 2; got {axis}")

    empty_prob = cfg.anchor.empty_prob
    if not 0.0 <= empty_prob <= 1.0:
        raise ValueError(f"anchor.empty_prob must be in [0, 1]; got {empty_prob}")

    min_gap = cfg.anchor.min_gap
    if min_gap < 1:
        raise ValueError(f"anchor.min_gap must be >= 1; got {min_gap}")


def _validate_shape_divisibility(cfg: DictConfig) -> None:
    total_stride = 2 ** len(cfg.generator.enc_channels)
    for i, d in enumerate(cfg.data.train_shape):
        if d % total_stride != 0:
            raise ValueError(
                f"train_shape[{i}]={d} not divisible by total stride {total_stride}"
            )


def _validate_image_pools(cfg: DictConfig) -> None:
    images = cfg.data.images
    resolve_pools(
        shared=images.shared,
        axis0=images.axis0,
        axis1=images.axis1,
        axis2=images.axis2,
    )


def build_image_loader(cfg: DictConfig) -> ImageDataset:
    images = cfg.data.images
    pools = resolve_pools(
        shared=images.shared,
        axis0=images.axis0,
        axis1=images.axis1,
        axis2=images.axis2,
    )
    return ImageDataset(
        pools=pools,
        train_shape=tuple(cfg.data.train_shape),
        in_channels=cfg.data.in_channels,
    )


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


def build_anchor_spec(cfg: DictConfig) -> AnchorSpec:
    return AnchorSpec(
        axis=cfg.anchor.axis,
        empty_prob=cfg.anchor.empty_prob,
        min_gap=cfg.anchor.min_gap,
    )


def build_trainer(
    cfg: DictConfig,
    netG: UNet3DGenerator,
    netCs: list[Critic2D],
    optG: Optimizer,
    optCs: list[Optimizer],
    image_loader: ImageDataset,
) -> ConditionalSliceGANTrainer:
    return ConditionalSliceGANTrainer(
        netG=netG,
        netCs=netCs,
        optG=optG,
        optCs=optCs,
        image_loader=image_loader,
        anchor=build_anchor_spec(cfg),
        train_shape=tuple(cfg.data.train_shape),
        in_channels=cfg.data.in_channels,
        batch_size=cfg.dl.batch_size,
        gp_lambda=cfg.trainer.gp_lambda,
        recon_lambda=cfg.trainer.recon_lambda,
        gen_freq=cfg.trainer.gen_freq,
        steps=cfg.trainer.steps,
        save_freq=cfg.trainer.save_freq,
    )
