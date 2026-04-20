from __future__ import annotations

import os
from datetime import datetime
from typing import Iterator

import torch
from einops import rearrange
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..model.critic import Critic2D
from ..model.generator import Generator3D
from .penalty import gradient_penalty


class SliceGANTrainer:
    def __init__(
        self,
        netG: Generator3D,
        netCs: list[Critic2D],
        optG: Optimizer,
        optCs: list[Optimizer],
        train_loaders: list[Iterator],
        gp_lambda: float = 10.0,
        gen_batch_size: int = 8,
        gen_freq: int = 5,
        steps: int = 360000,
        save_freq: int = 1000,
    ) -> None:
        assert len(netCs) == 3
        assert len(optCs) == 3
        assert len(train_loaders) == 3

        self.netG = netG
        self.netCs = netCs
        self.optG = optG
        self.optCs = optCs
        self.train_loaders = train_loaders
        self.gp_lambda = gp_lambda
        self.gen_batch_size = gen_batch_size
        self.gen_freq = gen_freq
        self.steps = steps
        self.save_freq = save_freq

        self.device = next(netG.parameters()).device
        self.save_dir = os.path.join(
            "run", datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(os.path.join(self.save_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "weights"), exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.save_dir, "logs"))

    @staticmethod
    def slice_along_axis(volume: torch.Tensor, axis: int) -> torch.Tensor:
        if axis == 0:
            return rearrange(volume, "b c x y z -> (b x) c y z")
        if axis == 1:
            return rearrange(volume, "b c x y z -> (b y) c x z")
        if axis == 2:
            return rearrange(volume, "b c x y z -> (b z) c x y")
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}")

    def step_critic(self, axis: int) -> dict[str, float]:
        self.netG.train()
        netC = self.netCs[axis]
        optC = self.optCs[axis]
        netC.train()
        optC.zero_grad(set_to_none=True)

        real = next(self.train_loaders[axis]).float().to(self.device)
        with torch.no_grad():
            fake_3d = self.netG.sample(self.gen_batch_size)
        fake = self.slice_along_axis(fake_3d, axis)

        real_score = netC(real).mean()
        fake_score = netC(fake).mean()
        gp = gradient_penalty(netC, real, fake, gp_lambda=self.gp_lambda)
        loss = fake_score - real_score + gp

        loss.backward()
        optC.step()

        return {
            "critic_fake_score": fake_score.item(),
            "critic_real_score": real_score.item(),
            "wass_dist": real_score.item() - fake_score.item(),
            "gp": gp.item(),
            "loss": loss.item(),
        }

    def step_generator(self) -> dict[str, float]:
        self.netG.train()
        self.optG.zero_grad(set_to_none=True)

        fake_3d = self.netG.sample(self.gen_batch_size)
        scores = [
            self.netCs[a](self.slice_along_axis(fake_3d, a)).mean() for a in range(3)
        ]
        loss = -torch.stack(scores).mean()
        loss.backward()
        self.optG.step()

        return {"generator_loss": loss.item()}

    def step(self, global_step: int) -> dict[str, float]:
        axis = global_step % 3
        losses = self.step_critic(axis=axis)
        for k, v in losses.items():
            self.writer.add_scalar(f"train/{k}", v, global_step)

        if global_step > 0 and global_step % self.gen_freq == 0:
            gen_losses = self.step_generator()
            for k, v in gen_losses.items():
                self.writer.add_scalar(f"train/{k}", v, global_step)
            losses.update(gen_losses)

        return losses

    def save(self) -> None:
        w = os.path.join(self.save_dir, "weights")
        torch.save(self.netG.state_dict(), os.path.join(w, "generator.pth"))
        for i, c in enumerate(self.netCs):
            torch.save(c.state_dict(), os.path.join(w, f"critic_{i}.pth"))

    def train(self) -> None:
        bar = tqdm(range(self.steps), desc="Training")
        for global_step in bar:
            losses = self.step(global_step)
            bar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})
            if global_step > 0 and global_step % self.save_freq == 0:
                self.save()
        self.save()
        self.writer.flush()
        self.writer.close()
