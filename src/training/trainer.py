import os
from datetime import datetime

import torch
from einops import rearrange
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..data.anchor_sampling import (
    axis_index,
    choose_anchor_count,
    sample_positions_with_gap,
)
from ..data.image_dataset import ImageDataset
from ..model.critic import Critic2D
from ..model.generator import UNet3DGenerator
from .penalty import gradient_penalty


def _slice_along_axis(volume: torch.Tensor, axis: int) -> torch.Tensor:
    if axis == 0:
        return rearrange(volume, "b c x y z -> (b x) c y z")
    if axis == 1:
        return rearrange(volume, "b c x y z -> (b y) c x z")
    if axis == 2:
        return rearrange(volume, "b c x y z -> (b z) c x y")
    raise ValueError(f"axis must be 0, 1, or 2; got {axis}")


def _recon_loss(
    fake: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """L1 between fake and target at mask==1 positions, normalized by mask sum."""
    denom = mask.sum()
    if denom.item() == 0:
        return torch.zeros((), device=fake.device, dtype=fake.dtype)
    return (mask * (fake - target).abs()).sum() / (denom * fake.shape[1])


class ConditionalSliceGANTrainer:
    def __init__(
        self,
        netG: UNet3DGenerator,
        netCs: list[Critic2D],
        optG: Optimizer,
        optCs: list[Optimizer],
        image_loader: ImageDataset,
        anchor_axis: int,
        empty_prob: float,
        full_prob: float,
        sparse_min: int,
        sparse_max: int | None,
        min_gap: int,
        train_shape: tuple[int, int, int],
        in_channels: int,
        batch_size: int,
        gp_lambda: float = 10.0,
        recon_lambda: float = 10.0,
        gen_freq: int = 5,
        steps: int = 360000,
        save_freq: int = 1000,
    ) -> None:
        assert len(netCs) == 3
        assert len(optCs) == 3

        self.netG = netG
        self.netCs = netCs
        self.optG = optG
        self.optCs = optCs
        self.image_loader = image_loader
        self.anchor_axis = anchor_axis
        self.empty_prob = empty_prob
        self.full_prob = full_prob
        self.sparse_min = sparse_min
        self.sparse_max = sparse_max
        self.min_gap = min_gap
        self.train_shape = tuple(train_shape)
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.gp_lambda = gp_lambda
        self.recon_lambda = recon_lambda
        self.gen_freq = gen_freq
        self.steps = steps
        self.save_freq = save_freq

        self.device = next(netG.parameters()).device
        self.save_dir = os.path.join("run", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(os.path.join(self.save_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "weights"), exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.save_dir, "logs"))

    def _make_anchor_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a batch-shared K, then synthesize (sparse, mask) per sample by
        placing K 2D images from the anchor-axis pool at distinct positions
        separated by at least `min_gap`."""
        D, H, W = self.train_shape
        D_axis = self.train_shape[self.anchor_axis]
        K = choose_anchor_count(
            D_axis, self.empty_prob, self.full_prob, self.sparse_min, self.sparse_max,
        )

        B = self.batch_size
        sparse = torch.zeros((B, self.in_channels, D, H, W), device=self.device)
        mask = torch.zeros((B, 1, D, H, W), device=self.device)

        if K == 0:
            return sparse, mask

        imgs = self.image_loader.sample(self.anchor_axis, B * K).to(self.device)
        imgs = imgs.view(B, K, *imgs.shape[1:])

        for b in range(B):
            positions = sample_positions_with_gap(D_axis, K, self.min_gap)
            for k, p in enumerate(positions):
                slot = (b,) + axis_index(self.anchor_axis, p)
                sparse[slot] = imgs[b, k]
                mask[slot] = 1.0

        return sparse, mask

    def _sample_batch(
        self, axis: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sparse, mask = self._make_anchor_batch()
        S_axis = self.train_shape[axis]
        real_2d = self.image_loader.sample(
            axis, count=self.batch_size * S_axis,
        ).to(self.device)
        return real_2d, sparse, mask

    def step_critic(
        self,
        axis: int,
        real: torch.Tensor,
        sparse: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, float]:
        self.netG.train()
        netC = self.netCs[axis]
        optC = self.optCs[axis]
        netC.train()
        optC.zero_grad(set_to_none=True)

        with torch.no_grad():
            fake_3d = self.netG(sparse, mask)
        fake = _slice_along_axis(fake_3d, axis)

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

    def step_generator(
        self, sparse: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, float]:
        self.netG.train()
        self.optG.zero_grad(set_to_none=True)

        fake_3d = self.netG(sparse, mask)
        scores = [self.netCs[a](_slice_along_axis(fake_3d, a)).mean() for a in range(3)]
        adv = -torch.stack(scores).mean()

        rec = _recon_loss(fake_3d, sparse, mask)

        loss = adv + self.recon_lambda * rec
        loss.backward()
        self.optG.step()

        return {
            "generator_loss": loss.item(),
            "adv_loss": adv.item(),
            "recon_loss": rec.item(),
        }

    def step(self, global_step: int) -> dict[str, float]:
        axis = global_step % 3
        real, sparse, mask = self._sample_batch(axis)

        losses = self.step_critic(axis, real, sparse, mask)
        for k, v in losses.items():
            self.writer.add_scalar(f"train/{k}", v, global_step)

        if global_step > 0 and global_step % self.gen_freq == 0:
            gen_losses = self.step_generator(sparse, mask)
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
