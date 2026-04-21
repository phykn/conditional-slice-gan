import os
from datetime import datetime
from typing import Iterator

import torch
from einops import rearrange
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..data.anchor_sampling import choose_anchor_count, place_anchor_slices
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


def _batch_anchor_sample(
    sub: torch.Tensor,
    anchor_axis: int,
    empty_prob: float,
    full_prob: float,
    sparse_min: int,
    sparse_max: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-sample anchor placement with a batch-shared anchor count K.
    Regime (empty/full/sparse) and K are drawn *once* per batch; indices differ
    per sample."""
    D_axis = sub.shape[2 + anchor_axis]
    K = choose_anchor_count(D_axis, empty_prob, full_prob, sparse_min, sparse_max)

    sparses: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for i in range(sub.shape[0]):
        sp, mk, _, _ = place_anchor_slices(sub[i], anchor_axis, K)
        sparses.append(sp)
        masks.append(mk)

    return torch.stack(sparses), torch.stack(masks)


def _recon_loss(
    fake: torch.Tensor,
    sub: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """L1 between fake and sub at mask==1 positions, normalized by mask sum."""
    denom = mask.sum()
    if denom.item() == 0:
        return torch.zeros((), device=fake.device, dtype=fake.dtype)
    # Broadcast mask across C.
    return (mask * (fake - sub).abs()).sum() / (denom * fake.shape[1])


class ConditionalSliceGANTrainer:
    def __init__(
        self,
        netG: UNet3DGenerator,
        netCs: list[Critic2D],
        optG: Optimizer,
        optCs: list[Optimizer],
        train_loader: Iterator,
        anchor_axis: int,
        empty_prob: float,
        full_prob: float,
        sparse_min: int,
        sparse_max: int | None,
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
        self.train_loader = train_loader
        self.anchor_axis = anchor_axis
        self.empty_prob = empty_prob
        self.full_prob = full_prob
        self.sparse_min = sparse_min
        self.sparse_max = sparse_max
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

    def _sample_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sub = next(self.train_loader).float().to(self.device)
        sparse, mask = _batch_anchor_sample(
            sub,
            self.anchor_axis,
            self.empty_prob,
            self.full_prob,
            self.sparse_min,
            self.sparse_max,
        )
        return sub, sparse.to(self.device), mask.to(self.device)

    def step_critic(self, axis: int, sub: torch.Tensor, sparse: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
        self.netG.train()
        netC = self.netCs[axis]
        optC = self.optCs[axis]
        netC.train()
        optC.zero_grad(set_to_none=True)

        real = _slice_along_axis(sub, axis)
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

    def step_generator(self, sub: torch.Tensor, sparse: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
        self.netG.train()
        self.optG.zero_grad(set_to_none=True)

        fake_3d = self.netG(sparse, mask)
        scores = [self.netCs[a](_slice_along_axis(fake_3d, a)).mean() for a in range(3)]
        adv = -torch.stack(scores).mean()

        rec = _recon_loss(fake_3d, sub, mask)

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
        sub, sparse, mask = self._sample_batch()

        losses = self.step_critic(axis, sub, sparse, mask)
        for k, v in losses.items():
            self.writer.add_scalar(f"train/{k}", v, global_step)

        if global_step > 0 and global_step % self.gen_freq == 0:
            gen_losses = self.step_generator(sub, sparse, mask)
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
