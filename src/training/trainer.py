import os
from datetime import datetime

import torch
from einops import rearrange
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..data.anchor_sampling import (
    AnchorSpec,
    choose_anchor_count,
    sample_positions_with_gap,
)
from ..data.image_dataset import ImageDataset
from ..model.critic import Critic2D
from ..model.generator import UNet3DGenerator
from .penalty import gradient_penalty


def _slice_along_axis(volume: torch.Tensor, axis: int) -> torch.Tensor:
    """Flatten a 3D volume into 2D slices along `axis`, merging slice index into batch.

    Input: ``(B, C, X, Y, Z)``. Output: ``(B * S_axis, C, H', W')`` where S_axis is the
    spatial size along `axis` and ``(H', W')`` are the remaining two spatial dims in
    their original order. Used to feed a 3D generator output to a 2D critic.
    """
    if axis == 0:
        return rearrange(volume, "b c x y z -> (b x) c y z")
    if axis == 1:
        return rearrange(volume, "b c x y z -> (b y) c x z")
    if axis == 2:
        return rearrange(volume, "b c x y z -> (b z) c x y")
    raise ValueError(f"axis must be 0, 1, or 2; got {axis}")


def _drop_axis0_anchors(
    slices_2d: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Filter out axis-0 slices at anchor positions.

    ``slices_2d`` is ``(B*D, C, H, W)`` from ``_slice_along_axis(..., 0)`` — flattened
    in (b, d) order. ``mask`` is ``(B, 1, D, H, W)`` and is constant 1 over (H, W)
    at anchor indices, so per-(b, d) anchor membership is read from ``mask[:, 0, :, 0, 0]``.
    """
    keep = (mask[:, 0, :, 0, 0] == 0).flatten()
    return slices_2d[keep]


def _recon_loss(
    fake: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """L1 between fake and target at mask==1 positions, normalized by mask sum."""
    denom = mask.sum().clamp(min=1.0)
    return (mask * (fake - target).abs()).sum() / (denom * fake.shape[1])


class ConditionalSliceGANTrainer:
    def __init__(
        self,
        netG: UNet3DGenerator,
        netCs: list[Critic2D],
        optG: Optimizer,
        optCs: list[Optimizer],
        image_loader: ImageDataset,
        anchor: AnchorSpec,
        batch_size: int,
        gp_lambda: float = 10.0,
        recon_lambda: float = 1.0,
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
        self.anchor = anchor
        self.train_shape = image_loader.train_shape
        self.in_channels = image_loader.in_channels
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
        placing K 2D images from the axis-0 pool at distinct positions along
        axis 0, separated by at least `min_gap`."""
        D, H, W = self.train_shape
        K = choose_anchor_count(D, self.anchor)

        B = self.batch_size
        sparse = torch.zeros((B, self.in_channels, D, H, W), device=self.device)
        mask = torch.zeros((B, 1, D, H, W), device=self.device)

        if K == 0:
            return sparse, mask

        imgs = self.image_loader.sample(0, B * K).to(self.device)
        imgs = imgs.view(B, K, *imgs.shape[1:])

        for b in range(B):
            positions = sample_positions_with_gap(D, K, self.anchor.min_gap)
            for k, p in enumerate(positions):
                sparse[b, :, p] = imgs[b, k]
                mask[b, :, p] = 1.0

        return sparse, mask

    def _sample_real_2d(self, axis: int) -> torch.Tensor:
        S_axis = self.train_shape[axis]
        return self.image_loader.sample(
            axis,
            count=self.batch_size * S_axis,
        ).to(self.device)

    def _sample_batch(
        self, axis: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sparse, mask = self._make_anchor_batch()
        real_2d = self._sample_real_2d(axis)
        return real_2d, sparse, mask

    def _update_critic(
        self,
        axis: int,
        real: torch.Tensor,
        fake_3d: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, float]:
        netC = self.netCs[axis]
        optC = self.optCs[axis]
        netC.train()
        optC.zero_grad(set_to_none=True)

        fake = _slice_along_axis(fake_3d, axis)
        if axis == 0:
            fake = _drop_axis0_anchors(fake, mask)

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
        scores = []
        for a in range(3):
            fake_a = _slice_along_axis(fake_3d, a)
            if a == 0:
                fake_a = _drop_axis0_anchors(fake_a, mask)
            scores.append(self.netCs[a](fake_a).mean())
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
        sparse, mask = self._make_anchor_batch()

        self.netG.train()
        with torch.no_grad():
            fake_3d = self.netG(sparse, mask)

        per_axis: list[dict[str, float]] = []
        for axis in range(3):
            real_2d = self._sample_real_2d(axis)
            c_losses = self._update_critic(axis, real_2d, fake_3d, mask)
            per_axis.append(c_losses)
            for k, v in c_losses.items():
                self.writer.add_scalar(f"train/axis{axis}/{k}", v, global_step)

        losses = {k: sum(d[k] for d in per_axis) / 3 for k in per_axis[0]}
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
