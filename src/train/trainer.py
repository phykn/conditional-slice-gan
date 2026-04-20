import os
import torch
from typing import Generator
from einops import rearrange
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

from ..model.generator import Generator
from ..model.critic import Critic
from .penalty import cal_gp


class Trainer:
    def __init__(
        self,
        loaders: list[Generator],
        netG: Generator,
        optG: Optimizer,
        netCs: list[Critic],
        optCs: list[Optimizer],
        gp_lambda: float = 10.0,
        batch_gen_size: int = 8,
        train_gen_interval: int = 5,
        max_step: int = 360000,
        save_interval: int = 1000,
        process_bar: bool = False,
    ):
        self.loaders = loaders
        self.netG = netG
        self.optG = optG
        self.netCs = netCs
        self.optCs = optCs
        self.gp_lambda = gp_lambda
        self.batch_gen_size = batch_gen_size
        self.train_gen_interval = train_gen_interval
        self.max_step = max_step
        self.save_interval = save_interval
        self.process_bar = process_bar

        self.device = netG.gen_device()
        self.folder = f"runs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.writer = SummaryWriter(os.path.join(self.folder))

        self.save_folder = os.path.join(self.folder, "weight")
        os.makedirs(self.save_folder, exist_ok=True)

    def set_train(self):
        self.netG.train()
        for netC in self.netCs:
            netC.train()

    def set_eval(self):
        self.netG.eval()
        for netC in self.netCs:
            netC.eval()

    @staticmethod
    def fake_3d_transform(fake_3d, axis=0):
        if axis == 0:
            return rearrange(fake_3d, "b c x y z -> (b x) c y z")
        elif axis == 1:
            return rearrange(fake_3d, "b c x y z -> (b y) c x z")
        elif axis == 2:
            return rearrange(fake_3d, "b c x y z -> (b z) c x y")
        else:
            raise ValueError

    def step_critic(self, axis=0):
        netC = self.netCs[axis]
        optC = self.optCs[axis]

        netC.zero_grad()

        real_data = next(self.loaders[axis])
        real_data = real_data.float().to(self.device)

        fake_3d = self.netG.generate(self.batch_gen_size).detach()
        fake_data = self.fake_3d_transform(fake_3d, axis=axis)

        real_score = netC.score(real_data)
        fake_score = netC.score(fake_data)
        gp = cal_gp(netC, real_data, fake_data, gp_lambda=self.gp_lambda)
        loss = fake_score - real_score + gp

        loss.backward()
        optC.step()

        return {
            "critic_fake_score": fake_score.mean(),
            "critic_real_score": real_score.mean(),
            "wass_dist": real_score.item() - fake_score.item(),
            "gp": gp.item(),
            "loss": loss.item(),
        }

    def step_generator(self):
        self.netG.zero_grad()

        fake_3d = self.netG.generate(self.batch_gen_size)

        loss = 0
        for axis in [0, 1, 2]:
            fake_data = self.fake_3d_transform(fake_3d, axis=axis)
            fake_score = self.netCs[axis].score(fake_data)
            loss -= fake_score / 3

        loss.backward()
        self.optG.step()

        return {"generator_fake_score": loss.item()}

    def step(self, num_step):
        self.set_train()

        axis = num_step % len(self.netCs)

        loss_dict = self.step_critic(axis=axis)
        for k, v in loss_dict.items():
            self.writer.add_scalar(
                tag=f"train/{k}", scalar_value=v, global_step=num_step
            )

        if (num_step > 0) and (num_step % self.train_gen_interval == 0):
            loss_dict = self.step_generator()
            for k, v in loss_dict.items():
                self.writer.add_scalar(
                    tag=f"train/{k}", scalar_value=v, global_step=num_step
                )

    def save_weight(self):
        torch.save(
            obj=self.netG.state_dict(),
            f=os.path.join(self.save_folder, f"generator.pth"),
        )

        for i, critic in enumerate(self.netCs):
            torch.save(
                obj=critic.state_dict(),
                f=os.path.join(self.save_folder, f"critic_{i}.pth"),
            )

    def run(self):
        pbar = tqdm(range(self.max_step), desc="Training", disable=not self.process_bar)
        for num_step in pbar:
            self.step(num_step)

            if (num_step > 0) and (num_step % self.save_interval == 0):
                self.save_weight()

        self.save_weight()
        self.writer.close()
