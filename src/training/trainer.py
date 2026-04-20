from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer


class SliceGANTrainer:
    """Stub — real implementation arrives in Task 6. Kept small so Task 5 can wire it."""

    def __init__(
        self,
        netG: Any,
        netCs: list[Any],
        optG: Optimizer,
        optCs: list[Optimizer],
        train_loaders: list[Any],
        gp_lambda: float = 10.0,
        gen_batch_size: int = 8,
        gen_freq: int = 5,
        steps: int = 360000,
        save_freq: int = 1000,
    ) -> None:
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
