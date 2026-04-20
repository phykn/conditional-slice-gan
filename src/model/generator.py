from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.init import constant_, trunc_normal_


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Generator3D(nn.Module):
    def __init__(
        self,
        latent_shape: list[int] = [32, 4, 4, 4],
        channels: list[int] = [1024, 512, 128, 32, 1],
        kernels: list[int] = [4, 4, 4, 4, 4],
        strides: list[int] = [2, 2, 2, 2, 2],
        paddings: list[int] = [2, 2, 2, 2, 3],
        output: str = "tanh",
    ) -> None:
        super().__init__()
        assert output in ("tanh", "softmax"), "output must be 'tanh' or 'softmax'"
        self.latent_shape = latent_shape
        self.output = output

        in_ch = channels[:-1]
        out_ch = channels[1:]
        n = len(kernels)
        acts = [True] * (n - 1) + [False]

        self.layers = nn.Sequential(
            *[
                UpBlock3D(
                    in_channels=a,
                    out_channels=b,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    bias=False,
                    act=act,
                )
                for a, b, k, s, p, act in zip(
                    in_ch, out_ch, kernels, strides, paddings, acts
                )
            ]
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.ConvTranspose3d):
            trunc_normal_(m.weight.data, std=0.02)
        elif isinstance(m, nn.BatchNorm3d):
            constant_(m.weight.data, 1.0)
            constant_(m.bias.data, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        if self.output == "tanh":
            return torch.tanh(x)
        return torch.softmax(x, dim=1)

    def sample_noise(self, batch_size: int) -> torch.Tensor:
        device = next(self.parameters()).device
        shape = [batch_size] + list(self.latent_shape)
        return torch.randn(shape, device=device)

    def sample(self, batch_size: int = 1) -> torch.Tensor:
        return self(self.sample_noise(batch_size))
