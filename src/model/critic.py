import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 2,
        padding: int = 1,
        act: bool = True,
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not norm,
        )
        if norm:
            num_groups = min(8, out_channels)
            assert out_channels % num_groups == 0, (
                f"out_channels={out_channels} must be divisible by "
                f"num_groups={num_groups} for GroupNorm"
            )
            self.norm = nn.GroupNorm(
                num_groups=num_groups, num_channels=out_channels, affine=True
            )
        else:
            self.norm = nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Critic2D(nn.Module):
    def __init__(
        self,
        channels: list[int] = [1, 64, 128, 256, 512, 1],
        kernels: list[int] = [4, 4, 4, 4, 4],
        strides: list[int] = [2, 2, 2, 2, 2],
        paddings: list[int] = [1, 1, 1, 1, 0],
    ) -> None:
        super().__init__()
        in_ch = channels[:-1]
        out_ch = channels[1:]
        n = len(kernels)
        acts = [True] * (n - 1) + [False]
        norms = [True] * (n - 1) + [False]

        self.layers = nn.Sequential(
            *[
                DownBlock2D(
                    in_channels=a,
                    out_channels=b,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    act=act,
                    norm=norm,
                )
                for a, b, k, s, p, act, norm in zip(
                    in_ch, out_ch, kernels, strides, paddings, acts, norms
                )
            ]
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight.data, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
