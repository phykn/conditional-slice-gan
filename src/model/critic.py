import torch.nn as nn
from torch.nn.init import trunc_normal_


class BlockConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 2,
        padding: int = 1,
        bias: bool = False,
        act: bool = True,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class Critic(nn.Module):
    def __init__(
        self,
        channels: list[int] = [2, 64, 128, 256, 512, 1],
        kernels: list[int] = [4, 4, 4, 4, 4],
        strides: list[int] = [2, 2, 2, 2, 2],
        paddings: list[int] = [1, 1, 1, 1, 0],
    ):
        super().__init__()

        in_channels = channels[:-1]
        out_channels = channels[1:]

        n_layers = len(kernels)
        acts = [True] * (n_layers - 1) + [False]

        layers = []
        for in_channel, out_channel, kernel, stride, padding, act in zip(
            in_channels, out_channels, kernels, strides, paddings, acts
        ):
            layer = BlockConv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=False,
                act=act,
            )
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight.data, std=0.02)

    def forward(self, x):
        return self.layers(x)

    def score(self, x):
        return self(x).mean()
