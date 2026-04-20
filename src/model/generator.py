import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_, constant_


class BlockConvTranspose3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        act: bool = False,
    ):
        super().__init__()

        self.conv3d = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Generator(nn.Module):
    def __init__(
        self,
        d_latent: list[int] = [32, 4, 4, 4],
        channels: list[int] = [1024, 512, 128, 32, 2],
        kernels: list[int] = [4, 4, 4, 4, 4],
        strides: list[int] = [2, 2, 2, 2, 2],
        paddings: list[int] = [2, 2, 2, 2, 3],
        otype: str = "clf",
    ):
        super().__init__()
        assert otype in ["clf", "reg"], "otype must be either 'clf' or 'reg'"

        self.otype = otype
        self.d_latent = d_latent

        in_channels = [d_latent[0]] + channels[:-1]
        out_channels = channels

        n_layers = len(kernels)
        act = [True] * (n_layers - 1) + [False]

        layers = []
        for in_channel, out_channel, kernel, stride, padding, act in zip(
            in_channels, out_channels, kernels, strides, paddings, act
        ):
            layer = BlockConvTranspose3d(
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
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.ConvTranspose3d):
            trunc_normal_(m.weight.data, std=0.02)
        if isinstance(m, nn.BatchNorm3d):
            trunc_normal_(m.weight.data, std=0.02)
            constant_(m.bias.data, val=0.0)

    def forward(self, x):
        x = self.layers(x)
        if self.otype == "clf":
            x = torch.softmax(x, dim=1)
        else:
            x = torch.tanh(x)

    def get_device(self):
        return next(self.layers.parameters()).device

    def gen_noise(self, batch_size: int):
        shape = [batch_size] + self.d_latent
        return torch.randn(shape, device=self.get_device())

    def generate(self, batch_size: int = 1):
        x = self.gen_noise(batch_size)
        return self(x)
