import torch
import torch.nn as nn
from torch.nn.init import constant_, trunc_normal_


class DownBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.norm = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        act: bool = True,
        norm: bool = True,
        inject_noise: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose3d(
            in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not norm
        )
        self.norm = nn.InstanceNorm3d(out_ch, affine=True) if norm else nn.Identity()
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
        self.inject_noise = inject_noise
        if inject_noise:
            self.noise_scale = nn.Parameter(torch.zeros(1, out_ch, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm(self.conv(x)))
        if self.inject_noise:
            h = h + self.noise_scale * torch.randn_like(h)
        return h


class UNet3DGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        enc_channels: list[int] = [64, 128, 256, 512],
        dec_channels: list[int] = [512, 256, 128, 64],
        noise_channels: int = 32,
        output: str = "tanh",
    ) -> None:
        super().__init__()
        assert output in ("tanh", "softmax"), "output must be 'tanh' or 'softmax'"
        assert len(enc_channels) == len(dec_channels), (
            "encoder/decoder depth must match"
        )
        self.in_channels = in_channels
        self.enc_channels = list(enc_channels)
        self.dec_channels = list(dec_channels)
        self.noise_channels = noise_channels
        self.output = output

        encs = []
        in_c = in_channels + 1
        for out_c in enc_channels:
            encs.append(DownBlock3D(in_c, out_c))
            in_c = out_c
        self.encoders = nn.ModuleList(encs)

        decs = []
        for i, out_c in enumerate(dec_channels):
            is_last = i == len(dec_channels) - 1
            if i == 0:
                in_c = enc_channels[-1] + noise_channels
            else:
                in_c = dec_channels[i - 1] + enc_channels[-(i + 1)]
            decs.append(
                UpBlock3D(
                    in_c,
                    out_c,
                    act=not is_last,
                    norm=not is_last,
                    inject_noise=not is_last,
                )
            )
        self.decoders = nn.ModuleList(decs)

        self.final = nn.Conv3d(dec_channels[-1], in_channels, kernel_size=1)

        self.apply(self._init_weights)

    @property
    def total_stride(self) -> int:
        return 2 ** len(self.enc_channels)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            trunc_normal_(m.weight.data, std=0.02)
            if m.bias is not None:
                constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.InstanceNorm3d) and m.affine:
            constant_(m.weight.data, 1.0)
            constant_(m.bias.data, 0.0)

    def _check_shape(self, D: int, H: int, W: int) -> None:
        s = self.total_stride
        for i, d in enumerate((D, H, W)):
            if d % s != 0:
                raise ValueError(
                    f"spatial dim {i} ({d}) not divisible by total stride {s}"
                )

    def forward(
        self,
        sparse: torch.Tensor,
        mask: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, C, D, H, W = sparse.shape
        assert mask.shape == (B, 1, D, H, W)
        self._check_shape(D, H, W)

        x = torch.cat([sparse, mask], dim=1)

        skips: list[torch.Tensor] = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        if noise is None:
            noise = torch.randn(
                B, self.noise_channels, *x.shape[2:], device=x.device, dtype=x.dtype
            )
        else:
            assert noise.shape == (B, self.noise_channels, *x.shape[2:])
        x = torch.cat([x, noise], dim=1)

        for i, dec in enumerate(self.decoders):
            x = dec(x)
            if i < len(self.decoders) - 1:
                skip = skips[-(i + 2)]
                x = torch.cat([x, skip], dim=1)

        x = self.final(x)
        if self.output == "tanh":
            return torch.tanh(x)
        return torch.softmax(x, dim=1)
