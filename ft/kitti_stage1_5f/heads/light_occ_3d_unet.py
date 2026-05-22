"""Lightweight 3D U-Net consuming (V_rec, W_rec) -> SSC logits."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class _ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            _gn(c_out),
            nn.SiLU(inplace=True),
            nn.Conv3d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            _gn(c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DownBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.down = nn.Conv3d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = _gn(c_out)
        self.act = nn.SiLU(inplace=True)
        self.conv = _ConvBlock(c_out, c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm(self.down(x)))
        return self.conv(x)


class _UpBlock(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(c_in, c_out, kernel_size=1)
        self.conv = _ConvBlock(c_out + c_skip, c_out)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class LightOcc3DUNet(nn.Module):
    """A small 3D U-Net for SemanticKITTI SSC (256x256x32 -> 20 classes).

    Input: (B, c_in, X, Y, Z). c_in is typically c_lift + 1 (V_rec || W_rec).
    """

    def __init__(
        self,
        c_in: int,
        num_classes: int = 20,
        base_channels: int = 64,
        channels: Tuple[int, int, int, int] = (64, 96, 128, 192),
    ) -> None:
        super().__init__()
        c0, c1, c2, c3 = channels
        if c0 != base_channels:
            raise ValueError(f"channels[0]={c0} must equal base_channels={base_channels}")

        self.stem = _ConvBlock(c_in, c0)
        self.down1 = _DownBlock(c0, c1)
        self.down2 = _DownBlock(c1, c2)
        self.down3 = _DownBlock(c2, c3)

        self.up3 = _UpBlock(c3, c2, c2)
        self.up2 = _UpBlock(c2, c1, c1)
        self.up1 = _UpBlock(c1, c0, c0)

        self.head = nn.Conv3d(c0, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)

        u2 = self.up3(s3, s2)
        u1 = self.up2(u2, s1)
        u0 = self.up1(u1, s0)
        return self.head(u0)
