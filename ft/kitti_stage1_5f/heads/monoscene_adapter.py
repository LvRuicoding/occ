"""Adapter that maps lifted 3D volumes to the MonoScene UNet3D input.

Uses confidence-weighted 2x average pooling so high-confidence voxels
dominate the merged half-resolution reconstruction feature. Optional extra
channels, such as dense LiDAR VFE features plus a mask, are pooled normally and
fused after the reconstruction pooling step.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MonoSceneFeatureAdapter(nn.Module):
    def __init__(
        self,
        eps: float = 1e-6,
        refine: bool = True,
        extra_channels: int = 0,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.extra_channels = int(extra_channels)
        if self.extra_channels < 0:
            raise ValueError(f"extra_channels must be non-negative, got {extra_channels}")

        if self.extra_channels > 0:
            self.extra_fuse = nn.Sequential(
                nn.Conv3d(64 + self.extra_channels, 64, kernel_size=1, bias=False),
                nn.GroupNorm(8, 64),
                nn.GELU(),
            )
        else:
            self.extra_fuse = nn.Identity()

        if refine:
            self.refine = nn.Sequential(
                nn.Conv3d(64, 64, kernel_size=1, bias=False),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv3d(64, 64, kernel_size=1, bias=False),
            )
        else:
            self.refine = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:      (B, 65+extra_channels, 256, 256, 32)
                channels[:64]=features, [64:65]=confidence
        return: (B, 64, 128, 128, 16)
        """
        expected_c = 65 + self.extra_channels
        if x.shape[1] != expected_c:
            raise RuntimeError(
                f"MonoSceneFeatureAdapter expected {expected_c} channels, got {x.shape[1]}."
            )
        feat = x[:, :64]
        conf = x[:, 64:65].clamp_min(0.0)

        num = F.avg_pool3d(feat * conf, kernel_size=2, stride=2)
        den = F.avg_pool3d(conf, kernel_size=2, stride=2)

        out = num / (den + self.eps)
        if self.extra_channels > 0:
            extra = F.avg_pool3d(x[:, 65:], kernel_size=2, stride=2)
            out = self.extra_fuse(torch.cat([out, extra], dim=1))
        out = self.refine(out)
        return out
