"""Adapter that maps the lifted 3D feature volume (B, 65, 256, 256, 32) -- 64
features + 1 confidence channel -- to the (B, 64, 128, 128, 16) tensor that
the vendored MonoScene UNet3D head expects.

Uses confidence-weighted 2x average pooling so high-confidence voxels
dominate the merged half-resolution feature, then a small refinement conv.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MonoSceneFeatureAdapter(nn.Module):
    def __init__(self, eps: float = 1e-6, refine: bool = True) -> None:
        super().__init__()
        self.eps = float(eps)

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
        x:      (B, 65, 256, 256, 32)  -- channels[:64]=features, [64:65]=confidence
        return: (B, 64, 128, 128, 16)
        """
        feat = x[:, :64]
        conf = x[:, 64:65].clamp_min(0.0)

        num = F.avg_pool3d(feat * conf, kernel_size=2, stride=2)
        den = F.avg_pool3d(conf, kernel_size=2, stride=2)

        out = num / (den + self.eps)
        out = self.refine(out)
        return out
