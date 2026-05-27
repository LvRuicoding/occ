"""MonoScene-style occupancy head: adapter + vendored UNet3D(kitti, context_prior=True).

Input:  (B, 65+extra, 256, 256, 32)  -- lifted volume from Stage1LiftingModule,
        optionally followed by extra dense 3D fusion channels.
Output: dict with keys
   - "ssc_logit": (B, num_classes, 256, 256, 32)
   - "P_logits":  (B, n_relations, flatten_context_size, flatten_size)
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .monoscene import UNet3D
from .monoscene_adapter import MonoSceneFeatureAdapter


class MonoSceneOccHead(nn.Module):
    def __init__(
        self,
        num_classes: int = 20,
        feature: int = 64,
        project_scale: int = 2,
        full_scene_size: Tuple[int, int, int] = (256, 256, 32),
        context_prior: bool = True,
        bn_momentum: float = 0.1,
        adapter_refine: bool = True,
        adapter_extra_channels: int = 0,
    ) -> None:
        super().__init__()
        self.adapter = MonoSceneFeatureAdapter(
            refine=adapter_refine,
            extra_channels=adapter_extra_channels,
        )
        self.unet3d = UNet3D(
            class_num=num_classes,
            norm_layer=nn.BatchNorm3d,
            full_scene_size=full_scene_size,
            feature=feature,
            project_scale=project_scale,
            context_prior=context_prior,
            bn_momentum=bn_momentum,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x3d = self.adapter(x)
        return self.unet3d({"x3d": x3d})
