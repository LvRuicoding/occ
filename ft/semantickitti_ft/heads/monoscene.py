"""MonoScene occupancy head adapter."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from ft.semantickitti_ft.interfaces import LiftedFeatures
from .registry import register_head


VGGT_HEAD_PATH = Path("/home/dataset-local/lr/code/openmm_vggt")
if str(VGGT_HEAD_PATH) not in sys.path:
    sys.path.insert(0, str(VGGT_HEAD_PATH))
from openmm_vggt.heads.monoscene_occupancy_head import MonoSceneOccupancyHead  # noqa: E402


@register_head("monoscene")
class MonoSceneSSCHead(nn.Module):
    """Adapter that maps lifted features to SSC logits."""

    def __init__(
        self,
        num_classes: int = 20,
        feature: int = 64,
        project_scale: int = 2,
        voxel_size: Tuple[float, float, float] = (0.2, 0.2, 0.2),
        point_cloud_range: Tuple[float, float, float, float, float, float] = (
            0.0, -25.6, -2.0, 51.2, 25.6, 4.4
        ),
        token_dim: int = 768,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.head = MonoSceneOccupancyHead(
            token_dim=token_dim,
            patch_size=patch_size,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            num_classes=num_classes,
            feature=feature,
            project_scale=project_scale,
            context_prior=False,
            n_relations=4,
        )

    def forward(self, features: LiftedFeatures) -> torch.Tensor:
        outputs = self.head(
            aggregated_tokens_list=features.aggregated_tokens_list,
            images=features.images,
            intrinsics=features.intrinsics,
            camera_to_world=features.camera_to_world.float(),
            lidar_to_world=features.lidar_to_world.float(),
        )
        return outputs["ssc_logit"]

