"""Composable OccAny SemanticKITTI SSC model."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ft.semantickitti_ft.heads import build_head
from ft.semantickitti_ft.lifting import build_lifter
from occany.model.model_must3r import (
    Dust3rEncoder,
    Must3rDecoder,
    RaymapEncoderDiT,
)
from occany.model.must3r_blocks.head import ActivationType


class OccAnySSCModel(nn.Module):
    """Assemble a lift module and an SSC head behind the old forward API."""

    def __init__(
        self,
        img_encoder: Dust3rEncoder,
        decoder: Must3rDecoder,
        raymap_encoder: Optional[RaymapEncoderDiT],
        gen_decoder: Optional[Must3rDecoder],
        num_classes: int = 20,
        feature: int = 64,
        project_scale: int = 2,
        voxel_size: Tuple[float, float, float] = (0.2, 0.2, 0.2),
        point_cloud_range: Tuple[float, float, float, float, float, float] = (
            0.0, -25.6, -2.0, 51.2, 25.6, 4.4
        ),
        n_render_views: int = 4,
        n_decoder_feature_layers: int = 4,
        last_frame_view_indices: Tuple[int, int] = (4, 5),
        token_dim: int = 768,
        patch_size: int = 16,
        pointmaps_activation: ActivationType = ActivationType.LINEAR,
        backbone_dtype: torch.dtype = torch.bfloat16,
        lift_type: str = "occany_render_tokens",
        head_type: str = "monoscene",
    ) -> None:
        super().__init__()
        self.lifter = build_lifter(
            lift_type,
            img_encoder=img_encoder,
            decoder=decoder,
            raymap_encoder=raymap_encoder,
            gen_decoder=gen_decoder,
            n_render_views=n_render_views,
            n_decoder_feature_layers=n_decoder_feature_layers,
            last_frame_view_indices=last_frame_view_indices,
            pointmaps_activation=pointmaps_activation,
            backbone_dtype=backbone_dtype,
        )
        self.head = build_head(
            head_type,
            num_classes=num_classes,
            feature=feature,
            project_scale=project_scale,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            token_dim=token_dim,
            patch_size=patch_size,
        )

    def forward(
        self,
        views: List[Dict[str, torch.Tensor]],
        anchor_pose: torch.Tensor,
        lidar_to_world: torch.Tensor,
    ) -> torch.Tensor:
        features = self.lifter(
            views=views,
            anchor_pose=anchor_pose,
            lidar_to_world=lidar_to_world,
        )
        return self.head(features)


class OccAnyOccHead(OccAnySSCModel):
    """Backward-compatible name used by older training scripts."""

    pass
