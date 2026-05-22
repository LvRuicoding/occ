"""Stage 1 SSC model with the MonoScene KITTI head.

Identical to ``Stage1SSCModel`` except that the occupancy head is the
vendored MonoScene UNet3D (with context_prior=True), prepended by a small
adapter that maps (B, 65, 256, 256, 32) -> (B, 64, 128, 128, 16).
The forward returns a dict ``{"ssc_logit": ..., "P_logits": ...}``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .lifting import OccAnyRecon5FrameBackbone, Stage1LiftingModule
from ..heads import MonoSceneOccHead


class Stage1SSCMonoModel(nn.Module):
    """OccAny backbone (frozen) + lifting + MonoScene-style head."""

    def __init__(
        self,
        occany_ckpt: Optional[str] = None,
        c_lift: int = 64,
        num_classes: int = 20,
        patch_size: int = 16,
        token_dim: int = 768,
        backbone_img_size: Tuple[int, int] = (512, 512),
        backbone_dtype: torch.dtype = torch.bfloat16,
        voxel_origin: Tuple[float, float, float] = (0.0, -25.6, -2.0),
        voxel_size: Tuple[float, float, float] = (0.2, 0.2, 0.2),
        grid_size: Tuple[int, int, int] = (256, 256, 32),
        head_feature: int = 64,
        head_project_scale: int = 2,
        head_context_prior: bool = True,
        head_bn_momentum: float = 0.1,
        adapter_refine: bool = True,
    ) -> None:
        super().__init__()
        if c_lift != 64:
            raise ValueError(
                f"Stage1SSCMonoModel requires c_lift=64 (adapter expects 64-channel "
                f"features + 1 confidence); got c_lift={c_lift}."
            )

        self.backbone = OccAnyRecon5FrameBackbone(
            img_size=backbone_img_size,
            embed_dim=token_dim,
            patch_size=patch_size,
            backbone_dtype=backbone_dtype,
        )
        if occany_ckpt is not None:
            self.backbone.load_checkpoint(occany_ckpt)

        self.lifting = Stage1LiftingModule(
            token_dim=token_dim,
            c_lift=c_lift,
            patch_size=patch_size,
            voxel_origin=voxel_origin,
            voxel_size=voxel_size,
            grid_size=grid_size,
        )

        self.occ_head = MonoSceneOccHead(
            num_classes=num_classes,
            feature=head_feature,
            project_scale=head_project_scale,
            full_scene_size=tuple(int(s) for s in grid_size),
            context_prior=head_context_prior,
            bn_momentum=head_bn_momentum,
            adapter_refine=adapter_refine,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the frozen OccAny backbone in eval mode regardless of caller.
        self.backbone.eval()
        return self

    def forward(
        self,
        views: List[Dict[str, torch.Tensor]],
        T_target_from_refcam: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        backbone_out = self.backbone(views)
        V_rec, W_rec = self.lifting(
            t_rec=backbone_out["t_rec"],
            p_rec_global=backbone_out["p_rec_global"],
            c_rec=backbone_out["c_rec"],
            T_target_from_refcam=T_target_from_refcam,
        )
        feats = torch.cat([V_rec, W_rec], dim=1)  # (B, 65, 256, 256, 32)
        return self.occ_head(feats)
