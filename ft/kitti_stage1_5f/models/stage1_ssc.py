"""Stage 1 SSC model: frozen OccAny recon + trainable lifting + SSC head."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .lifting import OccAnyRecon5FrameBackbone, Stage1LiftingModule
from ..heads import LightOcc3DUNet


class Stage1SSCModel(nn.Module):
    """End-to-end model composing the three Stage-1 pieces."""

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
    ) -> None:
        super().__init__()
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

        self.occ_head = LightOcc3DUNet(
            c_in=c_lift + 1,
            num_classes=num_classes,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Always keep the frozen backbone in eval mode.
        self.backbone.eval()
        return self

    def forward(
        self,
        views: List[Dict[str, torch.Tensor]],
        T_target_from_refcam: torch.Tensor,
    ) -> torch.Tensor:
        backbone_out = self.backbone(views)
        V_rec, W_rec = self.lifting(
            t_rec=backbone_out["t_rec"],
            p_rec_global=backbone_out["p_rec_global"],
            c_rec=backbone_out["c_rec"],
            T_target_from_refcam=T_target_from_refcam,
        )
        feats = torch.cat([V_rec, W_rec], dim=1)
        return self.occ_head(feats)
