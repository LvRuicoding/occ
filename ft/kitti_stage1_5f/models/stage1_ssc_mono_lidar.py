"""Stage-1 SSC + MonoScene head + LiDAR fusion on the post-decoder t_rec.

Same as ``Stage1SSCMonoModel`` except that the (frozen) OccAny reconstruction
backbone's ``t_rec`` patch tokens are passed through a LiDAR-image fusion block
before being handed to the lifting module. The default fusion interaction is
windowed image+voxel self-attention; the original cross-attention path remains
available via ``fusion_attn_type="cross"``. The OccAny encoder and decoder stay
fully frozen and remain inside ``@torch.no_grad()``; only the fusion module,
lifting, and occ_head are trained.

Forward signature adds the LiDAR/calib inputs produced by
``Kitti5FrameStage1MonoLidarDataset`` + ``collate_stage1_mono_lidar``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .lifting import OccAnyRecon5FrameBackbone, Stage1LiftingModule
from .lidar_fusion import LidarImageFusionModule
from ..heads import MonoSceneOccHead


class Stage1SSCMonoLidarModel(nn.Module):
    """Frozen OccAny backbone + LiDAR fusion on t_rec + lifting + MonoScene head."""

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
        # LiDAR fusion controls
        fusion_vox_origin: Tuple[float, float, float] = (-25.6, -2.0, 0.0),
        fusion_vox_size: Tuple[float, float, float] = (0.4, 0.4, 0.4),
        fusion_vox_grid: Tuple[int, int, int] = (128, 16, 128),
        fusion_num_heads: int = 8,
        fusion_window: int = 4,
        fusion_d_voxel: int = 128,
        fusion_pe_num_freqs: int = 8,
        fusion_attn_type: str = "self",
    ) -> None:
        super().__init__()
        if c_lift != 64:
            raise ValueError(
                f"Stage1SSCMonoLidarModel requires c_lift=64; got {c_lift}."
            )

        self.backbone = OccAnyRecon5FrameBackbone(
            img_size=backbone_img_size,
            embed_dim=token_dim,
            patch_size=patch_size,
            backbone_dtype=backbone_dtype,
        )
        if occany_ckpt is not None:
            self.backbone.load_checkpoint(occany_ckpt)

        H_t = backbone_img_size[0] // patch_size
        W_t = backbone_img_size[1] // patch_size
        self.fusion = LidarImageFusionModule(
            d_model=token_dim,
            H_t=H_t,
            W_t=W_t,
            patch_size=patch_size,
            num_heads=fusion_num_heads,
            window=fusion_window,
            vox_origin=fusion_vox_origin,
            vox_size=fusion_vox_size,
            vox_grid=fusion_vox_grid,
            vfe_d_voxel=fusion_d_voxel,
            pe_num_freqs=fusion_pe_num_freqs,
            attn_type=fusion_attn_type,
        )

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
        T_target_from_refcam: torch.Tensor,           # (B, 4, 4)
        points_per_frame: List[List[torch.Tensor]],   # [B][N] (P, 4)
        T_cam_from_velo: torch.Tensor,                # (B, 4, 4)
        K_per_frame: torch.Tensor,                    # (B, N, 3, 3)
        image_hw: torch.Tensor,                       # (B, 2)
    ) -> Dict[str, torch.Tensor]:
        backbone_out = self.backbone(views)
        t_rec_fused = self.fusion(
            backbone_out["t_rec"],
            points_per_frame=points_per_frame,
            T_cam_from_velo=T_cam_from_velo,
            K_per_frame=K_per_frame,
            image_hw=image_hw,
        )
        V_rec, W_rec = self.lifting(
            t_rec=t_rec_fused,
            p_rec_global=backbone_out["p_rec_global"],
            c_rec=backbone_out["c_rec"],
            T_target_from_refcam=T_target_from_refcam,
        )
        feats = torch.cat([V_rec, W_rec], dim=1)  # (B, 65, 256, 256, 32)
        return self.occ_head(feats)


__all__ = ["Stage1SSCMonoLidarModel"]
