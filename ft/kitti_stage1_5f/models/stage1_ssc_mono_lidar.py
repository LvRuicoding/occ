"""Stage-1 SSC + MonoScene head + LiDAR fusion on the post-decoder t_rec.

Same as ``Stage1SSCMonoModel`` except that the (frozen) OccAny reconstruction
backbone's ``t_rec`` patch tokens are passed through a LiDAR-image fusion block
before being handed to the lifting module. The default fusion interaction is
windowed image+voxel self-attention; the original cross-attention path remains
available via ``fusion_attn_type="cross"``. An optional second fusion stage
(``fusion3d_enabled=True``) applies fixed-length 3D-sorted self-attention over
image and voxel tokens using local pointmaps. A separate optional post-lift
branch (``post_lift_lidar_enabled=True``) aggregates all LiDAR sweeps into the
target voxel grid, fuses the dense LiDAR features with lifted reconstruction
features, and preserves the original adapter input shape. A separate optional
memory voxel branch (``memory_voxel_enabled=True``) builds a dense per-frame
voxel volume in each frame's velo coords, warps the historical frames to the
reference frame, max-pools them, and applies 3D NA cross-attention (natten) to
refine the post-lift output with this multi-frame memory. The OccAny encoder
and decoder stay fully frozen and remain inside ``@torch.no_grad()``; only
fusion, lifting, optional post-lift VFE, optional memory voxel fusion, and
occ_head are trained.

Forward signature adds the LiDAR/calib inputs produced by
``Kitti5FrameStage1MonoLidarDataset`` + ``collate_stage1_mono_lidar``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .lifting import OccAnyRecon5FrameBackbone, Stage1LiftingModule
from .lidar_fusion import (
    LidarImageFusionModule,
    MemoryVoxel3DFusion,
    TargetGridLidarFeatureEncoder,
)
from ..heads import MonoSceneOccHead


def _build_shared_point_mlp(hidden: int, d_voxel: int) -> nn.Sequential:
    """Per-point MLP shared between the cam-frame VFE and the target-grid VFE.

    Both VFEs use the same per-point feature layout (xyz, intensity, cell-offset
    = 7 dims) and the same d_voxel output width, so they can share the body of
    the encoder. Frame-specific signal in the target-grid path is injected by a
    separate ``time_embed`` *after* this MLP, so this module stays time-agnostic.
    """
    return nn.Sequential(
        nn.Linear(7, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Linear(hidden, d_voxel),
    )


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
        fusion3d_enabled: bool = False,
        fusion3d_seq_len: int = 80,
        fusion3d_num_heads: Optional[int] = None,
        fusion3d_ffn_ratio: float = 2.0,
        fusion3d_alpha_init: float = 0.0,
        post_lift_lidar_enabled: bool = False,
        post_lift_lidar_channels: int = 32,
        post_lift_lidar_d_voxel: int = 128,
        post_lift_lidar_hidden: int = 64,
        post_lift_lidar_pe_num_freqs: int = 8,
        # Memory voxel fusion (NA 3D cross-attn over warped per-frame voxel memory)
        memory_voxel_enabled: bool = False,
        memory_voxel_kernel: int = 7,
        memory_voxel_num_heads: int = 4,
        memory_voxel_num_layers: int = 2,
        memory_voxel_ffn_ratio: float = 2.0,
        memory_voxel_alpha_init: float = 0.0,
        memory_voxel_d_voxel: int = 128,
        memory_voxel_hidden: int = 64,
        memory_voxel_pe_num_freqs: int = 8,
        num_frames: int = 5,
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

        # When the post-lift target-grid VFE is enabled, share the per-point
        # MLP with the cam-frame VFE so both paths use a consistent
        # point->feature mapping. This requires matching d_voxel / hidden
        # between the two; we surface a clear error rather than silently
        # building two distinct MLPs. When memory voxel fusion is also enabled,
        # the same shared MLP is reused for its per-frame VFE, with an extra
        # constraint on memory_voxel_d_voxel.
        shared_point_mlp: Optional[nn.Module] = None
        if post_lift_lidar_enabled:
            if int(fusion_d_voxel) != int(post_lift_lidar_d_voxel):
                raise ValueError(
                    "post_lift_lidar requires fusion_d_voxel == "
                    f"post_lift_lidar_d_voxel; got {fusion_d_voxel} vs "
                    f"{post_lift_lidar_d_voxel}."
                )
            if memory_voxel_enabled and int(memory_voxel_d_voxel) != int(
                post_lift_lidar_d_voxel
            ):
                raise ValueError(
                    "When both post_lift_lidar and memory_voxel are enabled, "
                    "memory_voxel_d_voxel must equal post_lift_lidar_d_voxel; "
                    f"got {memory_voxel_d_voxel} vs {post_lift_lidar_d_voxel}."
                )
            shared_point_mlp = _build_shared_point_mlp(
                hidden=int(post_lift_lidar_hidden),
                d_voxel=int(post_lift_lidar_d_voxel),
            )

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
            vfe_hidden=int(post_lift_lidar_hidden) if shared_point_mlp is not None else 64,
            pe_num_freqs=fusion_pe_num_freqs,
            attn_type=fusion_attn_type,
            fusion3d_enabled=fusion3d_enabled,
            fusion3d_seq_len=fusion3d_seq_len,
            fusion3d_num_heads=fusion3d_num_heads,
            fusion3d_ffn_ratio=fusion3d_ffn_ratio,
            fusion3d_alpha_init=fusion3d_alpha_init,
            vfe_point_mlp=shared_point_mlp,
        )

        self.lifting = Stage1LiftingModule(
            token_dim=token_dim,
            c_lift=c_lift,
            patch_size=patch_size,
            voxel_origin=voxel_origin,
            voxel_size=voxel_size,
            grid_size=grid_size,
        )
        self.post_lift_lidar = (
            TargetGridLidarFeatureEncoder(
                vox_origin=voxel_origin,
                base_vox_size=tuple(float(v) * 2.0 for v in voxel_size),
                base_grid=tuple(max(1, int(v) // 2) for v in grid_size),
                full_grid=grid_size,
                d_voxel=post_lift_lidar_d_voxel,
                d_out=post_lift_lidar_channels,
                hidden=post_lift_lidar_hidden,
                pe_num_freqs=post_lift_lidar_pe_num_freqs,
                num_frames=num_frames,
                point_mlp=shared_point_mlp,
            )
            if post_lift_lidar_enabled
            else None
        )
        # +3 aux channels: occupancy mask, log1p point density, frame diversity.
        self.post_lift_fuse = (
            nn.Sequential(
                nn.Conv3d(
                    64 + int(post_lift_lidar_channels) + 3,
                    64,
                    kernel_size=1,
                    bias=False,
                ),
                nn.GroupNorm(8, 64),
                nn.GELU(),
            )
            if post_lift_lidar_enabled
            else None
        )

        # Memory voxel fusion: NA 3D cross-attn between warped per-frame voxel
        # memory and the post_lift_fuse output (or, when post_lift_lidar is
        # disabled, the raw lifted V_rec). Identity at init via alpha=0.
        self.memory_fusion = (
            MemoryVoxel3DFusion(
                c_in=c_lift,
                c_mem=c_lift,
                num_heads=int(memory_voxel_num_heads),
                kernel=int(memory_voxel_kernel),
                num_layers=int(memory_voxel_num_layers),
                ffn_ratio=float(memory_voxel_ffn_ratio),
                full_grid=tuple(int(s) for s in grid_size),
                base_grid=tuple(max(1, int(s) // 2) for s in grid_size),
                vox_origin=voxel_origin,
                vox_size=tuple(float(v) * 2.0 for v in voxel_size),
                d_voxel=int(memory_voxel_d_voxel),
                hidden=int(memory_voxel_hidden),
                pe_num_freqs=int(memory_voxel_pe_num_freqs),
                alpha_init=float(memory_voxel_alpha_init),
                point_mlp=shared_point_mlp if memory_voxel_enabled else None,
            )
            if memory_voxel_enabled
            else None
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
            p_rec_local=backbone_out.get("p_rec_local"),
            c_rec=backbone_out["c_rec"],
        )
        V_rec, W_rec = self.lifting(
            t_rec=t_rec_fused,
            p_rec_global=backbone_out["p_rec_global"],
            c_rec=backbone_out["c_rec"],
            T_target_from_refcam=T_target_from_refcam,
        )
        cam2world_per_frame: Optional[torch.Tensor] = None
        if self.post_lift_lidar is not None or self.memory_fusion is not None:
            cam2world_per_frame = self._stack_cam2world(views, device=V_rec.device)
        if self.post_lift_lidar is not None:
            V_lidar, M_lidar, C_lidar, D_lidar = self.post_lift_lidar(
                points_per_frame=points_per_frame,
                T_cam_from_velo=T_cam_from_velo.to(device=V_rec.device),
                T_target_from_refcam=T_target_from_refcam.to(device=V_rec.device),
                cam2world_per_frame=cam2world_per_frame,
                output_dtype=V_rec.dtype,
            )
            V_rec = self.post_lift_fuse(
                torch.cat([V_rec, V_lidar, M_lidar, C_lidar, D_lidar], dim=1)
            )
        if self.memory_fusion is not None:
            V_rec = self.memory_fusion(
                V_post_fuse=V_rec,
                points_per_frame=points_per_frame,
                T_cam_from_velo=T_cam_from_velo.to(device=V_rec.device),
                T_target_from_refcam=T_target_from_refcam.to(device=V_rec.device),
                cam2world_per_frame=cam2world_per_frame,
            )
        feats = torch.cat([V_rec, W_rec], dim=1)
        return self.occ_head(feats)

    @staticmethod
    def _stack_cam2world(
        views: List[Dict[str, torch.Tensor]],
        device: torch.device,
    ) -> torch.Tensor:
        if len(views) == 0:
            raise RuntimeError("views must contain at least one frame.")
        if "cam2world" not in views[0]:
            raise RuntimeError(
                "post-lift LiDAR fusion requires views[f]['cam2world'] for every frame."
            )
        return torch.stack(
            [v["cam2world"].to(device=device, dtype=torch.float32) for v in views],
            dim=1,
        )


__all__ = ["Stage1SSCMonoLidarModel"]
