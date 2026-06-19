"""Incremental pointmap ablation models for KITTI Stage-1 experiments."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .lidar_fusion import LidarImageFusionModule
from .lifting import OccAnyRecon5FrameBackbone
from .stage1_ssc_bevdetocc_lidar_dense_depth import SingleScaleDPTDepthHead
from .stage1_ssc_bevdetocc_lidar_pointmap import (
    PostFusionPointmapHead,
    Stage1SSCBEVDetOccLidarPointmapModel,
)


class Stage1PointmapOriginalModel(nn.Module):
    """Original OccAny pointmap head without LiDAR/image fusion or SSC backend."""

    def __init__(
        self,
        occany_ckpt: Optional[str] = None,
        c_lift: int = 64,
        patch_size: int = 16,
        token_dim: int = 768,
        backbone_img_size: Tuple[int, int] = (512, 512),
        backbone_dtype: torch.dtype = torch.bfloat16,
        num_frames: int = 5,
        freeze_backbone: bool = False,
        **_unused,
    ) -> None:
        super().__init__()
        del c_lift
        self.num_frames = int(num_frames)
        self.freeze_backbone = bool(freeze_backbone)
        self.backbone = OccAnyRecon5FrameBackbone(
            img_size=backbone_img_size,
            embed_dim=token_dim,
            patch_size=patch_size,
            backbone_dtype=backbone_dtype,
            freeze=self.freeze_backbone,
        )
        if occany_ckpt is not None:
            self.backbone.load_checkpoint(occany_ckpt)

    def pretrained_parameter_prefixes(self) -> Tuple[str, ...]:
        return ("backbone.",)

    def set_freeze_backbone(self, freeze: bool = True) -> None:
        self.freeze_backbone = bool(freeze)
        self.backbone.set_frozen(self.freeze_backbone)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(
        self,
        views: List[Dict[str, torch.Tensor]],
        T_target_from_refcam: Optional[torch.Tensor] = None,
        points_per_frame: Optional[List[List[torch.Tensor]]] = None,
        T_cam_from_velo: Optional[torch.Tensor] = None,
        K_per_frame: Optional[torch.Tensor] = None,
        image_hw: Optional[torch.Tensor] = None,
        gt_depth: Optional[torch.Tensor] = None,
        return_depth: bool = False,
    ) -> Dict[str, torch.Tensor]:
        del T_target_from_refcam, points_per_frame, T_cam_from_velo
        del K_per_frame, image_hw, gt_depth, return_depth
        backbone_out = self.backbone(views)
        p_rec_local = backbone_out.get("p_rec_local")
        if p_rec_local is None:
            raise RuntimeError("Original OccAny pointmap output lacks 'p_rec_local'.")
        if backbone_out["p_rec_global"].shape[1] != self.num_frames:
            raise RuntimeError(
                f"model was built for num_frames={self.num_frames}, "
                f"got input N={backbone_out['p_rec_global'].shape[1]}."
            )
        return {
            "pointmap_pts3d": backbone_out["p_rec_global"],
            "pointmap_pts3d_local": p_rec_local,
            "pointmap_conf": backbone_out["c_rec"],
        }


class Stage1PointmapPostFusionOnlyModel(nn.Module):
    """Post-2D-fusion OccAny pointmap head without the BEVDet-OCC 3D branch."""

    def __init__(
        self,
        occany_ckpt: Optional[str] = None,
        c_lift: int = 64,
        patch_size: int = 16,
        token_dim: int = 768,
        backbone_img_size: Tuple[int, int] = (512, 512),
        backbone_dtype: torch.dtype = torch.bfloat16,
        num_frames: int = 5,
        freeze_backbone: bool = False,
        pointmap_out_channels: int = 7,
        fusion_vox_origin: Tuple[float, float, float] = (-25.6, -2.0, 0.0),
        fusion_vox_size: Tuple[float, float, float] = (0.4, 0.4, 0.4),
        fusion_vox_grid: Tuple[int, int, int] = (128, 16, 128),
        fusion_num_heads: int = 8,
        fusion_window: int = 4,
        fusion_d_voxel: int = 128,
        fusion_pe_num_freqs: int = 8,
        fusion_attn_type: str = "cross",
        **_unused,
    ) -> None:
        super().__init__()
        del c_lift
        if fusion_attn_type != "cross":
            raise ValueError(
                "Stage1PointmapPostFusionOnlyModel keeps the 2D fusion as "
                f"cross-attention; got fusion_attn_type={fusion_attn_type!r}."
            )
        self.num_frames = int(num_frames)
        self.freeze_backbone = bool(freeze_backbone)
        self.backbone = OccAnyRecon5FrameBackbone(
            img_size=backbone_img_size,
            embed_dim=token_dim,
            patch_size=patch_size,
            backbone_dtype=backbone_dtype,
            freeze=self.freeze_backbone,
        )
        if occany_ckpt is not None:
            self.backbone.load_checkpoint(occany_ckpt)

        H_t = backbone_img_size[0] // int(patch_size)
        W_t = backbone_img_size[1] // int(patch_size)
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
            attn_type="cross",
            fusion3d_enabled=False,
        )
        self.pointmap_head = PostFusionPointmapHead(
            token_dim=token_dim,
            patch_size=patch_size,
            out_channels=int(pointmap_out_channels),
            pointmaps_activation=self.backbone.decoder.pointmaps_activation,
            source_decoder=self.backbone.decoder,
        )

    def pretrained_parameter_prefixes(self) -> Tuple[str, ...]:
        return ("backbone.", "pointmap_head.")

    def set_freeze_backbone(self, freeze: bool = True) -> None:
        self.freeze_backbone = bool(freeze)
        self.backbone.set_frozen(self.freeze_backbone)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(
        self,
        views: List[Dict[str, torch.Tensor]],
        T_target_from_refcam: torch.Tensor,
        points_per_frame: List[List[torch.Tensor]],
        T_cam_from_velo: torch.Tensor,
        K_per_frame: torch.Tensor,
        image_hw: torch.Tensor,
        gt_depth: Optional[torch.Tensor] = None,
        return_depth: bool = False,
        grid_config: Optional[Dict[str, torch.Tensor | Tuple[int, int, int]]] = None,
    ) -> Dict[str, torch.Tensor]:
        del T_target_from_refcam, gt_depth, return_depth
        fusion_origin = None
        fusion_size = None
        fusion_grid = None
        if grid_config is not None:
            fusion_origin = grid_config.get("fusion_vox_origin")
            fusion_size = grid_config.get("fusion_vox_size")
            value = grid_config.get("fusion_vox_grid")
            if isinstance(value, torch.Tensor):
                if value.ndim == 2:
                    if value.shape[0] > 1 and not torch.equal(value, value[:1].expand_as(value)):
                        raise RuntimeError(
                            f"fusion_vox_grid must be identical within a batch; got {value.tolist()}"
                        )
                    value = value[0]
                fusion_grid = tuple(int(v) for v in value.detach().cpu().tolist())
            elif value is not None:
                fusion_grid = tuple(int(v) for v in value)
        backbone_out = self.backbone(views)
        t_rec_fused = self.fusion(
            backbone_out["t_rec"],
            points_per_frame=points_per_frame,
            T_cam_from_velo=T_cam_from_velo,
            K_per_frame=K_per_frame,
            image_hw=image_hw,
            p_rec_local=backbone_out.get("p_rec_local"),
            c_rec=backbone_out["c_rec"],
            fusion_vox_origin=fusion_origin,
            fusion_vox_size=fusion_size,
            fusion_vox_grid=fusion_grid,
        )
        if t_rec_fused.shape[1] != self.num_frames:
            raise RuntimeError(
                f"model was built for num_frames={self.num_frames}, "
                f"got input N={t_rec_fused.shape[1]}."
            )
        return self.pointmap_head(
            t_rec_fused,
            image_hw.to(device=t_rec_fused.device),
        )


class Stage1SSCBEVDetOccLidarPointmapDenseDepthModel(
    Stage1SSCBEVDetOccLidarPointmapModel
):
    """BEVDet-OCC model with both post-fusion pointmap and dense depth heads."""

    def __init__(
        self,
        *args,
        dense_depth_features: int = 128,
        dense_depth_initial: float = 10.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        token_dim = int(kwargs.get("token_dim", getattr(self.backbone, "embed_dim", 768)))
        patch_size = int(kwargs.get("patch_size", getattr(self.backbone, "patch_size", 16)))
        self.dense_depth_head = SingleScaleDPTDepthHead(
            token_dim=token_dim,
            patch_size=patch_size,
            features=int(dense_depth_features),
            initial_depth=float(dense_depth_initial),
        )

    def pretrained_parameter_prefixes(self) -> Tuple[str, ...]:
        return ("backbone.", "pointmap_head.")

    def forward(
        self,
        views: List[Dict[str, torch.Tensor]],
        T_target_from_refcam: torch.Tensor,
        points_per_frame: List[List[torch.Tensor]],
        T_cam_from_velo: torch.Tensor,
        K_per_frame: torch.Tensor,
        image_hw: torch.Tensor,
        gt_depth: Optional[torch.Tensor] = None,
        return_depth: bool = False,
        grid_config: Optional[Dict[str, torch.Tensor | Tuple[int, int, int]]] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = int(T_target_from_refcam.shape[0])

        def _grid_tuple(name: str, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
            if grid_config is None or name not in grid_config:
                return default
            value = grid_config[name]
            if isinstance(value, torch.Tensor):
                if value.ndim == 2:
                    if value.shape[0] > 1 and not torch.equal(value, value[:1].expand_as(value)):
                        raise RuntimeError(
                            f"{name} must be identical within a batch; got {value.tolist()}"
                        )
                    value = value[0]
                return tuple(int(v) for v in value.detach().cpu().tolist())
            return tuple(int(v) for v in value)

        def _grid_tensor(name: str, default: torch.Tensor, device: torch.device) -> torch.Tensor:
            if grid_config is None or name not in grid_config:
                return default.to(device=device, dtype=torch.float32).view(1, 3)
            value = grid_config[name]
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.float32)
            value = value.to(device=device, dtype=torch.float32)
            if value.ndim == 1:
                value = value.view(1, 3)
            if value.shape[0] == 1 and batch_size > 1:
                value = value.expand(batch_size, -1)
            return value

        half_grid = _grid_tuple("half_grid_size", self.half_grid_size)
        full_grid = _grid_tuple("grid_size", self.occ_head.full_grid)

        backbone_out = self.backbone(views)
        fusion_origin = None
        fusion_size = None
        fusion_grid = None
        if grid_config is not None:
            fusion_origin = grid_config.get("fusion_vox_origin")
            fusion_size = grid_config.get("fusion_vox_size")
            fusion_grid = _grid_tuple("fusion_vox_grid", self.fusion.vfe.vox_grid)
        t_rec_fused = self.fusion(
            backbone_out["t_rec"],
            points_per_frame=points_per_frame,
            T_cam_from_velo=T_cam_from_velo,
            K_per_frame=K_per_frame,
            image_hw=image_hw,
            p_rec_local=backbone_out.get("p_rec_local"),
            c_rec=backbone_out["c_rec"],
            fusion_vox_origin=fusion_origin,
            fusion_vox_size=fusion_size,
            fusion_vox_grid=fusion_grid,
        )
        B, N = t_rec_fused.shape[:2]
        if N != self.num_frames:
            raise RuntimeError(
                f"model was built for num_frames={self.num_frames}, got input N={N}."
            )

        pointmap_out = self.pointmap_head(
            t_rec_fused,
            image_hw.to(device=t_rec_fused.device),
        )
        dense_depth = self.dense_depth_head(
            t_rec_fused,
            image_hw.to(device=t_rec_fused.device),
        )

        feat_2d = self.token_projector(t_rec_fused)
        half_origin = _grid_tensor("half_voxel_origin", self.lss.voxel_origin, feat_2d.device)
        half_size = _grid_tensor("half_voxel_size", self.lss.voxel_size, feat_2d.device)
        lss_volume, depth_logits = self.lss(
            feat_2d=feat_2d,
            K_per_frame=K_per_frame.to(device=feat_2d.device),
            T_cam_from_velo=T_cam_from_velo.to(device=feat_2d.device),
            image_hw=image_hw.to(device=feat_2d.device),
            gt_depth=gt_depth,
            voxel_origin=half_origin,
            voxel_size=half_size,
            grid_size=half_grid,
        )
        memory = self.lidar_memory(
            points_per_frame,
            output_dtype=lss_volume.dtype,
            voxel_origin=half_origin,
            voxel_size=half_size,
            grid_size=half_grid,
        )
        memory = memory.to(device=lss_volume.device, dtype=lss_volume.dtype)
        enhanced = self.natten_fusion(lss_volume, memory)
        per_frame = torch.cat([enhanced, memory], dim=2)

        cam2world = self._stack_cam2world(views, device=per_frame.device)
        warped = self.warper(
            per_frame,
            T_target_from_refcam=T_target_from_refcam.to(device=per_frame.device),
            T_cam_from_velo=T_cam_from_velo.to(device=per_frame.device),
            cam2world_per_frame=cam2world,
            voxel_origin=half_origin.to(device=per_frame.device),
            voxel_size=half_size.to(device=per_frame.device),
            grid_size=half_grid,
        )
        B, N, C, X, Y, Z = warped.shape
        temporal = warped.view(B, N * C, X, Y, Z)
        temporal = self.temporal_reduce(temporal)
        logits = self.occ_head(temporal, full_grid=full_grid)

        out: Dict[str, torch.Tensor] = {
            "ssc_logit": logits,
            "dense_depth": dense_depth,
        }
        out.update(pointmap_out)
        if return_depth:
            if gt_depth is None:
                gt_depth = self.lss.build_depth_target(
                    points_per_frame=points_per_frame,
                    K_per_frame=K_per_frame,
                    T_cam_from_velo=T_cam_from_velo,
                    image_hw=image_hw,
                    H_t=depth_logits.shape[-2],
                    W_t=depth_logits.shape[-1],
                    device=depth_logits.device,
                )
            out.update(
                depth_logits=depth_logits,
                gt_depth=gt_depth.to(device=depth_logits.device, dtype=torch.float32),
                depth_start=self.lss.depth_start,
                depth_step=self.lss.depth_step,
            )
        return out


__all__ = [
    "Stage1PointmapOriginalModel",
    "Stage1PointmapPostFusionOnlyModel",
    "Stage1SSCBEVDetOccLidarPointmapDenseDepthModel",
]
