"""BEVDet-OCC LiDAR model with post-2D-fusion OccAny pointmap supervision."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from occany.model.must3r_blocks.head import LinearHead, apply_activation

from .stage1_ssc_bevdetocc_lidar import Stage1SSCBEVDetOccLidarModel


class PostFusionPointmapHead(nn.Module):
    """OccAny LinearHead applied to LiDAR-fused reconstruction tokens."""

    def __init__(
        self,
        token_dim: int = 768,
        patch_size: int = 16,
        out_channels: int = 7,
        pointmaps_activation=None,
        source_decoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.out_channels = int(out_channels)
        self.pointmaps_activation = pointmaps_activation
        output_dim = self.patch_size * self.patch_size * self.out_channels
        self.head_dec = LinearHead(int(token_dim), output_dim, self.patch_size)

        if source_decoder is not None and hasattr(source_decoder, "head_dec"):
            self.head_dec.load_state_dict(source_decoder.head_dec.state_dict(), strict=True)

        if source_decoder is not None and hasattr(source_decoder, "pts3d_task_token"):
            task_token = source_decoder.pts3d_task_token.detach().clone().float()
            self.pts3d_task_token = nn.Parameter(task_token)
        else:
            self.register_parameter("pts3d_task_token", None)

    def forward(
        self,
        tokens: torch.Tensor,      # (B, N, H_t, W_t, D)
        image_hw: torch.Tensor,    # (B, 2)
    ) -> Dict[str, torch.Tensor]:
        B, N, H_t, W_t, D = tokens.shape
        if image_hw.ndim != 2 or image_hw.shape != (B, 2):
            raise RuntimeError(
                f"image_hw must be (B,2) matching tokens, got {tuple(image_hw.shape)} "
                f"for B={B}."
            )
        H_img = int(image_hw[0, 0].item())
        W_img = int(image_hw[0, 1].item())
        if not bool((image_hw[:, 0] == H_img).all().item()) or not bool(
            (image_hw[:, 1] == W_img).all().item()
        ):
            raise RuntimeError("PostFusionPointmapHead expects same image_hw within a batch.")
        if H_img // self.patch_size != H_t or W_img // self.patch_size != W_t:
            raise RuntimeError(
                f"token grid ({H_t},{W_t}) is incompatible with image_hw "
                f"({H_img},{W_img}) and patch_size={self.patch_size}."
            )

        dense = tokens.reshape(B * N, H_t * W_t, D).contiguous()
        if self.pts3d_task_token is not None:
            dense = dense + self.pts3d_task_token.to(device=dense.device, dtype=dense.dtype)

        raw = self.head_dec([dense], (H_img, W_img))
        raw = raw.view(B, N, H_img, W_img, self.out_channels).contiguous()

        pts3d = apply_activation(raw[..., :3], self.pointmaps_activation)
        pts3d_local = apply_activation(raw[..., 3:6], self.pointmaps_activation)
        conf = 1.0 + raw[..., 6].float().exp()
        return {
            "pointmap_raw": raw,
            "pointmap_pts3d": pts3d,
            "pointmap_pts3d_local": pts3d_local,
            "pointmap_conf": conf.to(dtype=raw.dtype),
        }


class Stage1SSCBEVDetOccLidarPointmapModel(Stage1SSCBEVDetOccLidarModel):
    """BEVDet-OCC LiDAR model plus a full-resolution pointmap auxiliary head."""

    def __init__(
        self,
        *args,
        pointmap_out_channels: int = 7,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        token_dim = int(kwargs.get("token_dim", getattr(self.backbone, "embed_dim", 768)))
        patch_size = int(kwargs.get("patch_size", getattr(self.backbone, "patch_size", 16)))
        self.pointmap_head = PostFusionPointmapHead(
            token_dim=token_dim,
            patch_size=patch_size,
            out_channels=int(pointmap_out_channels),
            pointmaps_activation=self.backbone.decoder.pointmaps_activation,
            source_decoder=self.backbone.decoder,
        )

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
        out: Dict[str, torch.Tensor] = {"ssc_logit": logits}
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


def _pointmap_targets_from_depth(
    dense_depth: torch.Tensor,          # (B, N, H, W)
    K_per_frame: torch.Tensor,          # (B, N, 3, 3)
    cam2world_per_frame: torch.Tensor,  # (B, N, 4, 4)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dense_depth.ndim != 4:
        raise RuntimeError(f"dense_depth must be (B,N,H,W), got {tuple(dense_depth.shape)}.")
    B, N, H, W = dense_depth.shape
    if K_per_frame.shape[:2] != (B, N) or K_per_frame.shape[-2:] != (3, 3):
        raise RuntimeError(
            f"K_per_frame shape {tuple(K_per_frame.shape)} does not match depth "
            f"shape {(B, N, H, W)}."
        )
    if cam2world_per_frame.shape[:2] != (B, N) or cam2world_per_frame.shape[-2:] != (4, 4):
        raise RuntimeError(
            f"cam2world_per_frame shape {tuple(cam2world_per_frame.shape)} does not "
            f"match depth shape {(B, N, H, W)}."
        )

    device = dense_depth.device
    z = dense_depth.float()
    K = K_per_frame.to(device=device, dtype=torch.float32)
    cam2world = cam2world_per_frame.to(device=device, dtype=torch.float32)

    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    v, u = torch.meshgrid(ys, xs, indexing="ij")
    u = u.view(1, 1, H, W)
    v = v.view(1, 1, H, W)

    fx = K[..., 0, 0].view(B, N, 1, 1).clamp(min=1e-6)
    fy = K[..., 1, 1].view(B, N, 1, 1).clamp(min=1e-6)
    cx = K[..., 0, 2].view(B, N, 1, 1)
    cy = K[..., 1, 2].view(B, N, 1, 1)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_local = torch.stack([x, y, z], dim=-1)
    valid = torch.isfinite(pts_local).all(dim=-1) & torch.isfinite(z) & (z > 0.0)

    T_ref_from_world = torch.linalg.inv(cam2world[:, 0])
    T_ref_from_cam = T_ref_from_world[:, None] @ cam2world
    R = T_ref_from_cam[:, :, :3, :3]
    t = T_ref_from_cam[:, :, :3, 3]
    pts_ref = torch.einsum("bnij,bnhwj->bnhwi", R, pts_local) + t[:, :, None, None, :]
    valid = valid & torch.isfinite(pts_ref).all(dim=-1)
    return pts_ref, pts_local, valid


def _gt_avg_dis_factor(
    gt_ref: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """OccAny PointmapLoss-style avg_dis factor from GT only."""
    gt_dis = gt_ref.float().norm(dim=-1)
    mask_f = valid.to(dtype=gt_dis.dtype)
    dims = tuple(range(1, gt_dis.ndim))
    valid_count = mask_f.sum(dim=dims, keepdim=True).clamp(min=1e-8)
    norm_factor = (gt_dis * mask_f).sum(dim=dims, keepdim=True) / valid_count
    return norm_factor.clamp(min=1e-8).unsqueeze(-1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values[mask].mean() if bool(mask.any().item()) else values.new_zeros(())


def pointmap_reconstruction_loss(
    pred_pts3d: torch.Tensor,
    pred_pts3d_local: torch.Tensor,
    pred_conf: torch.Tensor,
    dense_depth: torch.Tensor,
    K_per_frame: torch.Tensor,
    cam2world_per_frame: torch.Tensor,
    frame_mask: Optional[torch.Tensor] = None,
    loss_weight: float = 0.1,
    conf_alpha: float = 0.2,
    gt_scale: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """OccAny-style pointmap loss on valid dense-depth pixels.

    Training mirrors OccAny's confidence-aware pointmap objective:
    global and local L21 losses are computed separately, confidence-weighted
    separately, then summed. When gt_scale is False, both predictions and GT
    are divided by the same per-sample avg_dis factor computed from GT global
    pointmaps, matching the current PointmapLoss normalization semantics.
    """
    if pred_pts3d.shape != pred_pts3d_local.shape:
        raise RuntimeError(
            f"pred global/local pointmap shapes differ: {tuple(pred_pts3d.shape)} vs "
            f"{tuple(pred_pts3d_local.shape)}."
        )
    if pred_pts3d.ndim != 5 or pred_pts3d.shape[-1] != 3:
        raise RuntimeError(f"pred pointmaps must be (B,N,H,W,3), got {tuple(pred_pts3d.shape)}.")

    device = pred_pts3d.device
    with torch.amp.autocast(device_type=device.type, enabled=False):
        pred_ref = pred_pts3d.float()
        pred_local = pred_pts3d_local.float()
        conf = pred_conf.to(device=device, dtype=torch.float32)
        if conf.ndim == pred_ref.ndim and conf.shape[-1] == 1:
            conf = conf[..., 0]
        dense_depth = dense_depth.to(device=device, dtype=torch.float32)
        K_per_frame = K_per_frame.to(device=device, dtype=torch.float32)
        cam2world_per_frame = cam2world_per_frame.to(device=device, dtype=torch.float32)

        gt_ref, gt_local, valid = _pointmap_targets_from_depth(
            dense_depth,
            K_per_frame,
            cam2world_per_frame,
        )
        if pred_ref.shape != gt_ref.shape:
            raise RuntimeError(
                f"pred pointmap shape {tuple(pred_ref.shape)} does not match GT "
                f"{tuple(gt_ref.shape)}."
            )
        if frame_mask is not None:
            fm = frame_mask.to(device=device, dtype=torch.bool).view(
                dense_depth.shape[0], dense_depth.shape[1], 1, 1
            )
            valid = valid & fm

        valid_count = valid.sum()
        if not bool(valid_count.item()):
            zero = pred_ref.sum() * 0.0
            return zero, zero.detach(), {
                "pointmap": 0.0,
                "pointmap_weighted": 0.0,
                "pointmap_pts3d": 0.0,
                "pointmap_pts3d_local": 0.0,
                "pointmap_conf_loss_g": 0.0,
                "pointmap_conf_loss_l": 0.0,
                "pointmap_valid": 0.0,
            }

        norm_factor = None
        if not gt_scale:
            norm_factor = _gt_avg_dis_factor(gt_ref, valid)
            norm_factor = norm_factor.to(device=device, dtype=torch.float32)
            pred_ref = pred_ref / norm_factor
            pred_local = pred_local / norm_factor
            gt_ref = gt_ref / norm_factor
            gt_local = gt_local / norm_factor

        loss_ref = torch.norm(pred_ref - gt_ref, dim=-1)
        loss_local = torch.norm(pred_local - gt_local, dim=-1)
        if conf.shape != valid.shape:
            raise RuntimeError(
                f"pred_conf shape {tuple(conf.shape)} does not match valid mask {tuple(valid.shape)}."
            )

        loss_ref_mean = _masked_mean(loss_ref, valid)
        loss_local_mean = _masked_mean(loss_local, valid)
        use_confidence = (float(conf_alpha) > 0.0) and (not gt_scale)
        if use_confidence:
            conf_safe = conf.clamp(min=1e-6)
            conf_loss_ref = loss_ref * conf - float(conf_alpha) * torch.log(conf_safe)
            conf_loss_local = loss_local * conf - float(conf_alpha) * torch.log(conf_safe)
            loss_ref_out = _masked_mean(conf_loss_ref, valid)
            loss_local_out = _masked_mean(conf_loss_local, valid)
        else:
            loss_ref_out = loss_ref_mean
            loss_local_out = loss_local_mean

        loss_raw = loss_ref_out + loss_local_out

        weighted = float(loss_weight) * loss_raw
        details = {
            "pointmap": float(loss_raw.detach()),
            "pointmap_weighted": float(weighted.detach()),
            "pointmap_pts3d": float(loss_ref_mean.detach()),
            "pointmap_pts3d_local": float(loss_local_mean.detach()),
            "pointmap_conf_loss_g": float(loss_ref_out.detach()),
            "pointmap_conf_loss_l": float(loss_local_out.detach()),
            "pointmap_valid": float(valid_count.detach()),
        }
        if norm_factor is not None:
            details["pointmap_norm_factor"] = float(norm_factor.detach().mean())
    return weighted, loss_raw.detach(), details


__all__ = [
    "PostFusionPointmapHead",
    "Stage1SSCBEVDetOccLidarPointmapModel",
    "pointmap_reconstruction_loss",
]
