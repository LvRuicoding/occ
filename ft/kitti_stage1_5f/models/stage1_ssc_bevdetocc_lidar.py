"""Stage-1 SSC with OccAny token fusion and a BEVDet-OCC style 3D head.

This variant keeps the earliest LiDAR/image 2D cross-attention on OccAny
decoder tokens, then replaces the old lifting + MonoScene head with:

  token 1x1 projection -> LSS depth lifting on a half KITTI grid ->
  per-frame LiDAR voxel memory -> 3D NATTEN cross-attention ->
  temporal warp/concat -> BEVDet CustomResNet3D + LSSFPN3D ->
  full-grid upsample -> final_conv + predicter.

The returned layout always contains ``{"ssc_logit": (B, 20, X, Y, Z)}``.
Training can also request BEVDet-style sparse LiDAR depth supervision targets
for the LSS depth distribution branch.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bevdet3d_local import BEVDetOcc3DHead, conv_bn_relu_3d
from .lifting import OccAnyRecon5FrameBackbone
from .lidar_fusion import (
    LidarImageFusionModule,
    NA3DCrossAttnBlock,
    PerFrameMemoryVoxelEncoder,
)


class OccAnyTokenProjector(nn.Module):
    """1x1 Conv2d projection from OccAny tokens to LSS input channels."""

    def __init__(self, token_dim: int = 768, out_channels: int = 256) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(int(token_dim), int(out_channels), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(out_channels)),
            nn.GELU(),
        )

    def forward(self, t_rec: torch.Tensor) -> torch.Tensor:
        B, N, H, W, D = t_rec.shape
        x = t_rec.reshape(B * N, H, W, D).permute(0, 3, 1, 2).contiguous()
        x = self.proj(x)
        C = x.shape[1]
        return x.view(B, N, C, H, W).contiguous()


class GeometryResidualConvUnit(nn.Module):
    """DPT-style residual conv unit for 2D geometry refinement."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        channels = int(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.gelu(x + y)


class GeometryFeatureFusionBlock(nn.Module):
    """Top-down DPT-style feature fusion with optional upsampling."""

    def __init__(
        self,
        channels: int,
        has_residual: bool = True,
        upsample: bool = True,
    ) -> None:
        super().__init__()
        self.has_residual = bool(has_residual)
        self.upsample = bool(upsample)
        self.residual_unit = (
            GeometryResidualConvUnit(channels) if self.has_residual else None
        )
        self.output_unit = GeometryResidualConvUnit(channels)
        self.out_conv = nn.Conv2d(int(channels), int(channels), kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if self.has_residual and residual is not None and self.residual_unit is not None:
            x = x + self.residual_unit(residual)
        x = self.output_unit(x)
        if self.upsample:
            if size is None:
                x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            else:
                x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.out_conv(x)


class SharedGeometryAdapter(nn.Module):
    """Shared 2D geometry adapter used by both dense depth and occupancy."""

    def __init__(
        self,
        in_channels: int = 256,
        geometry_channels: int = 256,
        out_channels: int = 256,
        residual_gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        in_channels = int(in_channels)
        geometry_channels = int(geometry_channels)
        out_channels = int(out_channels)

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, geometry_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(geometry_channels),
            nn.GELU(),
        )
        self.level1 = GeometryResidualConvUnit(geometry_channels)
        self.level2 = nn.Sequential(
            nn.Conv2d(
                geometry_channels,
                geometry_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(geometry_channels),
            nn.GELU(),
            GeometryResidualConvUnit(geometry_channels),
        )
        self.level3 = nn.Sequential(
            nn.Conv2d(
                geometry_channels,
                geometry_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(geometry_channels),
            nn.GELU(),
            GeometryResidualConvUnit(geometry_channels),
        )
        self.level4 = nn.Sequential(
            nn.Conv2d(
                geometry_channels,
                geometry_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(geometry_channels),
            nn.GELU(),
            GeometryResidualConvUnit(geometry_channels),
        )

        self.refine4 = GeometryFeatureFusionBlock(
            geometry_channels, has_residual=False, upsample=True
        )
        self.refine3 = GeometryFeatureFusionBlock(
            geometry_channels, has_residual=True, upsample=True
        )
        self.refine2 = GeometryFeatureFusionBlock(
            geometry_channels, has_residual=True, upsample=True
        )
        self.refine1 = GeometryFeatureFusionBlock(
            geometry_channels, has_residual=True, upsample=False
        )
        self.shared_out = GeometryResidualConvUnit(geometry_channels)
        self.occ_guidance = nn.Sequential(
            GeometryResidualConvUnit(geometry_channels),
            nn.Conv2d(geometry_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.use_input_residual = in_channels == out_channels
        self.residual_gate = nn.Parameter(
            torch.tensor(float(residual_gate_init), dtype=torch.float32)
        )

    def forward(self, feat_2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if feat_2d.ndim != 5:
            raise RuntimeError(
                f"feat_2d must be (B, N, C, H, W), got {tuple(feat_2d.shape)}."
            )
        B, N, C, H, W = feat_2d.shape
        x = feat_2d.reshape(B * N, C, H, W).contiguous()

        l1 = self.level1(self.input_proj(x))
        l2 = self.level2(l1)
        l3 = self.level3(l2)
        l4 = self.level4(l3)

        y = self.refine4(l4, size=l3.shape[-2:])
        y = self.refine3(y, l3, size=l2.shape[-2:])
        y = self.refine2(y, l2, size=l1.shape[-2:])
        shared = self.shared_out(self.refine1(y, l1))

        delta = self.occ_guidance(shared)
        if self.use_input_residual:
            gate = self.residual_gate.to(device=delta.device, dtype=delta.dtype)
            refined = x + gate * delta
        else:
            refined = delta

        C_refined = refined.shape[1]
        C_shared = shared.shape[1]
        refined = refined.view(B, N, C_refined, H, W).contiguous()
        shared = shared.view(B, N, C_shared, H, W).contiguous()
        return refined, shared


class DenseDepthHead(nn.Module):
    """Continuous dense depth head on top of shared geometry features."""

    def __init__(self, in_channels: int = 256, hidden_channels: int = 128) -> None:
        super().__init__()
        in_channels = int(in_channels)
        hidden_channels = int(hidden_channels)
        mid_channels = max(hidden_channels // 4, 32)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )

    @staticmethod
    def _target_size(image_hw: torch.Tensor, fallback: Tuple[int, int]) -> Tuple[int, int]:
        if image_hw is None:
            return fallback
        if image_hw.ndim == 3:
            hw = image_hw[0, 0]
        elif image_hw.ndim == 2:
            hw = image_hw[0]
        else:
            hw = image_hw
        return int(hw[0].item()), int(hw[1].item())

    def forward(self, shared_geometry_feat: torch.Tensor, image_hw: torch.Tensor) -> torch.Tensor:
        if shared_geometry_feat.ndim == 5:
            B, N, C, H, W = shared_geometry_feat.shape
            x = shared_geometry_feat.reshape(B * N, C, H, W).contiguous()
        elif shared_geometry_feat.ndim == 4:
            B, C, H, W = shared_geometry_feat.shape
            N = None
            x = shared_geometry_feat.contiguous()
        else:
            raise RuntimeError(
                "shared_geometry_feat must be (B, C, H, W) or (B, N, C, H, W), got "
                f"{tuple(shared_geometry_feat.shape)}."
            )
        depth_logits = self.head(x)
        target_h, target_w = self._target_size(image_hw, fallback=(H, W))
        if depth_logits.shape[-2:] != (target_h, target_w):
            depth_logits = F.interpolate(
                depth_logits,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
        pred_depth = F.softplus(depth_logits) + 1e-3
        if N is None:
            return pred_depth.view(B, 1, target_h, target_w).contiguous()
        return pred_depth.view(B, N, 1, target_h, target_w).contiguous()


class LSSDepthLift(nn.Module):
    """Depth-distribution LSS lifting into each frame's own velo half-grid."""

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 32,
        depth_bound: Tuple[float, float, float] = (1.0, 52.0, 0.4),
        voxel_origin: Tuple[float, float, float] = (0.0, -25.6, -2.0),
        voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.4),
        grid_size: Tuple[int, int, int] = (128, 128, 16),
    ) -> None:
        super().__init__()
        self.depth_start = float(depth_bound[0])
        self.depth_end = float(depth_bound[1])
        self.depth_step = float(depth_bound[2])
        depth_values = torch.arange(
            float(depth_bound[0]), float(depth_bound[1]), float(depth_bound[2])
        )
        if depth_values.numel() <= 0:
            raise ValueError(f"invalid depth_bound={depth_bound}")
        self.depth_channels = int(depth_values.numel())
        self.out_channels = int(out_channels)
        self.grid_size = tuple(int(v) for v in grid_size)
        self.register_buffer("depth_values", depth_values.float(), persistent=False)
        self.register_buffer(
            "voxel_origin", torch.tensor(voxel_origin, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "voxel_size", torch.tensor(voxel_size, dtype=torch.float32), persistent=False
        )
        self.depth_net = nn.Sequential(
            nn.Conv2d(int(in_channels), int(in_channels), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(in_channels)),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                int(in_channels),
                self.depth_channels + self.out_channels,
                kernel_size=1,
                bias=True,
            ),
        )

    @torch.no_grad()
    def build_depth_target(
        self,
        points_per_frame: List[List[torch.Tensor]],
        K_per_frame: torch.Tensor,
        T_cam_from_velo: torch.Tensor,
        image_hw: torch.Tensor,
        H_t: int,
        W_t: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Project LiDAR points to the LSS feature grid and keep nearest depth.

        This follows BEVDet's ``PointToMultiViewDepth.points2depthmap`` logic:
        projected pixels are rounded to the downsampled grid and, when multiple
        points hit the same cell, the nearest valid depth is kept.
        """
        B = len(points_per_frame)
        if B == 0:
            raise RuntimeError("points_per_frame must contain at least one sample.")
        N = len(points_per_frame[0])
        depth_maps = torch.zeros((B, N, H_t, W_t), device=device, dtype=torch.float32)

        K = K_per_frame.to(device=device, dtype=torch.float32)
        T = T_cam_from_velo.to(device=device, dtype=torch.float32)
        image_hw = image_hw.to(device=device, dtype=torch.float32)

        with torch.amp.autocast(device_type=device.type, enabled=False):
            for b in range(B):
                img_h = image_hw[b, 0].clamp(min=1.0)
                img_w = image_hw[b, 1].clamp(min=1.0)
                scale_x = img_w / float(W_t)
                scale_y = img_h / float(H_t)
                R = T[b, :3, :3]
                t = T[b, :3, 3]
                for f in range(N):
                    pts = points_per_frame[b][f]
                    if pts.numel() == 0:
                        continue
                    pts_xyz = pts.to(device=device, dtype=torch.float32)[:, :3]
                    pts_cam = pts_xyz @ R.T + t
                    depth = pts_cam[:, 2]
                    valid_z = torch.isfinite(pts_cam).all(dim=1) & (depth > 1e-6)
                    if not bool(valid_z.any().item()):
                        continue

                    pts_cam = pts_cam[valid_z]
                    depth = depth[valid_z]
                    K_bf = K[b, f]
                    u = pts_cam[:, 0] / depth * K_bf[0, 0] + K_bf[0, 2]
                    v = pts_cam[:, 1] / depth * K_bf[1, 1] + K_bf[1, 2]
                    coor_x = torch.round(u / scale_x)
                    coor_y = torch.round(v / scale_y)
                    valid = (
                        torch.isfinite(coor_x)
                        & torch.isfinite(coor_y)
                        & (coor_x >= 0)
                        & (coor_x < W_t)
                        & (coor_y >= 0)
                        & (coor_y < H_t)
                        & (depth >= self.depth_start)
                        & (depth < self.depth_end)
                    )
                    if not bool(valid.any().item()):
                        continue

                    coor_x = coor_x[valid].long()
                    coor_y = coor_y[valid].long()
                    depth = depth[valid]
                    ranks = coor_x + coor_y * W_t
                    order = (ranks.to(torch.float32) + depth / 100.0).argsort()
                    ranks = ranks[order]
                    coor_x = coor_x[order]
                    coor_y = coor_y[order]
                    depth = depth[order]

                    keep = torch.ones_like(ranks, dtype=torch.bool)
                    keep[1:] = ranks[1:] != ranks[:-1]
                    depth_maps[b, f, coor_y[keep], coor_x[keep]] = depth[keep]

        return depth_maps

    @staticmethod
    def _pixel_centers(
        H_t: int,
        W_t: int,
        image_hw: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_h = float(image_hw[0].item())
        img_w = float(image_hw[1].item())
        ys = (torch.arange(H_t, device=device, dtype=torch.float32) + 0.5) * (
            img_h / float(H_t)
        )
        xs = (torch.arange(W_t, device=device, dtype=torch.float32) + 0.5) * (
            img_w / float(W_t)
        )
        v, u = torch.meshgrid(ys, xs, indexing="ij")
        return u, v

    def _frustum_voxel_indices(
        self,
        K: torch.Tensor,
        T_velo_from_cam: torch.Tensor,
        H_t: int,
        W_t: int,
        image_hw: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        D = self.depth_channels
        Gx, Gy, Gz = self.grid_size

        u, v = self._pixel_centers(H_t, W_t, image_hw, device=device)
        depths = self.depth_values.to(device=device, dtype=torch.float32).view(D, 1, 1)
        K = K.to(device=device, dtype=torch.float32)
        fx = K[0, 0].clamp(min=1e-6)
        fy = K[1, 1].clamp(min=1e-6)
        cx = K[0, 2]
        cy = K[1, 2]

        x_cam = (u.unsqueeze(0) - cx) / fx * depths
        y_cam = (v.unsqueeze(0) - cy) / fy * depths
        z_cam = depths.expand_as(x_cam)
        p_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (D, H, W, 3)

        T = T_velo_from_cam.to(device=device, dtype=torch.float32)
        R = T[:3, :3]
        t = T[:3, 3]
        p_velo = p_cam.reshape(-1, 3) @ R.T + t
        p_velo = p_velo.view(D, H_t, W_t, 3)

        origin = self.voxel_origin.to(device=device, dtype=torch.float32)
        size = self.voxel_size.to(device=device, dtype=torch.float32)
        idx = torch.floor((p_velo - origin) / size).long()
        valid = (
            torch.isfinite(p_velo).all(dim=-1)
            & (idx[..., 0] >= 0)
            & (idx[..., 0] < Gx)
            & (idx[..., 1] >= 0)
            & (idx[..., 1] < Gy)
            & (idx[..., 2] >= 0)
            & (idx[..., 2] < Gz)
        )
        lin = ((idx[..., 0].clamp(0, Gx - 1) * Gy + idx[..., 1].clamp(0, Gy - 1)) * Gz)
        lin = lin + idx[..., 2].clamp(0, Gz - 1)

        hw = torch.arange(H_t * W_t, device=device, dtype=torch.long).view(1, H_t, W_t)
        hw = hw.expand(D, H_t, W_t)
        return lin, valid, hw

    def forward(
        self,
        feat_2d: torch.Tensor,       # (B, N, C, H_t, W_t)
        K_per_frame: torch.Tensor,   # (B, N, 3, 3)
        T_cam_from_velo: torch.Tensor,  # (B, 4, 4)
        image_hw: torch.Tensor,      # (B, 2)
        gt_depth: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del gt_depth  # Depth supervision is computed outside the lift operator.

        B, N, C, H_t, W_t = feat_2d.shape
        Gx, Gy, Gz = self.grid_size
        device = feat_2d.device

        x = feat_2d.reshape(B * N, C, H_t, W_t)
        depth_out = self.depth_net(x)
        depth_logits = depth_out[:, : self.depth_channels]
        img_feat = depth_out[:, self.depth_channels :]
        depth_prob = torch.softmax(depth_logits.float(), dim=1).to(dtype=img_feat.dtype)

        volumes = img_feat.new_zeros((B, N, self.out_channels, Gx, Gy, Gz))
        T_velo_from_cam = torch.linalg.inv(T_cam_from_velo.to(device=device, dtype=torch.float32))

        for b in range(B):
            for f in range(N):
                lin, valid, hw_idx = self._frustum_voxel_indices(
                    K_per_frame[b, f],
                    T_velo_from_cam[b],
                    H_t,
                    W_t,
                    image_hw[b],
                    device,
                )
                valid_flat = valid.reshape(-1)
                if not bool(valid_flat.any().item()):
                    continue

                frame_id = b * N + f
                feat_hw = (
                    img_feat[frame_id]
                    .permute(1, 2, 0)
                    .reshape(H_t * W_t, self.out_channels)
                )
                weights = depth_prob[frame_id].reshape(-1)[valid_flat]
                src_feat = feat_hw[hw_idx.reshape(-1)[valid_flat]]
                contrib = src_feat * weights.unsqueeze(-1).to(dtype=src_feat.dtype)

                vol_flat = img_feat.new_zeros((Gx * Gy * Gz, self.out_channels))
                vol_flat.index_add_(0, lin.reshape(-1)[valid_flat], contrib)
                volumes[b, f] = (
                    vol_flat.view(Gx, Gy, Gz, self.out_channels)
                    .permute(3, 0, 1, 2)
                    .contiguous()
                )

        depth_logits = depth_logits.view(B, N, self.depth_channels, H_t, W_t)
        return volumes, depth_logits


def bevdet_depth_loss(
    depth_logits: torch.Tensor,
    gt_depth: torch.Tensor,
    depth_start: float = 1.0,
    depth_step: float = 0.4,
    loss_weight: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BEVDet-style one-hot depth BCE over valid LiDAR-projected cells.

    ``gt_depth`` is a sparse raw-depth map with zeros for empty cells. The
    discretization mirrors BEVDet's ``get_downsampled_gt_depth`` for linear
    depth bins: class 0 is the ignored/no-depth bin and is removed before BCE.
    """
    if depth_logits.ndim != 5:
        raise RuntimeError(
            f"depth_logits must be (B, N, D, H, W), got {tuple(depth_logits.shape)}."
        )
    expected_shape = (
        depth_logits.shape[0],
        depth_logits.shape[1],
        depth_logits.shape[3],
        depth_logits.shape[4],
    )
    if gt_depth.shape != expected_shape:
        raise RuntimeError(
            f"gt_depth shape {tuple(gt_depth.shape)} does not match depth_logits "
            f"{tuple(depth_logits.shape)}."
        )

    device_type = depth_logits.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):
        logits = depth_logits.float()
        gt = gt_depth.to(device=logits.device, dtype=torch.float32)
        D = logits.shape[2]
        depth_labels = (gt - (float(depth_start) - float(depth_step))) / float(depth_step)
        valid = (gt > 0.0) & (depth_labels >= 1.0) & (depth_labels < float(D + 1))
        valid_count = valid.sum()
        if not bool(valid_count.item()):
            zero = logits.sum() * 0.0
            return zero, zero.detach(), valid_count.to(dtype=torch.float32)

        label_ids = depth_labels.long().clamp(min=0, max=D)
        labels = F.one_hot(label_ids, num_classes=D + 1)[..., 1:].float()
        preds = logits.softmax(dim=2).permute(0, 1, 3, 4, 2).contiguous()
        labels = labels[valid]
        preds = preds[valid].clamp(min=1e-6, max=1.0 - 1e-6)
        raw_loss = F.binary_cross_entropy(preds, labels, reduction="none").sum()
        raw_loss = raw_loss / valid_count.clamp(min=1).to(dtype=torch.float32)
        weighted_loss = float(loss_weight) * raw_loss
    return weighted_loss, raw_loss.detach(), valid_count.to(dtype=torch.float32)


def dense_lss_depth_loss(
    depth_logits: torch.Tensor,
    dense_depth_gt: torch.Tensor,
    has_dense_depth: Optional[torch.Tensor] = None,
    depth_start: float = 1.0,
    depth_step: float = 0.4,
    min_depth: float = 1.0,
    max_depth: float = 80.0,
    loss_weight: float = 0.05,
    target_index: int = 0,
    min_valid_ratio: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masked target-frame dense-depth CE over the LSS depth distribution."""
    if depth_logits.ndim != 5:
        raise RuntimeError(
            f"depth_logits must be (B, N, D, H, W), got {tuple(depth_logits.shape)}."
        )
    B, N, D, H_t, W_t = depth_logits.shape
    target_index = int(target_index)
    if target_index < 0 or target_index >= N:
        raise RuntimeError(f"target_index={target_index} is out of range for N={N}.")

    gt = dense_depth_gt
    if gt.ndim == 5:
        gt = gt[:, target_index, 0]
    elif gt.ndim == 4:
        if gt.shape[1] == 1:
            gt = gt[:, 0]
        else:
            gt = gt[:, target_index]
    elif gt.ndim != 3:
        raise RuntimeError(
            "dense_depth_gt must be (B,H,W), (B,N,H,W), or (B,N,1,H,W), got "
            f"{tuple(gt.shape)}."
        )
    if gt.shape[0] != B:
        raise RuntimeError(
            f"dense_depth_gt batch {gt.shape[0]} does not match depth_logits batch {B}."
        )

    device_type = depth_logits.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):
        logits = depth_logits[:, target_index].float()
        gt_f = gt.to(device=logits.device, dtype=torch.float32)
        valid_src = (
            torch.isfinite(gt_f)
            & (gt_f >= float(min_depth))
            & (gt_f <= float(max_depth))
        )
        if has_dense_depth is not None:
            has = has_dense_depth.to(device=logits.device, dtype=torch.bool)
            if has.ndim == 2:
                has = has[:, target_index]
            elif has.ndim != 1:
                raise RuntimeError(
                    "has_dense_depth must be (B,) or (B,N), got "
                    f"{tuple(has_dense_depth.shape)}."
                )
            valid_src = valid_src & has.view(B, 1, 1)

        if gt_f.shape[-2:] == (H_t, W_t):
            gt_down = gt_f
            valid = valid_src
        else:
            valid_f = valid_src.float()
            gt_clean = torch.where(valid_src, gt_f, torch.zeros_like(gt_f))
            depth_sum = F.interpolate(
                (gt_clean * valid_f).unsqueeze(1),
                size=(H_t, W_t),
                mode="area",
            )[:, 0]
            valid_ratio = F.interpolate(
                valid_f.unsqueeze(1),
                size=(H_t, W_t),
                mode="area",
            )[:, 0]
            gt_down = depth_sum / valid_ratio.clamp(min=1e-6)
            valid = valid_ratio > float(min_valid_ratio)

        depth_labels = (gt_down - (float(depth_start) - float(depth_step))) / float(depth_step)
        valid = valid & (depth_labels >= 1.0) & (depth_labels < float(D + 1))
        valid_count = valid.sum()
        if not bool(valid_count.item()):
            zero = logits.sum() * 0.0
            return zero, zero.detach(), valid_count.to(dtype=torch.float32)

        label_ids = depth_labels.long().clamp(min=1, max=D) - 1
        logits_flat = logits.permute(0, 2, 3, 1).contiguous()[valid]
        labels_flat = label_ids[valid]
        raw_loss = F.cross_entropy(logits_flat, labels_flat, reduction="mean")
        weighted_loss = float(loss_weight) * raw_loss
    return weighted_loss, raw_loss.detach(), valid_count.to(dtype=torch.float32)


def dense_depth_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    has_dense_depth: Optional[torch.Tensor] = None,
    min_depth: float = 1.0,
    max_depth: float = 80.0,
    loss_weight: float = 0.3,
    si_weight: float = 0.05,
    target_index: int = 0,
    eps: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masked continuous dense-depth loss on the target frame only."""
    if pred_depth.ndim == 5:
        pred = pred_depth[:, int(target_index), 0]
    elif pred_depth.ndim == 4:
        if pred_depth.shape[1] != 1:
            raise RuntimeError(
                f"4D pred_depth must be (B, 1, H, W), got {tuple(pred_depth.shape)}."
            )
        pred = pred_depth[:, 0]
    elif pred_depth.ndim == 3:
        pred = pred_depth
    else:
        raise RuntimeError(
            "pred_depth must be (B,H,W), (B,1,H,W), or (B,N,1,H,W), got "
            f"{tuple(pred_depth.shape)}."
        )

    gt = gt_depth
    if gt.ndim == 5:
        gt = gt[:, int(target_index), 0]
    elif gt.ndim == 4:
        if gt.shape[1] == 1:
            gt = gt[:, 0]
        else:
            gt = gt[:, int(target_index)]
    elif gt.ndim != 3:
        raise RuntimeError(
            f"gt_depth must be (B,H,W), (B,N,H,W), or (B,N,1,H,W), got {tuple(gt.shape)}."
        )
    if gt.shape != pred.shape:
        raise RuntimeError(
            f"dense depth GT shape {tuple(gt.shape)} does not match prediction {tuple(pred.shape)}."
        )

    device_type = pred.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):
        pred_f = pred.float().clamp(min=float(eps))
        gt_f = gt.to(device=pred_f.device, dtype=torch.float32)
        valid = (
            torch.isfinite(pred_f)
            & torch.isfinite(gt_f)
            & (gt_f >= float(min_depth))
            & (gt_f <= float(max_depth))
        )
        if has_dense_depth is not None:
            has = has_dense_depth.to(device=pred_f.device, dtype=torch.bool)
            if has.ndim == 2:
                has = has[:, int(target_index)]
            elif has.ndim != 1:
                raise RuntimeError(
                    "has_dense_depth must be (B,) or (B,N), got "
                    f"{tuple(has_dense_depth.shape)}."
                )
            valid = valid & has.view(-1, 1, 1)

        valid_count = valid.sum()
        if not bool(valid_count.item()):
            zero = pred_f.sum() * 0.0
            return zero, zero.detach(), zero.detach(), valid_count.to(dtype=torch.float32)

        log_diff = torch.log(pred_f[valid]) - torch.log(gt_f[valid].clamp(min=float(eps)))
        log_l1 = log_diff.abs().mean()
        if float(si_weight) > 0.0:
            si_loss = (log_diff.square().mean() - log_diff.mean().square()).clamp(min=0.0)
        else:
            si_loss = log_l1 * 0.0
        raw_loss = log_l1 + float(si_weight) * si_loss
        weighted_loss = float(loss_weight) * raw_loss
    return (
        weighted_loss,
        log_l1.detach(),
        si_loss.detach(),
        valid_count.to(dtype=torch.float32),
    )


class PerFrameLidarMemory(nn.Module):
    """Dense 32-channel LiDAR memory volume in each frame's velo coordinates."""

    def __init__(
        self,
        voxel_origin: Tuple[float, float, float] = (0.0, -25.6, -2.0),
        voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.4),
        grid_size: Tuple[int, int, int] = (128, 128, 16),
        d_voxel: int = 128,
        out_channels: int = 32,
        hidden: int = 64,
        pe_num_freqs: int = 8,
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.encoder = PerFrameMemoryVoxelEncoder(
            vox_origin=voxel_origin,
            vox_size=voxel_size,
            vox_grid=grid_size,
            d_voxel=d_voxel,
            d_out=out_channels,
            hidden=hidden,
            pe_num_freqs=pe_num_freqs,
        )
        self.empty_embed = nn.Parameter(torch.zeros(out_channels))

    def forward(
        self,
        points_per_frame: List[List[torch.Tensor]],
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        B = len(points_per_frame)
        if B == 0:
            raise RuntimeError("points_per_frame must contain at least one sample.")
        N = len(points_per_frame[0])
        memories: List[torch.Tensor] = []
        for f in range(N):
            points_list = [points_per_frame[b][f] for b in range(B)]
            mem, occ = self.encoder(points_list)
            mem = mem.to(dtype=output_dtype)
            occ = occ.to(dtype=output_dtype)
            empty = self.empty_embed.view(1, -1, 1, 1, 1).to(
                device=mem.device, dtype=output_dtype
            )
            memories.append(mem * occ + empty * (1.0 - occ))
        return torch.stack(memories, dim=1).contiguous()  # (B, N, C, X, Y, Z)


class PerFrameNattenFusion(nn.Module):
    """Per-frame 3D NA cross-attention, Q=LSS and KV=LiDAR memory."""

    def __init__(
        self,
        channels: int = 32,
        num_heads: int = 2,
        kernel_size: int = 7,
        num_layers: int = 1,
        ffn_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                NA3DCrossAttnBlock(
                    d_model=int(channels),
                    num_heads=int(num_heads),
                    kernel_size=int(kernel_size),
                    ffn_ratio=float(ffn_ratio),
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(self, lss: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if lss.shape != memory.shape:
            raise RuntimeError(
                f"NATTEN fusion requires matching LSS/memory shapes, got "
                f"{tuple(lss.shape)} vs {tuple(memory.shape)}."
            )
        B, N, C, X, Y, Z = lss.shape
        q = lss.reshape(B * N, C, X, Y, Z).permute(0, 2, 3, 4, 1).contiguous()
        kv = memory.reshape(B * N, C, X, Y, Z).permute(0, 2, 3, 4, 1).contiguous()
        out = q
        for block in self.blocks:
            out = block(out, kv)
        return (
            out.permute(0, 4, 1, 2, 3)
            .contiguous()
            .view(B, N, C, X, Y, Z)
        )


class FrameVolumeWarper(nn.Module):
    """Warp per-frame velo volumes to the target velo half-grid."""

    def __init__(
        self,
        voxel_origin: Tuple[float, float, float] = (0.0, -25.6, -2.0),
        voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.4),
        grid_size: Tuple[int, int, int] = (128, 128, 16),
    ) -> None:
        super().__init__()
        self.grid_size = tuple(int(v) for v in grid_size)
        Gx, Gy, Gz = self.grid_size
        ix = torch.arange(Gx, dtype=torch.float32)
        iy = torch.arange(Gy, dtype=torch.float32)
        iz = torch.arange(Gz, dtype=torch.float32)
        grid_ix, grid_iy, grid_iz = torch.meshgrid(ix, iy, iz, indexing="ij")
        idx = torch.stack([grid_ix, grid_iy, grid_iz], dim=-1)
        origin = torch.tensor(voxel_origin, dtype=torch.float32)
        size = torch.tensor(voxel_size, dtype=torch.float32)
        centers = (idx + 0.5) * size + origin
        self.register_buffer("target_centers", centers, persistent=False)
        self.register_buffer("voxel_origin", origin, persistent=False)
        self.register_buffer("voxel_size", size, persistent=False)

    def _warp_one(
        self,
        volume: torch.Tensor,  # (B, C, X, Y, Z)
        T_target_from_frame_velo: torch.Tensor,
    ) -> torch.Tensor:
        B, _C, Gx, Gy, Gz = volume.shape
        device = volume.device
        T_inv = torch.linalg.inv(T_target_from_frame_velo.to(device=device, dtype=torch.float32))
        R_inv = T_inv[:, :3, :3]
        t_inv = T_inv[:, :3, 3]

        centers = self.target_centers.to(device=device, dtype=torch.float32)
        flat_centers = centers.reshape(-1, 3).unsqueeze(0).expand(B, -1, 3)
        p_frame = torch.einsum("bij,bvj->bvi", R_inv, flat_centers) + t_inv.unsqueeze(1)
        p_frame = p_frame.view(B, Gx, Gy, Gz, 3)

        origin = self.voxel_origin.to(device=device, dtype=torch.float32)
        size = self.voxel_size.to(device=device, dtype=torch.float32)
        frac = (p_frame - origin) / size
        norm_x = (frac[..., 0] / float(Gx)) * 2.0 - 1.0
        norm_y = (frac[..., 1] / float(Gy)) * 2.0 - 1.0
        norm_z = (frac[..., 2] / float(Gz)) * 2.0 - 1.0
        sample_grid = torch.stack([norm_z, norm_y, norm_x], dim=-1)

        warped = F.grid_sample(
            volume.to(dtype=torch.float32),
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return warped.to(dtype=volume.dtype)

    def forward(
        self,
        volumes: torch.Tensor,                 # (B, N, C, X, Y, Z)
        T_target_from_refcam: torch.Tensor,    # (B, 4, 4)
        T_cam_from_velo: torch.Tensor,         # (B, 4, 4)
        cam2world_per_frame: torch.Tensor,     # (B, N, 4, 4)
    ) -> torch.Tensor:
        B, N, C, X, Y, Z = volumes.shape
        if (X, Y, Z) != self.grid_size:
            raise RuntimeError(
                f"volume grid {(X, Y, Z)} does not match warper grid {self.grid_size}."
            )
        device = volumes.device
        T_tr = T_target_from_refcam.to(device=device, dtype=torch.float32)
        T_cv = T_cam_from_velo.to(device=device, dtype=torch.float32)
        c2w = cam2world_per_frame.to(device=device, dtype=torch.float32)
        c2w_ref_inv = torch.linalg.inv(c2w[:, 0])

        warped: List[torch.Tensor] = []
        for f in range(N):
            T_target_from_frame_velo = T_tr @ c2w_ref_inv @ c2w[:, f] @ T_cv
            warped.append(self._warp_one(volumes[:, f], T_target_from_frame_velo))
        return torch.stack(warped, dim=1).view(B, N, C, X, Y, Z).contiguous()


class Stage1SSCBEVDetOccLidarModel(nn.Module):
    """Frozen OccAny + 2D cross-attn fusion + BEVDet-OCC style 3D backend."""

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
        half_voxel_size: Tuple[float, float, float] = (0.4, 0.4, 0.4),
        half_grid_size: Tuple[int, int, int] = (128, 128, 16),
        depth_bound: Tuple[float, float, float] = (1.0, 52.0, 0.4),
        # Early 2D LiDAR/image fusion controls. This variant defaults to cross.
        fusion_vox_origin: Tuple[float, float, float] = (-25.6, -2.0, 0.0),
        fusion_vox_size: Tuple[float, float, float] = (0.4, 0.4, 0.4),
        fusion_vox_grid: Tuple[int, int, int] = (128, 16, 128),
        fusion_num_heads: int = 8,
        fusion_window: int = 4,
        fusion_d_voxel: int = 128,
        fusion_pe_num_freqs: int = 8,
        fusion_attn_type: str = "cross",
        lss_in_channels: int = 256,
        lss_out_channels: int = 32,
        use_shared_geometry_adapter: bool = False,
        geometry_channels: int = 256,
        geometry_adapter_gate_init: float = 0.0,
        lidar_d_voxel: int = 128,
        lidar_hidden: int = 64,
        lidar_pe_num_freqs: int = 8,
        natten_kernel: int = 7,
        natten_num_heads: int = 2,
        natten_num_layers: int = 1,
        natten_ffn_ratio: float = 2.0,
        temporal_channels: int = 64,
        bevdet_neck_channels: int = 32,
        num_frames: int = 5,
        with_cp: bool = False,
    ) -> None:
        super().__init__()
        del c_lift  # kept for train.py compatibility; this backend does not use Stage1LiftingModule.
        self.num_frames = int(num_frames)
        self.half_grid_size = tuple(int(v) for v in half_grid_size)

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
        if fusion_attn_type != "cross":
            raise ValueError(
                "Stage1SSCBEVDetOccLidarModel keeps the initial 2D fusion as "
                f"cross-attention; got fusion_attn_type={fusion_attn_type!r}."
            )
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

        self.token_projector = OccAnyTokenProjector(
            token_dim=token_dim,
            out_channels=lss_in_channels,
        )
        self.use_shared_geometry_adapter = bool(use_shared_geometry_adapter)
        if self.use_shared_geometry_adapter:
            self.geometry_adapter = SharedGeometryAdapter(
                in_channels=lss_in_channels,
                geometry_channels=geometry_channels,
                out_channels=lss_in_channels,
                residual_gate_init=geometry_adapter_gate_init,
            )
            self.dense_depth_head = DenseDepthHead(in_channels=geometry_channels)
        else:
            self.geometry_adapter = None
            self.dense_depth_head = None
        self.lss = LSSDepthLift(
            in_channels=lss_in_channels,
            out_channels=lss_out_channels,
            depth_bound=depth_bound,
            voxel_origin=voxel_origin,
            voxel_size=half_voxel_size,
            grid_size=half_grid_size,
        )
        self.lidar_memory = PerFrameLidarMemory(
            voxel_origin=voxel_origin,
            voxel_size=half_voxel_size,
            grid_size=half_grid_size,
            d_voxel=lidar_d_voxel,
            out_channels=lss_out_channels,
            hidden=lidar_hidden,
            pe_num_freqs=lidar_pe_num_freqs,
        )
        self.natten_fusion = PerFrameNattenFusion(
            channels=lss_out_channels,
            num_heads=natten_num_heads,
            kernel_size=natten_kernel,
            num_layers=natten_num_layers,
            ffn_ratio=natten_ffn_ratio,
        )
        self.warper = FrameVolumeWarper(
            voxel_origin=voxel_origin,
            voxel_size=half_voxel_size,
            grid_size=half_grid_size,
        )

        per_frame_channels = int(lss_out_channels) * 2
        if int(temporal_channels) != per_frame_channels:
            raise ValueError(
                "temporal_channels must match concat(enhanced_lss, memory), "
                f"got temporal_channels={temporal_channels}, expected {per_frame_channels}."
            )
        self.temporal_reduce = conv_bn_relu_3d(
            self.num_frames * per_frame_channels,
            int(temporal_channels),
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.occ_head = BEVDetOcc3DHead(
            num_classes=num_classes,
            in_channels=temporal_channels,
            neck_channels=bevdet_neck_channels,
            full_grid=grid_size,
            with_cp=with_cp,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @staticmethod
    def _stack_cam2world(
        views: List[Dict[str, torch.Tensor]],
        device: torch.device,
    ) -> torch.Tensor:
        if len(views) == 0:
            raise RuntimeError("views must contain at least one frame.")
        if "cam2world" not in views[0]:
            raise RuntimeError(
                "BEVDetOcc LiDAR backend requires views[f]['cam2world'] for every frame."
            )
        return torch.stack(
            [v["cam2world"].to(device=device, dtype=torch.float32) for v in views],
            dim=1,
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
        return_lss_depth: bool = False,
        dense_depth_gt: Optional[torch.Tensor] = None,
        has_dense_depth: Optional[torch.Tensor] = None,
        return_dense_depth: bool = False,
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
        B, N = t_rec_fused.shape[:2]
        if N != self.num_frames:
            raise RuntimeError(
                f"model was built for num_frames={self.num_frames}, got input N={N}."
            )

        feat_2d = self.token_projector(t_rec_fused)
        shared_geometry_feat: Optional[torch.Tensor] = None
        if self.geometry_adapter is not None:
            feat_2d, shared_geometry_feat = self.geometry_adapter(feat_2d)
        elif return_dense_depth:
            raise RuntimeError(
                "return_dense_depth=True requires use_shared_geometry_adapter=True."
            )
        pred_dense_depth: Optional[torch.Tensor] = None
        if return_dense_depth:
            if self.dense_depth_head is None or shared_geometry_feat is None:
                raise RuntimeError("dense depth head is not initialized.")
            pred_dense_depth = self.dense_depth_head(
                shared_geometry_feat[:, 0],
                image_hw.to(device=feat_2d.device),
            )
        lss_volume, depth_logits = self.lss(
            feat_2d=feat_2d,
            K_per_frame=K_per_frame.to(device=feat_2d.device),
            T_cam_from_velo=T_cam_from_velo.to(device=feat_2d.device),
            image_hw=image_hw.to(device=feat_2d.device),
            gt_depth=gt_depth,
        )
        memory = self.lidar_memory(points_per_frame, output_dtype=lss_volume.dtype)
        memory = memory.to(device=lss_volume.device, dtype=lss_volume.dtype)
        enhanced = self.natten_fusion(lss_volume, memory)
        per_frame = torch.cat([enhanced, memory], dim=2)  # (B, N, 64, X, Y, Z)

        cam2world = self._stack_cam2world(views, device=per_frame.device)
        warped = self.warper(
            per_frame,
            T_target_from_refcam=T_target_from_refcam.to(device=per_frame.device),
            T_cam_from_velo=T_cam_from_velo.to(device=per_frame.device),
            cam2world_per_frame=cam2world,
        )
        B, N, C, X, Y, Z = warped.shape
        temporal = warped.view(B, N * C, X, Y, Z)
        temporal = self.temporal_reduce(temporal)
        logits = self.occ_head(temporal)
        out: Dict[str, torch.Tensor] = {"ssc_logit": logits}
        if return_dense_depth:
            if pred_dense_depth is None:
                raise RuntimeError("dense depth prediction was not computed.")
            out["pred_dense_depth"] = pred_dense_depth
        if dense_depth_gt is not None:
            out["dense_depth_gt"] = dense_depth_gt.to(
                device=depth_logits.device, dtype=torch.float32
            )
        if has_dense_depth is not None:
            out["has_dense_depth"] = has_dense_depth.to(
                device=depth_logits.device, dtype=torch.bool
            )
        if return_depth or return_lss_depth:
            out.update(
                depth_logits=depth_logits,
                depth_start=self.lss.depth_start,
                depth_step=self.lss.depth_step,
            )
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
                gt_depth=gt_depth.to(device=depth_logits.device, dtype=torch.float32),
            )
        return out


__all__ = [
    "Stage1SSCBEVDetOccLidarModel",
    "bevdet_depth_loss",
    "dense_lss_depth_loss",
    "dense_depth_loss",
]
