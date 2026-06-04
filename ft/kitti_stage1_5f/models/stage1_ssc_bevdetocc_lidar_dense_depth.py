"""BEVDet-OCC LiDAR model with DPT-style dense depth auxiliary supervision."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage1_ssc_bevdetocc_lidar import Stage1SSCBEVDetOccLidarModel


class _ResidualConvUnit(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + residual


class _FeatureFusionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.res1 = _ResidualConvUnit(channels)
        self.res2 = _ResidualConvUnit(channels)

    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
            x = x + self.res1(skip)
        return self.res2(x)


class SingleScaleDPTDepthHead(nn.Module):
    """DPT-style dense depth head for a single post-fusion token map.

    Depth-Anything-3's DPT head fuses four transformer feature levels. This
    head keeps the same projection/resize/fusion idea but synthesizes the
    pyramid from one ``(B, N, H_t, W_t, C)`` token map.
    """

    def __init__(
        self,
        token_dim: int = 768,
        patch_size: int = 16,
        features: int = 128,
        out_channels: Tuple[int, int, int, int] = (96, 192, 384, 384),
        initial_depth: float = 10.0,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.norm = nn.LayerNorm(int(token_dim))
        self.projects = nn.ModuleList(
            [nn.Conv2d(int(token_dim), int(c), kernel_size=1) for c in out_channels]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )
        self.adapters = nn.ModuleList(
            [nn.Conv2d(int(c), int(features), kernel_size=3, padding=1) for c in out_channels]
        )
        self.refinenet1 = _FeatureFusionBlock(int(features))
        self.refinenet2 = _FeatureFusionBlock(int(features))
        self.refinenet3 = _FeatureFusionBlock(int(features))
        self.refinenet4 = _FeatureFusionBlock(int(features))
        self.output_conv1 = nn.Conv2d(int(features), int(features) // 2, kernel_size=3, padding=1)
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(int(features) // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        nn.init.constant_(self.output_conv2[-1].bias, float(initial_depth))

    def forward(
        self,
        tokens: torch.Tensor,
        image_hw: torch.Tensor,
    ) -> torch.Tensor:
        B, N, H_t, W_t, C = tokens.shape
        x = self.norm(tokens)
        x = x.reshape(B * N, H_t, W_t, C).permute(0, 3, 1, 2).contiguous()

        feats = []
        for project, resize, adapter in zip(self.projects, self.resize_layers, self.adapters):
            feats.append(adapter(resize(project(x))))

        path = self.refinenet4(feats[3])
        path = self.refinenet3(path, feats[2])
        path = self.refinenet2(path, feats[1])
        path = self.refinenet1(path, feats[0])

        H_img = int(image_hw[0, 0].item())
        W_img = int(image_hw[0, 1].item())
        if not bool((image_hw[:, 0] == H_img).all().item()) or not bool(
            (image_hw[:, 1] == W_img).all().item()
        ):
            raise RuntimeError("SingleScaleDPTDepthHead expects same image_hw within a batch.")

        path = self.output_conv1(path)
        path = F.interpolate(path, size=(H_img, W_img), mode="bilinear", align_corners=True)
        logits = self.output_conv2(path)
        depth = F.softplus(logits.float()).to(dtype=tokens.dtype) + 1e-3
        return depth.view(B, N, H_img, W_img)


def dense_metric_depth_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    frame_mask: Optional[torch.Tensor] = None,
    loss_weight: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masked metric dense-depth loss.

    Uses log-depth L1 plus relative L1 over finite positive depth pixels. Frames
    with no dense depth are skipped by ``frame_mask`` or naturally by the valid
    pixel mask.
    """
    if pred_depth.shape != gt_depth.shape:
        raise RuntimeError(
            f"pred_depth shape {tuple(pred_depth.shape)} != gt_depth {tuple(gt_depth.shape)}"
        )

    device_type = pred_depth.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):
        pred = pred_depth.float().clamp(min=1e-3, max=120.0)
        gt = gt_depth.to(device=pred.device, dtype=torch.float32)
        valid = torch.isfinite(pred) & torch.isfinite(gt) & (gt > 0.0)
        if frame_mask is not None:
            fm = frame_mask.to(device=pred.device, dtype=torch.bool).view(
                pred.shape[0], pred.shape[1], 1, 1
            )
            valid = valid & fm

        valid_count = valid.sum()
        frame_count = valid.view(pred.shape[0], pred.shape[1], -1).any(dim=-1).sum()
        if not bool(valid_count.item()):
            zero = pred.sum() * 0.0
            return zero, zero.detach(), valid_count.float(), frame_count.float()

        pred_v = pred[valid]
        gt_v = gt[valid].clamp(min=1e-3, max=120.0)
        log_l1 = F.l1_loss(torch.log(pred_v), torch.log(gt_v), reduction="mean")
        rel_l1 = (pred_v - gt_v).abs().div(gt_v.clamp(min=1.0)).mean()
        raw_loss = log_l1 + rel_l1
        weighted_loss = float(loss_weight) * raw_loss
    return weighted_loss, raw_loss.detach(), valid_count.float(), frame_count.float()


class Stage1SSCBEVDetOccLidarDenseDepthModel(Stage1SSCBEVDetOccLidarModel):
    """Full BEVDet-OCC LiDAR model with a post-2D-fusion dense depth head."""

    def __init__(
        self,
        *args,
        dense_depth_features: int = 128,
        dense_depth_initial: float = 10.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        token_dim = int(kwargs.get("token_dim", 768))
        patch_size = int(kwargs.get("patch_size", 16))
        self.dense_depth_head = SingleScaleDPTDepthHead(
            token_dim=token_dim,
            patch_size=patch_size,
            features=int(dense_depth_features),
            initial_depth=float(dense_depth_initial),
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

        dense_depth = self.dense_depth_head(
            t_rec_fused,
            image_hw.to(device=t_rec_fused.device),
        )

        feat_2d = self.token_projector(t_rec_fused)
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
        per_frame = torch.cat([enhanced, memory], dim=2)

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
        out: Dict[str, torch.Tensor] = {
            "ssc_logit": logits,
            "dense_depth": dense_depth,
        }
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
    "SingleScaleDPTDepthHead",
    "Stage1SSCBEVDetOccLidarDenseDepthModel",
    "dense_metric_depth_loss",
]
