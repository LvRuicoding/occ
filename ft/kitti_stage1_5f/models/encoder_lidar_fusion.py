"""Encoder-side LiDAR/image fusion for Stage-1 depth experiments."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .lidar_fusion import VoxelFeatureEncoder, WindowedCrossAttnLayer


def parse_encoder_lidar_layers(
    layers: Optional[str | Sequence[int]],
    *,
    depth: int,
) -> Tuple[int, ...]:
    """Parse 1-based user layer ids into sorted 0-based block indices."""
    if layers is None:
        return ()
    if isinstance(layers, str):
        value = layers.strip()
        if value == "" or value.lower() in {"none", "off", "false", "0"}:
            return ()
        parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
    else:
        parts = list(layers)

    out = []
    for part in parts:
        idx_1based = int(part)
        if idx_1based < 1 or idx_1based > int(depth):
            raise ValueError(
                f"encoder_lidar_layers must be in [1, {int(depth)}], got {idx_1based}."
            )
        out.append(idx_1based - 1)
    return tuple(sorted(set(out)))


class EncoderLidarFusionLayer(nn.Module):
    """One alpha-gated encoder residual: alpha * cross_attn(norm_lidar(x), lidar)."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        H_t: int,
        W_t: int,
        window: int,
        ffn_ratio: float = 2.0,
        alpha_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.H_t = int(H_t)
        self.W_t = int(W_t)
        self.norm_lidar = nn.LayerNorm(int(d_model))
        self.cross_attn = WindowedCrossAttnLayer(
            d_model=int(d_model),
            num_heads=int(num_heads),
            window=int(window),
            shift=0,
            H_t=self.H_t,
            W_t=self.W_t,
            ffn_ratio=float(ffn_ratio),
        )
        # The outer norm_lidar is the query normalization for this residual.
        self.cross_attn.norm_q = nn.Identity()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self._zero_init_update_path()

    def _zero_init_update_path(self) -> None:
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        if self.cross_attn.out_proj.bias is not None:
            nn.init.zeros_(self.cross_attn.out_proj.bias)
        last = self.cross_attn.ffn[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor, context: Dict[str, torch.Tensor]) -> torch.Tensor:
        F_n, L, D = x.shape
        if L != self.H_t * self.W_t:
            raise RuntimeError(
                f"encoder token grid mismatch: {L} tokens != {self.H_t}*{self.W_t}."
            )

        x_norm = self.norm_lidar(x)
        image_feat = x_norm.view(F_n, self.H_t, self.W_t, D).contiguous()
        update = self.cross_attn(
            image_feat,
            context["voxel_feat"].to(device=x.device, dtype=x.dtype),
            context["voxel_frame_idx"].to(device=x.device),
            context["voxel_h_t"].to(device=x.device),
            context["voxel_w_t"].to(device=x.device),
            context["voxel_valid"].to(device=x.device),
            return_update=True,
        )
        update = update.view(F_n, L, D).to(dtype=x.dtype)
        return self.alpha.to(device=x.device, dtype=x.dtype) * update


class EncoderLidarFusion(nn.Module):
    """Shared VFE plus selected per-layer encoder fusion residuals."""

    def __init__(
        self,
        *,
        d_model: int,
        selected_layers: Sequence[int],
        H_t: int,
        W_t: int,
        patch_size: int,
        num_heads: int = 8,
        window: int = 4,
        vox_origin: Tuple[float, float, float] = (-25.6, -2.0, 0.0),
        vox_size: Tuple[float, float, float] = (0.4, 0.4, 0.4),
        vox_grid: Tuple[int, int, int] = (128, 16, 128),
        vfe_d_voxel: int = 128,
        vfe_hidden: int = 64,
        pe_num_freqs: int = 8,
        ffn_ratio: float = 2.0,
        alpha_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.H_t = int(H_t)
        self.W_t = int(W_t)
        self.patch_size = int(patch_size)
        self.selected_layers = tuple(int(v) for v in selected_layers)
        self.vfe = VoxelFeatureEncoder(
            vox_origin=vox_origin,
            vox_size=vox_size,
            vox_grid=vox_grid,
            d_voxel=int(vfe_d_voxel),
            d_out=self.d_model,
            hidden=int(vfe_hidden),
            pe_num_freqs=int(pe_num_freqs),
        )
        self.layers = nn.ModuleDict(
            {
                str(layer_idx): EncoderLidarFusionLayer(
                    d_model=self.d_model,
                    num_heads=int(num_heads),
                    H_t=self.H_t,
                    W_t=self.W_t,
                    window=int(window),
                    ffn_ratio=float(ffn_ratio),
                    alpha_init=float(alpha_init),
                )
                for layer_idx in self.selected_layers
            }
        )

    def has_layer(self, layer_idx: int) -> bool:
        return str(int(layer_idx)) in self.layers

    def forward(
        self,
        x: torch.Tensor,
        *,
        layer_idx: int,
        context: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.layers[str(int(layer_idx))](x, context)

    def _project_voxels_to_patches(
        self,
        voxel_center_cam: torch.Tensor,
        K: torch.Tensor,
        img_H: int,
        img_W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        K = K.to(dtype=torch.float32)
        center = voxel_center_cam.to(dtype=torch.float32)
        z = center[:, 2]
        valid_z = z > 0.1
        z_safe = torch.where(valid_z, z, torch.ones_like(z))
        uv = (K @ center.T).T
        u = uv[:, 0] / z_safe
        v = uv[:, 1] / z_safe
        valid = valid_z & (u >= 0) & (u < img_W) & (v >= 0) & (v < img_H)
        h_t = (v / self.patch_size).floor().long().clamp(0, self.H_t - 1)
        w_t = (u / self.patch_size).floor().long().clamp(0, self.W_t - 1)
        return h_t, w_t, valid

    def _empty_context(self, device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            "voxel_feat": torch.zeros((0, self.d_model), dtype=torch.float32, device=device),
            "voxel_frame_idx": torch.zeros((0,), dtype=torch.long, device=device),
            "voxel_h_t": torch.zeros((0,), dtype=torch.long, device=device),
            "voxel_w_t": torch.zeros((0,), dtype=torch.long, device=device),
            "voxel_valid": torch.zeros((0,), dtype=torch.bool, device=device),
        }

    def build_context(
        self,
        *,
        points_per_frame: List[List[torch.Tensor]],
        T_cam_from_velo: torch.Tensor,
        K_per_frame: torch.Tensor,
        image_hw: torch.Tensor,
        B: int,
        N: int,
        device: torch.device,
        fusion_vox_origin: Optional[torch.Tensor] = None,
        fusion_vox_size: Optional[torch.Tensor] = None,
        fusion_vox_grid: Optional[Tuple[int, int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        if len(points_per_frame) != B:
            raise RuntimeError(
                f"points_per_frame batch size {len(points_per_frame)} != image batch {B}."
            )
        T_cv_all = T_cam_from_velo.to(device=device, dtype=torch.float32)
        if T_cv_all.ndim == 3:
            T_cv_all = T_cv_all[:, None].expand(B, N, 4, 4)
        if T_cv_all.ndim != 4 or T_cv_all.shape[:2] != (B, N):
            raise RuntimeError(
                f"T_cam_from_velo must be (B,4,4) or (B,N,4,4), got {tuple(T_cv_all.shape)}."
            )
        K_all = K_per_frame.to(device=device, dtype=torch.float32)
        image_hw = image_hw.to(device=device, dtype=torch.long)
        dyn_origin = (
            None
            if fusion_vox_origin is None
            else fusion_vox_origin.to(device=device, dtype=torch.float32)
        )
        dyn_size = (
            None
            if fusion_vox_size is None
            else fusion_vox_size.to(device=device, dtype=torch.float32)
        )

        feats = []
        frame_indices = []
        patch_h = []
        patch_w = []
        valid_masks = []
        for b in range(B):
            if len(points_per_frame[b]) != N:
                raise RuntimeError(
                    f"points_per_frame[{b}] has {len(points_per_frame[b])} frames, expected {N}."
                )
            img_H = int(image_hw[b, 0].item())
            img_W = int(image_hw[b, 1].item())
            for f in range(N):
                pts = points_per_frame[b][f].to(device=device, non_blocking=True)
                feat, center = self.vfe(
                    pts,
                    T_cv_all[b, f],
                    vox_origin=None if dyn_origin is None else dyn_origin[b],
                    vox_size=None if dyn_size is None else dyn_size[b],
                    vox_grid=fusion_vox_grid,
                )
                if feat is None or center is None or feat.shape[0] == 0:
                    continue
                h_t, w_t, valid = self._project_voxels_to_patches(
                    center,
                    K_all[b, f],
                    img_H,
                    img_W,
                )
                global_frame = b * N + f
                frame_idx = torch.full(
                    (feat.shape[0],), global_frame, dtype=torch.long, device=device
                )
                feats.append(feat)
                frame_indices.append(frame_idx)
                patch_h.append(h_t)
                patch_w.append(w_t)
                valid_masks.append(valid)

        if not feats:
            return self._empty_context(device)
        return {
            "voxel_feat": torch.cat(feats, dim=0).to(dtype=torch.float32),
            "voxel_frame_idx": torch.cat(frame_indices, dim=0),
            "voxel_h_t": torch.cat(patch_h, dim=0),
            "voxel_w_t": torch.cat(patch_w, dim=0),
            "voxel_valid": torch.cat(valid_masks, dim=0),
        }


__all__ = [
    "EncoderLidarFusion",
    "EncoderLidarFusionLayer",
    "parse_encoder_lidar_layers",
]
