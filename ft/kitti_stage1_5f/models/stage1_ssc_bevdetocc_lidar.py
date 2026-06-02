"""Stage-1 SSC with OccAny token fusion and a BEVDet-OCC style 3D head.

This variant keeps the earliest LiDAR/image 2D cross-attention on OccAny
decoder tokens, then replaces the old lifting + MonoScene head with:

  token 1x1 projection -> LSS depth lifting on a half KITTI grid ->
  per-frame LiDAR voxel memory -> 3D NATTEN cross-attention ->
  temporal warp/concat -> BEVDet CustomResNet3D + LSSFPN3D ->
  full-grid upsample -> final_conv + predicter.

The returned layout is always ``{"ssc_logit": (B, 20, X, Y, Z)}``.
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
        del gt_depth  # Reserved for future sparse depth supervision.

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
        lss_volume, _depth_logits = self.lss(
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
        return {"ssc_logit": logits}


__all__ = ["Stage1SSCBEVDetOccLidarModel"]
