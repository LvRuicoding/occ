"""Stage-1 lifting: configurable OccAny recon backbone + confidence-weighted voxel lift.

Implements stage1.txt:
- Dust3rEncoder + Must3rDecoder run once on 5 frames (target frame
  first → reconstruction reference). Outputs T_rec (patch tokens),
  P_rec_global (pixel pointmaps in refcam coords), C_rec (pixel confidence).
- Pixel feature MLP consumes nearest-upsampled T_rec concatenated with
  normalized pixel coords, producing f_rec at pointmap resolution.
- Pixels are transformed via T_target_from_refcam and scatter-added into a
  voxel grid with confidence weights. Returns V_rec and W_rec.
"""
from __future__ import annotations

from .. import _paths  # noqa: F401

from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from occany.model.model_must3r import Dust3rEncoder, Must3rDecoder
from occany.model.must3r_blocks.head import ActivationType
from occany.must3r_inference import (
    inference_encoder,
    postprocess,
    prepare_imgs_or_raymaps_and_true_shape_mem_batches,
)
from occany.utils.checkpoint_io import register_legacy_checkpoint_modules

from .encoder_lidar_fusion import EncoderLidarFusion, parse_encoder_lidar_layers


class _DecoderNormCapturer:
    """Forward-hook helper that records the output of decoder.norm_dec.

    The decoder applies norm_dec to its last-block feature in
    `_compute_prediction_head`. The output shape is (B*nimgs, N+1, D) where
    the first token along N is the pose token.
    """

    def __init__(self, detach_output: bool = True) -> None:
        self._handle = None
        self._captured: Optional[torch.Tensor] = None
        self.detach_output = bool(detach_output)

    def _hook(self, module, inputs, output):
        self._captured = output.detach() if self.detach_output else output

    def attach(self, norm_module: nn.Module) -> None:
        if self._handle is not None:
            raise RuntimeError("capturer already attached")
        self._handle = norm_module.register_forward_hook(self._hook)
        self._captured = None

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def pop(self) -> torch.Tensor:
        if self._captured is None:
            raise RuntimeError("no capture available")
        out = self._captured
        self._captured = None
        return out


class OccAnyRecon5FrameBackbone(nn.Module):
    """OccAny reconstruction encoder + decoder for 5 (or N) frames.

    The first view passed in `forward(views)` is the reconstruction reference;
    the rest are joined via Must3rDecoder's joint pass. `p_rec_global` is in
    the *reference camera* coordinate system. In the Stage-1 pipeline we set
    view 0 = target frame's left camera (cam2), so `T_target_from_refcam` is
    the static `T_velo_from_cam2`. Don't change the order in `views` without
    also updating the dataset / downstream transform.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (512, 512),
        enc_embed_dim: int = 1024,
        embed_dim: int = 768,
        patch_size: int = 16,
        backbone_dtype: torch.dtype = torch.bfloat16,
        freeze: bool = True,
        encoder_lidar_layers: Optional[str | Sequence[int]] = None,
        encoder_lidar_alpha_init: float = 1.0,
        encoder_lidar_num_heads: int = 8,
        encoder_lidar_window: int = 4,
        encoder_lidar_vfe_d_voxel: int = 128,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.backbone_dtype = backbone_dtype
        self.freeze = bool(freeze)

        self.encoder = Dust3rEncoder()
        self.decoder = Must3rDecoder(
            img_size=img_size,
            enc_embed_dim=enc_embed_dim,
            embed_dim=embed_dim,
            pointmaps_activation=ActivationType.LINEAR,
            pred_sam_features=True,
            feedback_type="single_mlp",
            memory_mode="kv",
            ray_map_encoder_depth=6,
            use_multitask_token=True,
        )

        self._capturer = _DecoderNormCapturer(detach_output=self.freeze)
        self.encoder_lidar_layers = parse_encoder_lidar_layers(
            encoder_lidar_layers,
            depth=int(getattr(self.encoder, "depth", 24)),
        )
        H_t = int(img_size[0]) // self.patch_size
        W_t = int(img_size[1]) // self.patch_size
        self.encoder_lidar_fusion = (
            EncoderLidarFusion(
                d_model=int(enc_embed_dim),
                selected_layers=self.encoder_lidar_layers,
                H_t=H_t,
                W_t=W_t,
                patch_size=self.patch_size,
                num_heads=int(encoder_lidar_num_heads),
                window=int(encoder_lidar_window),
                vfe_d_voxel=int(encoder_lidar_vfe_d_voxel),
                alpha_init=float(encoder_lidar_alpha_init),
            )
            if self.encoder_lidar_layers
            else None
        )
        self.set_frozen(self.freeze)

    def set_frozen(self, freeze: bool = True) -> None:
        self.freeze = bool(freeze)
        self._capturer.detach_output = self.freeze
        for p in self.parameters():
            p.requires_grad = not self.freeze
        if self.freeze:
            self.encoder.eval()
            self.decoder.eval()
        else:
            self.encoder.train(self.training)
            self.decoder.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
            self.decoder.eval()
        return self

    def load_checkpoint(self, ckpt_path: str) -> None:
        register_legacy_checkpoint_modules()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        enc_status = self.encoder.load_state_dict(
            ckpt.get("encoder", {}), strict=False
        )
        dec_status = self.decoder.load_state_dict(
            ckpt.get("decoder", {}), strict=False
        )
        print(f"[OccAnyRecon5FrameBackbone] encoder load: missing={len(enc_status.missing_keys)} unexpected={len(enc_status.unexpected_keys)}")
        print(f"[OccAnyRecon5FrameBackbone] decoder load: missing={len(dec_status.missing_keys)} unexpected={len(dec_status.unexpected_keys)}")
        ckpt_args = ckpt.get("args", None)
        if ckpt_args is not None:
            pm_act = getattr(ckpt_args, "pointmaps_activation", None)
            if pm_act is not None:
                self.decoder.pointmaps_activation = pm_act
                print(f"[OccAnyRecon5FrameBackbone] pointmaps_activation set to {pm_act}")
        del ckpt

    def forward(
        self,
        views: List[Dict[str, torch.Tensor]],
        device: Optional[torch.device] = None,
        points_per_frame: Optional[List[List[torch.Tensor]]] = None,
        T_cam_from_velo: Optional[torch.Tensor] = None,
        K_per_frame: Optional[torch.Tensor] = None,
        image_hw: Optional[torch.Tensor] = None,
        fusion_vox_origin: Optional[torch.Tensor] = None,
        fusion_vox_size: Optional[torch.Tensor] = None,
        fusion_vox_grid: Optional[Tuple[int, int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        if device is None:
            device = views[0]["img"].device

        grad_context = torch.no_grad() if self.freeze else nullcontext()
        with grad_context, torch.autocast("cuda", dtype=self.backbone_dtype):
            imgs, true_shape, _mem_batches, _timesteps = (
                prepare_imgs_or_raymaps_and_true_shape_mem_batches(
                    views, device, is_raymap=False
                )
            )
            B, nimgs = imgs.shape[:2]

            if self.encoder_lidar_fusion is None:
                x, pos = inference_encoder(
                    encoder=self.encoder,
                    imgs=imgs,
                    true_shape_view=true_shape.view(B * nimgs, 2),
                    max_bs=None,
                    requires_grad=not self.freeze,
                )
            else:
                if (
                    points_per_frame is None
                    or T_cam_from_velo is None
                    or K_per_frame is None
                    or image_hw is None
                ):
                    raise RuntimeError(
                        "encoder LiDAR fusion requires points_per_frame, "
                        "T_cam_from_velo, K_per_frame, and image_hw."
                    )
                encoder_lidar_context = self.encoder_lidar_fusion.build_context(
                    points_per_frame=points_per_frame,
                    T_cam_from_velo=T_cam_from_velo,
                    K_per_frame=K_per_frame,
                    image_hw=image_hw,
                    B=B,
                    N=nimgs,
                    device=device,
                    fusion_vox_origin=fusion_vox_origin,
                    fusion_vox_size=fusion_vox_size,
                    fusion_vox_grid=fusion_vox_grid,
                )
                imgs_view = imgs.reshape(B * nimgs, *imgs.shape[2:])
                true_shape_view = true_shape.view(B * nimgs, 2)
                x_flat, pos_flat = self.encoder(
                    imgs_view,
                    true_shape_view,
                    encoder_lidar_fusion=self.encoder_lidar_fusion,
                    encoder_lidar_context=encoder_lidar_context,
                )
                x = x_flat.view(B, nimgs, *x_flat.shape[1:])
                pos = pos_flat.view(B, nimgs, *pos_flat.shape[1:])

            # Joint decoder pass: first frame is reference (no image2_embed),
            # remaining nimgs-1 frames get image2_embed.
            self._capturer.attach(self.decoder.norm_dec)
            try:
                _out, pointmaps_raw, _pose_out, _sam_feats = self.decoder(
                    x, pos, true_shape, current_mem=None
                )
                dense_with_pose = self._capturer.pop()
            finally:
                self._capturer.detach()

            post = postprocess(
                pointmaps_raw,
                pose_out=None,
                pointmaps_activation=self.decoder.pointmaps_activation,
                compute_cam=False,
            )

        # T_rec patch tokens: strip pose token; reshape to (B, nimgs, H_t, W_t, D).
        ref_shape = true_shape[0, 0]
        if not torch.equal(true_shape, ref_shape.view(1, 1, 2).expand_as(true_shape)):
            raise RuntimeError(
                "OccAnyRecon5FrameBackbone requires all views to share the same "
                "true_shape; got mixed shapes within a batch."
            )
        H_p = int(ref_shape[0].item())
        W_p = int(ref_shape[1].item())
        H_t = H_p // self.patch_size
        W_t = W_p // self.patch_size
        bxn, N1, D = dense_with_pose.shape
        if bxn != B * nimgs:
            raise RuntimeError(
                f"unexpected dense feature batch size {bxn} vs B*N={B*nimgs}"
            )
        if N1 - 1 != H_t * W_t:
            raise RuntimeError(
                f"patch grid mismatch: {N1-1} tokens != H_t*W_t={H_t*W_t}"
            )
        dense = dense_with_pose[:, 1:, :].float()           # drop pose token
        t_rec = dense.view(B, nimgs, H_t, W_t, D).contiguous()

        p_rec_global = post["pts3d"].float()                # (B, nimgs, H_p, W_p, 3)
        p_rec_local = post.get("pts3d_local", None)
        if p_rec_local is not None:
            p_rec_local = p_rec_local.float()               # (B, nimgs, H_p, W_p, 3)
        c_rec = post["conf"].float()                        # (B, nimgs, H_p, W_p)

        if self.freeze:
            t_rec = t_rec.detach()
            p_rec_global = p_rec_global.detach()
            if p_rec_local is not None:
                p_rec_local = p_rec_local.detach()
            c_rec = c_rec.detach()

        return dict(
            t_rec=t_rec,
            p_rec_global=p_rec_global,
            p_rec_local=p_rec_local,
            c_rec=c_rec,
        )


class Stage1LiftingModule(nn.Module):
    """Pixel-aligned, confidence-weighted scatter from OccAny tokens to voxels."""

    def __init__(
        self,
        token_dim: int = 768,
        c_lift: int = 64,
        patch_size: int = 16,
        voxel_origin: Tuple[float, float, float] = (0.0, -25.6, -2.0),
        voxel_size: Tuple[float, float, float] = (0.2, 0.2, 0.2),
        grid_size: Tuple[int, int, int] = (256, 256, 32),
        hidden_dim: int = 256,
        conf_clamp_max: float = 50.0,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.c_lift = int(c_lift)
        self.patch_size = int(patch_size)
        self.conf_clamp_max = float(conf_clamp_max)
        self._scale_sanity_logged = False
        self.register_buffer(
            "voxel_origin",
            torch.tensor(voxel_origin, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "voxel_size",
            torch.tensor(voxel_size, dtype=torch.float32),
            persistent=False,
        )
        self.grid_size = tuple(int(v) for v in grid_size)

        coord_dim = 2
        self.pixel_mlp = nn.Sequential(
            nn.Linear(self.token_dim + coord_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.c_lift),
        )

    @staticmethod
    def _make_pixel_coords(H: int, W: int, device, dtype) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx, gy], dim=-1)  # (H, W, 2)

    def forward(
        self,
        t_rec: torch.Tensor,            # (B, N, H_t, W_t, D)
        p_rec_global: torch.Tensor,     # (B, N, H_p, W_p, 3)
        c_rec: torch.Tensor,            # (B, N, H_p, W_p)
        T_target_from_refcam: torch.Tensor,  # (B, 4, 4)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, H_t, W_t, D = t_rec.shape
        _, _, H_p, W_p, _ = p_rec_global.shape
        device = t_rec.device
        dtype = t_rec.dtype

        if H_p != H_t * self.patch_size or W_p != W_t * self.patch_size:
            raise RuntimeError(
                f"patch/pixel shape mismatch: H_p={H_p}, H_t*patch={H_t*self.patch_size}; "
                f"W_p={W_p}, W_t*patch={W_t*self.patch_size}"
            )

        # 1) Nearest-neighbor upsample T_rec to pixel resolution.
        t_up = (
            t_rec.repeat_interleave(self.patch_size, dim=2)
                 .repeat_interleave(self.patch_size, dim=3)
        )  # (B, N, H_p, W_p, D)

        # 2) Pixel coordinate encoding shared across (B, N).
        pix_coord = self._make_pixel_coords(H_p, W_p, device=device, dtype=dtype)
        pix_coord = pix_coord.view(1, 1, H_p, W_p, 2).expand(B, N, H_p, W_p, 2)

        # 3) MLP -> pixel features.
        x_rec = torch.cat([t_up, pix_coord], dim=-1)            # (B, N, H_p, W_p, D+2)
        f_rec = self.pixel_mlp(x_rec)                           # (B, N, H_p, W_p, C_lift)

        # 4) Transform pointmap into target (voxel) coords.
        T = T_target_from_refcam.to(device=device, dtype=p_rec_global.dtype)  # (B, 4, 4)
        R = T[:, :3, :3]                                        # (B, 3, 3)
        t = T[:, :3, 3]                                         # (B, 3)
        p_flat = p_rec_global.view(B, N * H_p * W_p, 3)         # (B, M, 3)
        p_target = torch.einsum("bij,bmj->bmi", R, p_flat) + t.unsqueeze(1)
        p_target = p_target.view(B, N, H_p, W_p, 3)

        # 5) Voxel indices.
        origin = self.voxel_origin.view(1, 1, 1, 1, 3).to(p_target.dtype)
        vs = self.voxel_size.view(1, 1, 1, 1, 3).to(p_target.dtype)
        idx = torch.floor((p_target - origin) / vs).long()      # (B, N, H_p, W_p, 3)
        X, Y, Z = self.grid_size

        in_bounds = (
            (idx[..., 0] >= 0) & (idx[..., 0] < X)
            & (idx[..., 1] >= 0) & (idx[..., 1] < Y)
            & (idx[..., 2] >= 0) & (idx[..., 2] < Z)
        )
        finite = torch.isfinite(p_target).all(dim=-1) & torch.isfinite(c_rec) & (c_rec > 0)
        mask = in_bounds & finite                               # (B, N, H_p, W_p)

        # Sanity: print p_target range + in-bounds rate once. If the OccAny
        # checkpoint isn't metric, this will reveal it immediately (e.g.
        # X-range nowhere near [0, 51.2] m or in_rate ~ 0).
        if not self._scale_sanity_logged:
            with torch.no_grad():
                finite_pts = p_target[finite]
                if finite_pts.numel() > 0:
                    pmin = finite_pts.min(dim=0).values
                    pmax = finite_pts.max(dim=0).values
                    in_rate = mask.float().mean().item()
                    print(
                        "[Stage1LiftingModule sanity] p_target range "
                        f"x=[{pmin[0]:.2f},{pmax[0]:.2f}] "
                        f"y=[{pmin[1]:.2f},{pmax[1]:.2f}] "
                        f"z=[{pmin[2]:.2f},{pmax[2]:.2f}] m, "
                        f"in-bounds rate={in_rate:.3f}. Expected (KITTI): "
                        "x~[0,51.2], y~[-25.6,25.6], z~[-2,4.4]."
                    )
            self._scale_sanity_logged = True

        # Batch index per pixel (broadcasted).
        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, N, H_p, W_p)

        i = idx[..., 0].clamp(0, X - 1)
        j = idx[..., 1].clamp(0, Y - 1)
        k = idx[..., 2].clamp(0, Z - 1)
        lin = (((batch_idx * X + i) * Y + j) * Z + k)            # (B, N, H_p, W_p)
        lin = lin[mask]                                          # (M_valid,)
        # Clamp confidence to avoid a single outlier dominating sum_w.
        w_raw = c_rec[mask].to(f_rec.dtype)
        w = w_raw.clamp(max=self.conf_clamp_max)                 # (M_valid,)
        f = f_rec[mask]                                          # (M_valid, C_lift)

        n_voxels = B * X * Y * Z
        sum_wf = torch.zeros(n_voxels, self.c_lift, device=device, dtype=f_rec.dtype)
        sum_w = torch.zeros(n_voxels, device=device, dtype=f_rec.dtype)
        if lin.numel() > 0:
            sum_wf.index_add_(0, lin, f * w.unsqueeze(-1))
            sum_w.index_add_(0, lin, w)

        eps = 1e-6
        V_rec = (sum_wf / (sum_w.unsqueeze(-1) + eps)).view(B, X, Y, Z, self.c_lift)
        V_rec = V_rec.permute(0, 4, 1, 2, 3).contiguous()        # (B, C_lift, X, Y, Z)
        # Compress dynamic range of the raw weight sum before feeding conv.
        W_rec = torch.log1p(sum_w).view(B, 1, X, Y, Z)           # (B, 1, X, Y, Z)
        return V_rec, W_rec
