"""OccAny token lifter with optional novel-view render tokens."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ft.semantickitti_ft.interfaces import LiftedFeatures
from ft.semantickitti_ft.lifting.registry import register_lifter
from ft.semantickitti_ft.lifting.render_poses import generate_render_poses
from ft.semantickitti_ft.lifting.token_capture import (
    DecoderTokenCapturer,
    drop_pose_token,
)
from occany.model.model_must3r import (
    Dust3rEncoder,
    Must3rDecoder,
    RaymapEncoderDiT,
)
from occany.must3r_inference import (
    create_gen_conditioning,
    inference_encoder,
    inference_encoder_raymap,
    inference_img_online,
    inference_render,
    postprocess,
    prepare_imgs_or_raymaps_and_true_shape_mem_batches,
)
from occany.model.must3r_blocks.head import ActivationType


@register_lifter("occany_render_tokens")
class OccAnyRenderTokenLifter(nn.Module):
    """Extract Multi-view OccAny tokens and package them for an SSC head.

    The current implementation matches the previous experiment behavior:
    last-frame stereo recon tokens are always used, and K novel render-token
    views are appended when raymap/gen modules are available.
    """

    def __init__(
        self,
        img_encoder: Dust3rEncoder,
        decoder: Must3rDecoder,
        raymap_encoder: Optional[RaymapEncoderDiT],
        gen_decoder: Optional[Must3rDecoder],
        n_render_views: int = 4,
        n_decoder_feature_layers: int = 4,
        last_frame_view_indices: Tuple[int, int] = (4, 5),
        pointmaps_activation: ActivationType = ActivationType.LINEAR,
        backbone_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.img_encoder = img_encoder
        self.decoder = decoder
        self.raymap_encoder = raymap_encoder
        self.gen_decoder = gen_decoder
        self.use_render = (raymap_encoder is not None) and (gen_decoder is not None)

        self.n_render_views = int(n_render_views) if self.use_render else 0
        self.n_decoder_feature_layers = int(n_decoder_feature_layers)
        self.last_frame_view_indices = tuple(int(i) for i in last_frame_view_indices)
        self.pointmaps_activation = pointmaps_activation
        self.backbone_dtype = backbone_dtype

        self._freeze_backbone()
        self._capturer = DecoderTokenCapturer(self.n_decoder_feature_layers)

    def _freeze_backbone(self) -> None:
        for m in (self.img_encoder, self.decoder):
            for p in m.parameters():
                p.requires_grad = False
            m.eval()
        if self.raymap_encoder is not None:
            for p in self.raymap_encoder.parameters():
                p.requires_grad = False
            self.raymap_encoder.eval()
        if self.gen_decoder is not None:
            for p in self.gen_decoder.parameters():
                p.requires_grad = False
            self.gen_decoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.img_encoder.eval()
        self.decoder.eval()
        if self.raymap_encoder is not None:
            self.raymap_encoder.eval()
        if self.gen_decoder is not None:
            self.gen_decoder.eval()
        return self

    @torch.no_grad()
    def _encode_views(
        self,
        views: List[Dict[str, torch.Tensor]],
        device: torch.device,
    ):
        imgs, true_shape_img, mem_batches, img_timesteps = (
            prepare_imgs_or_raymaps_and_true_shape_mem_batches(
                views, device, is_raymap=False
            )
        )
        B, nimgs = imgs.shape[:2]
        x_img, pos_img = inference_encoder(
            encoder=self.img_encoder,
            imgs=imgs,
            true_shape_view=true_shape_img.view(B * nimgs, 2),
            max_bs=None,
            requires_grad=False,
        )
        return x_img, pos_img, true_shape_img, mem_batches, img_timesteps

    @torch.no_grad()
    def _run_recon_decoder(
        self,
        x_img: torch.Tensor,
        pos_img: torch.Tensor,
        true_shape_img: torch.Tensor,
        mem_batches: List[int],
    ):
        _img_out_0, _pose_out_0, _sam_feats_0, mem = inference_img_online(
            decoder=self.decoder,
            x=x_img,
            pos=pos_img,
            true_shape=true_shape_img,
            mem_batches=mem_batches,
            verbose=False,
        )
        return mem

    @torch.no_grad()
    def _render_last_frame_recon(
        self,
        x_img: torch.Tensor,
        pos_img: torch.Tensor,
        true_shape_img: torch.Tensor,
        mem,
        views: List[Dict[str, torch.Tensor]],
    ):
        last_idx = list(self.last_frame_view_indices)
        x_last = x_img[:, last_idx].contiguous()
        pos_last = pos_img[:, last_idx].contiguous()
        ts_last = true_shape_img[:, last_idx].contiguous()

        self._capturer.attach(self.decoder)
        try:
            _, pointmaps_raw, pose_out_raw, sam_feats_last = inference_render(
                decoder=self.decoder,
                x=x_last,
                pos=pos_last,
                true_shape=ts_last,
                mem=mem,
                freeze_decoder=True,
                verbose=False,
            )
            recon_feats = self._capturer.pop()
        finally:
            self._capturer.detach()

        post = postprocess(
            pointmaps_raw,
            pose_out_raw,
            pointmaps_activation=self.pointmaps_activation,
            compute_cam=True,
        )
        pts3d = post["pts3d"]
        conf = post["conf"]
        focal = post["focal"].mean(dim=1)
        c2w_pred = post.get("c2w_pose", post["c2w"])
        B, nimgs_last, H, W = pts3d.shape[:4]

        rgb = torch.stack([views[i]["img"] for i in last_idx], dim=1).to(pts3d.device)
        rgb = rgb.permute(0, 1, 3, 4, 2)

        if sam_feats_last is not None:
            sam_feats_last = sam_feats_last[:3]
            sam_feats_resized = []
            for sf in sam_feats_last:
                sf2 = F.interpolate(
                    sf.reshape(B * nimgs_last, -1, sf.shape[3], sf.shape[4]),
                    (H, W),
                    mode="bilinear",
                    align_corners=False,
                )
                sf2 = sf2.reshape(B, nimgs_last, -1, H, W).permute(0, 1, 3, 4, 2)
                sam_feats_resized.append(sf2)
        else:
            sam_feats_resized = [
                torch.zeros(B, nimgs_last, H, W, c, device=pts3d.device, dtype=pts3d.dtype)
                for c in (256, 64, 32)
            ]

        proj_feats = (
            getattr(
                self.raymap_encoder,
                "projection_features",
                ["pts3d_local", "pts3d", "rgb", "conf", "sam"],
            )
            if self.raymap_encoder is not None
            else []
        )
        feats_list = []
        if "pts3d" in proj_feats:
            feats_list.append(pts3d)
        if "rgb" in proj_feats:
            feats_list.append(rgb)
        if "conf" in proj_feats:
            feats_list.append(conf.unsqueeze(-1) - 1.0)
        if "sam" in proj_feats or "sam3" in proj_feats:
            feats_list.extend(sam_feats_resized)
        if len(feats_list) > 0:
            pts_features = torch.cat(feats_list, dim=-1)
        else:
            pts_features = torch.zeros(
                B, nimgs_last, H, W, 0, device=pts3d.device, dtype=pts3d.dtype
            )

        return recon_feats, nimgs_last, pts3d, focal, pts_features, c2w_pred

    @torch.no_grad()
    def _render_novel_views(
        self,
        x_img: torch.Tensor,
        pos_img: torch.Tensor,
        true_shape_img: torch.Tensor,
        mem,
        img_timesteps: torch.Tensor,
        novel_c2w: torch.Tensor,
        last_frame_recon_pts3d: torch.Tensor,
        last_frame_focal: torch.Tensor,
        last_frame_pts_features: torch.Tensor,
    ):
        if not self.use_render:
            return [], 0

        B, K = novel_c2w.shape[:2]
        device = x_img.device
        H = int(true_shape_img[0, 0, 0].item())
        W = int(true_shape_img[0, 0, 1].item())

        cond_features = create_gen_conditioning(
            last_frame_recon_pts3d,
            last_frame_pts_features,
            last_frame_focal,
            novel_c2w,
            raymap_views=None,
            use_raymap_only_conditioning=False,
            projection_features=getattr(
                self.raymap_encoder,
                "projection_features",
                ["pts3d_local", "pts3d", "rgb", "conf", "sam"],
            ),
        )
        raymaps = cond_features.permute(0, 1, 4, 2, 3).contiguous()

        true_shape_render = torch.full((B, K, 2), 0, dtype=torch.int32, device=device)
        true_shape_render[..., 0] = H
        true_shape_render[..., 1] = W

        timesteps_render = torch.zeros(
            (B, K), dtype=img_timesteps.dtype, device=img_timesteps.device
        )

        x_ray, pos_ray = inference_encoder_raymap(
            encoder=self.raymap_encoder,
            raymaps=raymaps,
            true_shape_view=true_shape_render.view(B * K, 2),
            max_bs=None,
            requires_grad=False,
            mem=x_img,
            mem_pos=pos_img,
            mem_timesteps=img_timesteps,
            timesteps=timesteps_render,
        )

        self._capturer.attach(self.gen_decoder)
        try:
            _, _pm, _pose, _sam = inference_render(
                decoder=self.gen_decoder,
                x=x_ray,
                pos=pos_ray,
                true_shape=true_shape_render,
                mem=mem,
                freeze_decoder=True,
                verbose=False,
            )
            feats = self._capturer.pop()
        finally:
            self._capturer.detach()
        return feats, K

    def forward(
        self,
        views: List[Dict[str, torch.Tensor]],
        anchor_pose: torch.Tensor,
        lidar_to_world: torch.Tensor,
    ) -> LiftedFeatures:
        device = views[0]["img"].device
        B = views[0]["img"].shape[0]

        with torch.no_grad():
            with torch.autocast("cuda", dtype=self.backbone_dtype):
                x_img, pos_img, true_shape_img, mem_batches, img_timesteps = (
                    self._encode_views(views, device)
                )
                mem = self._run_recon_decoder(
                    x_img, pos_img, true_shape_img, mem_batches
                )
                (
                    recon_feats_capt,
                    n_recon,
                    pts3d_last,
                    focal_last,
                    pts_feats_last,
                    c2w_pred_last,
                ) = self._render_last_frame_recon(
                    x_img, pos_img, true_shape_img, mem, views
                )

                if self.use_render:
                    target_left_c2w_net = c2w_pred_last[:, 0].to(pts3d_last.dtype)
                    novel_c2w = generate_render_poses(
                        target_left_c2w_net, n_views=self.n_render_views
                    )
                    render_feats_capt, n_render = self._render_novel_views(
                        x_img,
                        pos_img,
                        true_shape_img,
                        mem,
                        img_timesteps,
                        novel_c2w,
                        pts3d_last,
                        focal_last,
                        pts_feats_last,
                    )
                else:
                    render_feats_capt = []
                    n_render = 0

        if len(recon_feats_capt) != self.n_decoder_feature_layers:
            raise RuntimeError(
                f"expected {self.n_decoder_feature_layers} recon feats, got "
                f"{len(recon_feats_capt)}"
            )

        aggregated_layers: List[torch.Tensor] = []
        for layer_idx in range(self.n_decoder_feature_layers):
            recon_t = drop_pose_token(recon_feats_capt[layer_idx], B, n_recon)
            if n_render > 0 and len(render_feats_capt) == self.n_decoder_feature_layers:
                render_t = drop_pose_token(render_feats_capt[layer_idx], B, n_render)
                merged = torch.cat([recon_t, render_t], dim=1)
            else:
                merged = recon_t
            aggregated_layers.append(merged.float())

        last_idx = list(self.last_frame_view_indices)
        recon_imgs = torch.stack([views[i]["img"] for i in last_idx], dim=1)
        recon_intr = torch.stack(
            [views[i]["camera_intrinsics"] for i in last_idx], dim=1
        ).float()
        recon_c2w = torch.stack(
            [views[i]["camera_pose"] for i in last_idx], dim=1
        ).float()

        if n_render > 0:
            render_imgs = torch.zeros(
                B,
                n_render,
                recon_imgs.shape[2],
                recon_imgs.shape[3],
                recon_imgs.shape[4],
                device=device,
                dtype=recon_imgs.dtype,
            )
            render_intr = recon_intr[:, :1].expand(-1, n_render, -1, -1).contiguous()
            render_c2w = generate_render_poses(
                torch.eye(4, device=device, dtype=recon_c2w.dtype)
                .unsqueeze(0)
                .expand(B, -1, -1),
                n_views=n_render,
            )
            images = torch.cat([recon_imgs, render_imgs], dim=1)
            intrinsics = torch.cat([recon_intr, render_intr], dim=1)
            camera_to_world = torch.cat([recon_c2w, render_c2w], dim=1)
        else:
            images = recon_imgs
            intrinsics = recon_intr
            camera_to_world = recon_c2w

        return LiftedFeatures(
            aggregated_tokens_list=aggregated_layers,
            images=images,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            lidar_to_world=lidar_to_world,
        )

