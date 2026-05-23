"""Tiny offline sanity check for the LiDAR fusion stack (post-decoder variant).

Verifies (on a toy synthetic batch, CPU or single GPU):
  - Forward shapes line up through OccAny backbone (frozen) → fusion (VFE +
    windowed cross-attn on t_rec) → lifting → MonoScene head.
  - A backward pass populates grads in VFE, attention, lifting, and occ_head;
    OccAny encoder/decoder receive NO grad (they are frozen and wrapped in
    @torch.no_grad).

Run:
    python -m ft.kitti_stage1_5f.tools.sanity_lidar_fusion
"""
from __future__ import annotations

from .. import _paths  # noqa: F401

import sys

import numpy as np
import torch

from ..models import Stage1SSCMonoLidarModel


def _toy_views(B: int, N: int, H: int, W: int, device: torch.device):
    views = []
    for k in range(N):
        views.append(
            dict(
                img=torch.randn(B, 3, H, W, device=device),
                true_shape=torch.tensor([[H, W]] * B, dtype=torch.int32, device=device),
                camera_pose=torch.eye(4, device=device).unsqueeze(0).expand(B, 4, 4).contiguous(),
                camera_intrinsics=torch.tensor(
                    [[707.0, 0.0, 256.0], [0.0, 707.0, 80.0], [0.0, 0.0, 1.0]],
                    device=device,
                ).unsqueeze(0).expand(B, 3, 3).contiguous(),
                cam2world=torch.eye(4, device=device).unsqueeze(0).expand(B, 4, 4).contiguous(),
                timestep=torch.tensor([k] * B, device=device),
                is_raymap=torch.tensor([False] * B, device=device),
                is_metric_scale=torch.tensor([True] * B, device=device),
            )
        )
    return views


def _toy_lidar(B: int, N: int, device: torch.device):
    points_per_frame = []
    for b in range(B):
        per_frame = []
        for _ in range(N):
            P = 5000
            xyz = torch.empty(P, 3).uniform_(-30, 30)
            xyz[:, 0].uniform_(0, 50)
            xyz[:, 2].uniform_(-2, 4)
            intensity = torch.rand(P, 1)
            pts = torch.cat([xyz, intensity], dim=-1).to(device)
            per_frame.append(pts)
        points_per_frame.append(per_frame)
    R = torch.tensor(
        [[0.0, -1.0, 0.0],
         [0.0, 0.0, -1.0],
         [1.0, 0.0, 0.0]],
        device=device,
    )
    T = torch.eye(4, device=device).unsqueeze(0).expand(B, 4, 4).contiguous()
    T[:, :3, :3] = R
    K = torch.tensor(
        [[707.0, 0.0, 256.0], [0.0, 707.0, 80.0], [0.0, 0.0, 1.0]], device=device
    )
    K_per_frame = K.unsqueeze(0).unsqueeze(0).expand(B, N, 3, 3).contiguous()
    image_hw = torch.tensor([[160, 512]] * B, dtype=torch.int32, device=device)
    return points_per_frame, T, K_per_frame, image_hw


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)

    B, N = 1, 5
    print("[1] Building Stage1SSCMonoLidarModel (no ckpt, random init)...")
    model = Stage1SSCMonoLidarModel(
        occany_ckpt=None,
        backbone_img_size=(160, 512),
        backbone_dtype=torch.float32,
    ).to(device)
    # Freeze the OccAny backbone exactly as train.py does.
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.train()

    pts, T_cv, K_pf, hw = _toy_lidar(B, N, device)
    T_target_from_refcam = torch.eye(4, device=device).unsqueeze(0).expand(B, 4, 4).contiguous()
    views = _toy_views(B, N, H=160, W=512, device=device)

    print("[2] Forward + backward...")
    out = model(
        views,
        T_target_from_refcam=T_target_from_refcam,
        points_per_frame=pts,
        T_cam_from_velo=T_cv,
        K_per_frame=K_pf,
        image_hw=hw,
    )
    logits = out["ssc_logit"] if isinstance(out, dict) else out
    print(f"    logits shape: {tuple(logits.shape)}")
    loss = logits.float().square().mean()
    loss.backward()

    expected_grad = {
        "vfe.point_mlp": list(model.fusion.vfe.point_mlp.parameters()),
        "vfe.voxel_proj": list(model.fusion.vfe.voxel_proj.parameters()),
        "layer_w.q_proj": list(model.fusion.layer_w.q_proj.parameters()),
        "layer_w.out_proj": list(model.fusion.layer_w.out_proj.parameters()),
        "layer_sw.q_proj": list(model.fusion.layer_sw.q_proj.parameters()),
        "lifting": list(model.lifting.parameters()),
        "occ_head.first": list(model.occ_head.parameters())[:1],
    }
    expected_no_grad = {
        "encoder.first": list(model.backbone.encoder.parameters())[:1],
        "decoder.first": list(model.backbone.decoder.parameters())[:1],
    }

    ok = True
    for name, params in expected_grad.items():
        has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in params)
        print(f"    {name}: grad={'yes' if has_grad else 'NO'}")
        if not has_grad:
            ok = False
    for name, params in expected_no_grad.items():
        any_grad = any(p.grad is not None for p in params)
        any_req = any(p.requires_grad for p in params)
        print(f"    {name}: requires_grad={'yes' if any_req else 'no'} grad_present={'yes' if any_grad else 'no'}")
        if any_req or any_grad:
            ok = False
    if not ok:
        sys.exit("Sanity check failed; see log above.")
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
