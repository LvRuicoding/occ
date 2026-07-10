"""Evaluate KITTI Stage-1 pointmap quality.

Example:
  python -m ft.kitti_stage1_5f.tools.eval_pointmap_quality \
    --ckpt /path/to/output/checkpoint-last.pth \
    --batch_size 1

Distributed example:
  torchrun --standalone --nproc_per_node=4 \
    -m ft.kitti_stage1_5f.tools.eval_pointmap_quality \
    --ckpt /path/to/output/checkpoint-last.pth \
    --batch_size 1

The script evaluates the BEVDet-OCC LiDAR pointmap head against the dense
depth maps used for pointmap supervision. Predicted ``pointmap_pts3d`` is
assumed to be in the reference camera frame, matching
``_pointmap_targets_from_depth`` in the pointmap training loss.
"""
from __future__ import annotations

try:
    from .. import _paths  # noqa: F401  (must run before project imports)
except ImportError:  # Allows direct `python ft/.../eval_pointmap_quality.py`.
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from ft.kitti_stage1_5f import _paths  # noqa: F401

import argparse
import copy
import datetime as _datetime
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import Subset

import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc
from occany.model.must3r_blocks.attention import toggle_memory_efficient_attention
from occany.utils.checkpoint_io import register_legacy_checkpoint_modules

from ft.kitti_stage1_5f.models import (
    Stage1PointmapOriginalModel,
    Stage1PointmapPostFusionOnlyModel,
    Stage1SSCBEVDetOccLidarPointmapDenseDepthModel,
    Stage1SSCBEVDetOccLidarPointmapModel,
)
from ft.kitti_stage1_5f.models.stage1_ssc_bevdetocc_lidar_pointmap import (
    _pointmap_targets_from_depth,
)
from ft.kitti_stage1_5f.tools.train import (
    _build_dataset,
    _build_loader,
    _model_forward,
    _stack_cam2world_from_views,
    _state_dict_hash,
)


POINTMAP_EVAL_EXPS = (
    "bevdetocc_lidar_pointmap",
    "pointmap_postfusion_only",
    "pointmap_original",
    "bevdetocc_lidar_pointmap_dense_depth",
)
POINTMAP_LIDAR_EXPS = (
    "bevdetocc_lidar_pointmap",
    "pointmap_postfusion_only",
    "bevdetocc_lidar_pointmap_dense_depth",
)


def get_args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Evaluate Stage-1 pointmap quality", add_help=True)
    p.add_argument("--ckpt", required=True, type=str,
                   help="Checkpoint file, or a directory containing checkpoint-last.pth.")
    p.add_argument("--processed_root", default=None, type=str)
    p.add_argument("--velodyne_root", default=None, type=str)
    p.add_argument("--occany_ckpt", default=None, type=str)

    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--num_frames", type=int, default=None)
    p.add_argument("--frame_stride", type=int, default=None)
    p.add_argument("--c_lift", type=int, default=None)
    p.add_argument("--token_dim", type=int, default=None)
    p.add_argument("--patch_size", type=int, default=None)
    p.add_argument("--max_points_per_sweep", type=int, default=None)
    p.add_argument("--dense_depth_features", type=int, default=None)
    p.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=None)

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--amp", choices=["bf16", "fp16", "none"], default=None)
    p.add_argument("--device", default="auto", type=str)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--print_freq", type=int, default=20)
    p.add_argument("--max_batches", type=int, default=0,
                   help="If >0, stop after this many validation batches.")
    p.add_argument("--world_size", default=1, type=int)
    p.add_argument("--local_rank", default=-1, type=int)
    p.add_argument("--dist_url", default="env://", type=str)
    p.add_argument("--nodist", action="store_true",
                   help="Disable distributed mode even if torchrun environment is present.")

    p.add_argument("--chamfer_max_points", type=int, default=8192,
                   help="Max points sampled per sample/frame for Chamfer/F-score.")
    p.add_argument("--chamfer_chunk_size", type=int, default=2048)
    p.add_argument("--fscore_threshold", type=float, default=0.2,
                   help="Distance threshold in meters for point-cloud F-score.")
    p.add_argument("--cross_view_pairs", choices=["adjacent", "all", "none"], default="adjacent")
    p.add_argument("--cross_view_max_points", type=int, default=4096,
                   help="Max source points sampled per ordered view pair.")
    p.add_argument("--stat_sample_max_points", type=int, default=2_000_000,
                   help="Reservoir size for median/quantile-like statistics.")
    p.add_argument("--confidence_max_points", type=int, default=2_000_000,
                   help="Reservoir size for confidence AUC.")
    p.add_argument("--confidence_good_threshold", type=float, default=None,
                   help="Good-point threshold for confidence ROC AUC. Defaults to --fscore_threshold.")
    p.add_argument("--output_json", default=None, type=str,
                   help="Optional path for a JSON metrics file.")
    return p


def _resolve_ckpt_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_dir():
        path = path / "checkpoint-last.pth"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _ckpt_arg(ckpt_args, name: str, default):
    if isinstance(ckpt_args, dict):
        return ckpt_args.get(name, default)
    return getattr(ckpt_args, name, default)


def _override_or_ckpt(args: argparse.Namespace, ckpt_args, name: str, default):
    value = getattr(args, name)
    return value if value is not None else _ckpt_arg(ckpt_args, name, default)


def _fill_args_from_checkpoint(args: argparse.Namespace, ckpt_args) -> None:
    args.exp = _ckpt_arg(ckpt_args, "exp", "bevdetocc_lidar_pointmap")
    if args.exp not in POINTMAP_EVAL_EXPS:
        raise ValueError(
            "eval_pointmap_quality expects a checkpoint trained with "
            f"one of {POINTMAP_EVAL_EXPS}; checkpoint exp={args.exp!r}."
        )

    args.processed_root = _override_or_ckpt(args, ckpt_args, "processed_root", None)
    args.velodyne_root = _override_or_ckpt(args, ckpt_args, "velodyne_root", None)
    args.occany_ckpt = _override_or_ckpt(args, ckpt_args, "occany_ckpt", None)
    args.width = int(_override_or_ckpt(args, ckpt_args, "width", 512))
    args.height = int(_override_or_ckpt(args, ckpt_args, "height", 160))
    args.num_frames = int(_override_or_ckpt(args, ckpt_args, "num_frames", 5))
    args.frame_stride = int(_override_or_ckpt(args, ckpt_args, "frame_stride", 4))
    args.c_lift = int(_override_or_ckpt(args, ckpt_args, "c_lift", 64))
    args.token_dim = int(_override_or_ckpt(args, ckpt_args, "token_dim", 768))
    args.patch_size = int(_override_or_ckpt(args, ckpt_args, "patch_size", 16))
    args.max_points_per_sweep = int(
        _override_or_ckpt(args, ckpt_args, "max_points_per_sweep", 0)
    )
    args.dense_depth_features = int(
        _override_or_ckpt(args, ckpt_args, "dense_depth_features", 128)
    )
    args.freeze_backbone = bool(
        _override_or_ckpt(args, ckpt_args, "freeze_backbone", False)
    )
    args.num_workers = int(_override_or_ckpt(args, ckpt_args, "num_workers", 4))
    args.amp = args.amp or _ckpt_arg(ckpt_args, "amp", "bf16")

    # These are consumed by train.py helpers.
    args.depth_supervision = False
    args.dense_depth_supervision = False
    args.pointmap_supervision = False
    if not hasattr(args, "distributed"):
        args.distributed = False

    if not args.processed_root:
        raise ValueError("--processed_root is required when checkpoint args do not contain it.")
    if args.exp in POINTMAP_LIDAR_EXPS and not args.velodyne_root:
        raise ValueError(f"--velodyne_root is required for {args.exp}.")
    if not args.occany_ckpt:
        raise ValueError("--occany_ckpt is required when checkpoint args do not contain it.")


def _build_pointmap_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if args.amp == "bf16" and device.type == "cuda":
        backbone_dtype = torch.bfloat16
    elif args.amp == "fp16" and device.type == "cuda":
        backbone_dtype = torch.float16
    else:
        backbone_dtype = torch.float32

    if args.exp == "pointmap_original":
        model_cls = Stage1PointmapOriginalModel
    elif args.exp == "pointmap_postfusion_only":
        model_cls = Stage1PointmapPostFusionOnlyModel
    elif args.exp == "bevdetocc_lidar_pointmap_dense_depth":
        model_cls = Stage1SSCBEVDetOccLidarPointmapDenseDepthModel
    else:
        model_cls = Stage1SSCBEVDetOccLidarPointmapModel

    model_kwargs = dict(
        occany_ckpt=args.occany_ckpt,
        c_lift=args.c_lift,
        num_classes=20,
        patch_size=args.patch_size,
        token_dim=args.token_dim,
        backbone_img_size=(args.height, args.width),
        backbone_dtype=backbone_dtype,
        num_frames=args.num_frames,
        freeze_backbone=args.freeze_backbone,
    )
    if args.exp in POINTMAP_LIDAR_EXPS:
        model_kwargs["fusion_attn_type"] = "cross"
    if args.exp == "bevdetocc_lidar_pointmap_dense_depth":
        model_kwargs["dense_depth_features"] = args.dense_depth_features
    model = model_cls(**model_kwargs).to(device)
    return model


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state):
        return state
    return {k.removeprefix("module."): v for k, v in state.items()}


def _load_pointmap_weights(model: torch.nn.Module, ckpt: Dict) -> Tuple[List[str], List[str]]:
    if "model" not in ckpt:
        raise KeyError("BEVDet-OCC pointmap checkpoints must contain a 'model' state_dict.")
    state = _strip_module_prefix(ckpt["model"])
    status = model.load_state_dict(state, strict=False)
    critical_missing = [
        k for k in status.missing_keys
        if not k.startswith("backbone.")
        and not k.endswith("num_batches_tracked")
    ]
    if critical_missing:
        preview = ", ".join(critical_missing[:10])
        raise RuntimeError(
            f"Checkpoint is missing non-backbone model keys ({len(critical_missing)}): {preview}"
        )
    return list(status.missing_keys), list(status.unexpected_keys)


def _amp_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


class Reservoir:
    """Approximate streaming sample buffer for medians/curves."""

    def __init__(self, max_size: int, seed: int = 0) -> None:
        self.max_size = int(max_size)
        self.values: Optional[torch.Tensor] = None
        self.seen = 0
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(int(seed))

    def add(self, values: torch.Tensor) -> None:
        if self.max_size <= 0:
            self.seen += int(values.numel())
            return
        v = values.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        n = int(v.numel())
        if n == 0:
            return
        if self.values is None:
            take = min(self.max_size, n)
            if take < n:
                idx = torch.randperm(n, generator=self.gen)[:take]
                self.values = v[idx].clone()
            else:
                self.values = v.clone()
            self.seen += n
            return

        cur = int(self.values.numel())
        if cur < self.max_size:
            take = min(self.max_size - cur, n)
            self.values = torch.cat([self.values, v[:take].clone()], dim=0)
            v = v[take:]
            n = int(v.numel())
            self.seen += take
            if n == 0:
                return

        positions = torch.arange(
            self.seen + 1,
            self.seen + n + 1,
            dtype=torch.float32,
        )
        keep = torch.rand(n, generator=self.gen) < (float(self.max_size) / positions)
        if bool(keep.any()):
            dst = torch.randint(self.max_size, (int(keep.sum()),), generator=self.gen)
            self.values[dst] = v[keep]
        self.seen += n

    def median(self) -> float:
        if self.values is None or self.values.numel() == 0:
            return float("nan")
        return float(torch.median(self.values).item())


class PairReservoir:
    """Reservoir for paired confidence/error samples."""

    def __init__(self, max_size: int, seed: int = 0) -> None:
        self.max_size = int(max_size)
        self.conf: Optional[torch.Tensor] = None
        self.err: Optional[torch.Tensor] = None
        self.seen = 0
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(int(seed))

    def add(self, conf: torch.Tensor, err: torch.Tensor) -> None:
        if self.max_size <= 0:
            self.seen += int(err.numel())
            return
        c = conf.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        e = err.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        valid = torch.isfinite(c) & torch.isfinite(e)
        c = c[valid]
        e = e[valid]
        n = int(e.numel())
        if n == 0:
            return
        if self.err is None:
            take = min(self.max_size, n)
            if take < n:
                idx = torch.randperm(n, generator=self.gen)[:take]
                c = c[idx]
                e = e[idx]
            self.conf = c[:take].clone()
            self.err = e[:take].clone()
            self.seen += n
            return

        cur = int(self.err.numel())
        if cur < self.max_size:
            take = min(self.max_size - cur, n)
            self.conf = torch.cat([self.conf, c[:take].clone()], dim=0)
            self.err = torch.cat([self.err, e[:take].clone()], dim=0)
            c = c[take:]
            e = e[take:]
            n = int(e.numel())
            self.seen += take
            if n == 0:
                return

        positions = torch.arange(
            self.seen + 1,
            self.seen + n + 1,
            dtype=torch.float32,
        )
        keep = torch.rand(n, generator=self.gen) < (float(self.max_size) / positions)
        if bool(keep.any()):
            dst = torch.randint(self.max_size, (int(keep.sum()),), generator=self.gen)
            self.conf[dst] = c[keep]
            self.err[dst] = e[keep]
        self.seen += n

    def aucs(self, good_threshold: float) -> Dict[str, float]:
        if self.conf is None or self.err is None or self.err.numel() < 2:
            return {}
        order = torch.argsort(self.conf, descending=True)
        err = self.err[order]
        coverage = torch.arange(1, err.numel() + 1, dtype=torch.float32) / float(err.numel())
        risk = torch.cumsum(err, dim=0) / torch.arange(1, err.numel() + 1, dtype=torch.float32)
        confidence_auc = float(torch.trapz(risk, coverage).item())

        labels = (self.err <= float(good_threshold)).to(torch.int64)
        n_pos = int(labels.sum().item())
        n_neg = int(labels.numel() - n_pos)
        roc_auc = float("nan")
        if n_pos > 0 and n_neg > 0:
            order_asc = torch.argsort(self.conf)
            ranks = torch.empty_like(order_asc, dtype=torch.float32)
            ranks[order_asc] = torch.arange(1, labels.numel() + 1, dtype=torch.float32)
            pos_rank_sum = ranks[labels.bool()].sum()
            roc_auc = float(((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)).item())

        return {
            "confidence_auc": confidence_auc,
            "confidence_roc_auc": roc_auc,
            "confidence_samples": float(self.err.numel()),
            "confidence_good_threshold": float(good_threshold),
        }


class PointmapMetricAccumulator:
    _STATE_SCALARS = (
        "count",
        "pts_l1_sum",
        "pts_l2_sum",
        "scale_count",
        "scale_l1_sum",
        "scale_l2_sum",
        "depth_count",
        "depth_absrel_sum",
        "depth_sq_sum",
        "depth_delta_sum",
        "reproj_count",
        "reproj_sum",
        "chamfer_f_sum",
        "chamfer_b_sum",
        "chamfer_f_count",
        "chamfer_b_count",
        "fscore_prec_hits",
        "fscore_rec_hits",
        "cross_count",
        "cross_sum",
    )

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.count = 0.0
        self.pts_l1_sum = 0.0
        self.pts_l2_sum = 0.0
        self.pts_l2_med = Reservoir(args.stat_sample_max_points, args.seed + 11)

        self.scale_count = 0.0
        self.scale_l1_sum = 0.0
        self.scale_l2_sum = 0.0
        self.scale_l2_med = Reservoir(args.stat_sample_max_points, args.seed + 12)

        self.depth_count = 0.0
        self.depth_absrel_sum = 0.0
        self.depth_sq_sum = 0.0
        self.depth_delta_sum = 0.0

        self.reproj_count = 0.0
        self.reproj_sum = 0.0
        self.reproj_med = Reservoir(args.stat_sample_max_points, args.seed + 13)

        self.chamfer_f_sum = 0.0
        self.chamfer_b_sum = 0.0
        self.chamfer_f_count = 0.0
        self.chamfer_b_count = 0.0
        self.fscore_prec_hits = 0.0
        self.fscore_rec_hits = 0.0

        self.cross_count = 0.0
        self.cross_sum = 0.0
        self.cross_med = Reservoir(args.stat_sample_max_points, args.seed + 14)

        self.conf_pairs = PairReservoir(args.confidence_max_points, args.seed + 15)

    def update_point_errors(
        self,
        pred_ref: torch.Tensor,
        gt_ref: torch.Tensor,
        valid: torch.Tensor,
        pred_conf: Optional[torch.Tensor],
    ) -> None:
        diff = pred_ref.float() - gt_ref.float()
        finite = torch.isfinite(diff).all(dim=-1)
        mask = valid & finite
        if not bool(mask.any().item()):
            return
        l1 = diff.abs().sum(dim=-1)[mask]
        l2 = torch.linalg.norm(diff, dim=-1)[mask]
        self.count += float(l2.numel())
        self.pts_l1_sum += float(l1.sum().item())
        self.pts_l2_sum += float(l2.sum().item())
        self.pts_l2_med.add(l2)
        if pred_conf is not None:
            self.conf_pairs.add(pred_conf[mask], l2)

    def update_scale_aligned(
        self,
        pred_ref: torch.Tensor,
        gt_ref: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        pred = pred_ref.float()
        gt = gt_ref.float()
        finite = torch.isfinite(pred).all(dim=-1) & torch.isfinite(gt).all(dim=-1)
        mask = valid & finite
        if not bool(mask.any().item()):
            return
        mask_f = mask.to(dtype=pred.dtype)
        dot = (pred * gt).sum(dim=-1)
        denom = (pred * pred).sum(dim=-1)
        # One scalar scale per sample across all frames and pixels. This keeps
        # cross-frame scale consistency meaningful while removing global scale drift.
        dims = tuple(range(1, dot.ndim))
        dot_sum = (dot * mask_f).sum(dim=dims, keepdim=True)
        denom_sum = (denom * mask_f).sum(dim=dims, keepdim=True).clamp(min=1e-8)
        scale = (dot_sum / denom_sum).clamp(min=1e-6, max=1e6)
        diff = pred * scale.unsqueeze(-1) - gt
        l1 = diff.abs().sum(dim=-1)[mask]
        l2 = torch.linalg.norm(diff, dim=-1)[mask]
        self.scale_count += float(l2.numel())
        self.scale_l1_sum += float(l1.sum().item())
        self.scale_l2_sum += float(l2.sum().item())
        self.scale_l2_med.add(l2)

    def update_depth(
        self,
        pred_local: torch.Tensor,
        dense_depth: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        pred_z = pred_local[..., 2].float()
        gt_z = dense_depth.float()
        mask = valid & torch.isfinite(pred_z) & torch.isfinite(gt_z) & (gt_z > 0.0) & (pred_z > 0.0)
        if not bool(mask.any().item()):
            return
        p = pred_z[mask]
        g = gt_z[mask]
        abs_rel = (p - g).abs() / g.clamp(min=1e-6)
        sq = (p - g).pow(2)
        ratio = torch.maximum(p / g.clamp(min=1e-6), g / p.clamp(min=1e-6))
        self.depth_count += float(p.numel())
        self.depth_absrel_sum += float(abs_rel.sum().item())
        self.depth_sq_sum += float(sq.sum().item())
        self.depth_delta_sum += float((ratio < 1.25).to(torch.float32).sum().item())

    def update_reprojection(
        self,
        pred_local: torch.Tensor,
        K_per_frame: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        B, N, H, W, _ = pred_local.shape
        device = pred_local.device
        ys = torch.arange(H, device=device, dtype=torch.float32)
        xs = torch.arange(W, device=device, dtype=torch.float32)
        v0, u0 = torch.meshgrid(ys, xs, indexing="ij")
        u0 = u0.view(1, 1, H, W)
        v0 = v0.view(1, 1, H, W)

        pts = pred_local.float()
        z = pts[..., 2]
        K = K_per_frame.to(device=device, dtype=torch.float32)
        fx = K[..., 0, 0].view(B, N, 1, 1)
        fy = K[..., 1, 1].view(B, N, 1, 1)
        cx = K[..., 0, 2].view(B, N, 1, 1)
        cy = K[..., 1, 2].view(B, N, 1, 1)
        u = pts[..., 0] / z.clamp(min=1e-6) * fx + cx
        v = pts[..., 1] / z.clamp(min=1e-6) * fy + cy
        err = torch.sqrt((u - u0).pow(2) + (v - v0).pow(2))
        mask = valid & torch.isfinite(err) & torch.isfinite(pts).all(dim=-1) & (z > 1e-6)
        if not bool(mask.any().item()):
            return
        vals = err[mask]
        self.reproj_count += float(vals.numel())
        self.reproj_sum += float(vals.sum().item())
        self.reproj_med.add(vals)

    def update_chamfer(
        self,
        pred_ref: torch.Tensor,
        gt_ref: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        max_points = int(self.args.chamfer_max_points)
        if max_points <= 0:
            return
        B, N = pred_ref.shape[:2]
        gen = torch.Generator(device=pred_ref.device)
        gen.manual_seed(int(self.args.seed) + int(self.count) % 1000003)
        for b in range(B):
            for n in range(N):
                mask = (
                    valid[b, n]
                    & torch.isfinite(pred_ref[b, n]).all(dim=-1)
                    & torch.isfinite(gt_ref[b, n]).all(dim=-1)
                )
                if not bool(mask.any().item()):
                    continue
                pred = pred_ref[b, n][mask].float()
                gt = gt_ref[b, n][mask].float()
                pred = _sample_points(pred, max_points, gen)
                gt = _sample_points(gt, max_points, gen)
                if pred.numel() == 0 or gt.numel() == 0:
                    continue
                d_pg = _nearest_distances(pred, gt, int(self.args.chamfer_chunk_size))
                d_gp = _nearest_distances(gt, pred, int(self.args.chamfer_chunk_size))
                thr = float(self.args.fscore_threshold)
                self.chamfer_f_sum += float(d_pg.sum().item())
                self.chamfer_b_sum += float(d_gp.sum().item())
                self.chamfer_f_count += float(d_pg.numel())
                self.chamfer_b_count += float(d_gp.numel())
                self.fscore_prec_hits += float((d_pg < thr).to(torch.float32).sum().item())
                self.fscore_rec_hits += float((d_gp < thr).to(torch.float32).sum().item())

    def update_cross_view(
        self,
        pred_ref: torch.Tensor,
        pred_local: torch.Tensor,
        K_per_frame: torch.Tensor,
        cam2world: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        if self.args.cross_view_pairs == "none":
            return
        B, N, H, W, _ = pred_ref.shape
        pairs = _view_pairs(N, self.args.cross_view_pairs)
        max_points = int(self.args.cross_view_max_points)
        if not pairs or max_points <= 0:
            return
        gen = torch.Generator(device=pred_ref.device)
        gen.manual_seed(int(self.args.seed) + 7919 + int(self.cross_count) % 1000003)

        T_ref_from_world = torch.linalg.inv(cam2world[:, 0].float())
        T_ref_from_cam = T_ref_from_world[:, None] @ cam2world.float()
        T_cam_from_ref = torch.linalg.inv(T_ref_from_cam)
        K = K_per_frame.to(device=pred_ref.device, dtype=torch.float32)

        for b in range(B):
            for src, dst in pairs:
                src_mask = (
                    valid[b, src]
                    & torch.isfinite(pred_ref[b, src]).all(dim=-1)
                    & torch.isfinite(pred_local[b, src]).all(dim=-1)
                )
                if not bool(src_mask.any().item()):
                    continue
                src_ref = pred_ref[b, src][src_mask].float()
                src_ref = _sample_points(src_ref, max_points, gen)
                if src_ref.numel() == 0:
                    continue

                T = T_cam_from_ref[b, dst]
                dst_local = src_ref @ T[:3, :3].T + T[:3, 3]
                z = dst_local[:, 2]
                ok_z = torch.isfinite(dst_local).all(dim=-1) & (z > 1e-6)
                if not bool(ok_z.any().item()):
                    continue
                src_ref = src_ref[ok_z]
                dst_local = dst_local[ok_z]
                z = z[ok_z]

                K_bd = K[b, dst]
                u = dst_local[:, 0] / z * K_bd[0, 0] + K_bd[0, 2]
                v = dst_local[:, 1] / z * K_bd[1, 1] + K_bd[1, 2]
                x = torch.round(u).long()
                y = torch.round(v).long()
                in_img = (x >= 0) & (x < W) & (y >= 0) & (y < H)
                if not bool(in_img.any().item()):
                    continue
                src_ref = src_ref[in_img]
                x = x[in_img]
                y = y[in_img]
                target_valid = valid[b, dst, y, x]
                if not bool(target_valid.any().item()):
                    continue
                src_ref = src_ref[target_valid]
                x = x[target_valid]
                y = y[target_valid]
                dst_ref = pred_ref[b, dst, y, x].float()
                finite = torch.isfinite(dst_ref).all(dim=-1)
                if not bool(finite.any().item()):
                    continue
                err = torch.linalg.norm(src_ref[finite] - dst_ref[finite], dim=-1)
                self.cross_count += float(err.numel())
                self.cross_sum += float(err.sum().item())
                self.cross_med.add(err)

    def state_dict(self) -> Dict:
        state = {name: float(getattr(self, name)) for name in self._STATE_SCALARS}
        state.update(
            {
                "pts_l2_med_values": self.pts_l2_med.values,
                "pts_l2_med_seen": int(self.pts_l2_med.seen),
                "scale_l2_med_values": self.scale_l2_med.values,
                "scale_l2_med_seen": int(self.scale_l2_med.seen),
                "reproj_med_values": self.reproj_med.values,
                "reproj_med_seen": int(self.reproj_med.seen),
                "cross_med_values": self.cross_med.values,
                "cross_med_seen": int(self.cross_med.seen),
                "conf_values": self.conf_pairs.conf,
                "conf_err_values": self.conf_pairs.err,
                "conf_seen": int(self.conf_pairs.seen),
            }
        )
        return state

    @classmethod
    def from_states(cls, args: argparse.Namespace, states: List[Dict]) -> "PointmapMetricAccumulator":
        merged = cls(args)
        for name in cls._STATE_SCALARS:
            setattr(merged, name, float(sum(float(s.get(name, 0.0)) for s in states)))

        def merge_values(key: str, max_size: int, seed: int) -> Optional[torch.Tensor]:
            values = [
                s[key].reshape(-1).to(device="cpu", dtype=torch.float32)
                for s in states
                if s.get(key) is not None and int(s[key].numel()) > 0
            ]
            if not values or max_size <= 0:
                return None
            out = torch.cat(values, dim=0)
            if out.numel() > max_size:
                gen = torch.Generator(device="cpu")
                gen.manual_seed(int(seed))
                idx = torch.randperm(out.numel(), generator=gen)[:max_size]
                out = out[idx]
            return out.contiguous()

        merged.pts_l2_med.values = merge_values(
            "pts_l2_med_values", int(args.stat_sample_max_points), int(args.seed) + 1011
        )
        merged.pts_l2_med.seen = int(sum(int(s.get("pts_l2_med_seen", 0)) for s in states))
        merged.scale_l2_med.values = merge_values(
            "scale_l2_med_values", int(args.stat_sample_max_points), int(args.seed) + 1012
        )
        merged.scale_l2_med.seen = int(sum(int(s.get("scale_l2_med_seen", 0)) for s in states))
        merged.reproj_med.values = merge_values(
            "reproj_med_values", int(args.stat_sample_max_points), int(args.seed) + 1013
        )
        merged.reproj_med.seen = int(sum(int(s.get("reproj_med_seen", 0)) for s in states))
        merged.cross_med.values = merge_values(
            "cross_med_values", int(args.stat_sample_max_points), int(args.seed) + 1014
        )
        merged.cross_med.seen = int(sum(int(s.get("cross_med_seen", 0)) for s in states))

        conf_values = [
            s["conf_values"].reshape(-1).to(device="cpu", dtype=torch.float32)
            for s in states
            if s.get("conf_values") is not None and int(s["conf_values"].numel()) > 0
        ]
        err_values = [
            s["conf_err_values"].reshape(-1).to(device="cpu", dtype=torch.float32)
            for s in states
            if s.get("conf_err_values") is not None and int(s["conf_err_values"].numel()) > 0
        ]
        if conf_values and err_values and int(args.confidence_max_points) > 0:
            conf = torch.cat(conf_values, dim=0)
            err = torch.cat(err_values, dim=0)
            if conf.numel() != err.numel():
                raise RuntimeError(
                    f"Confidence reservoir merge mismatch: conf={conf.numel()} err={err.numel()}"
                )
            if conf.numel() > int(args.confidence_max_points):
                gen = torch.Generator(device="cpu")
                gen.manual_seed(int(args.seed) + 1015)
                idx = torch.randperm(conf.numel(), generator=gen)[: int(args.confidence_max_points)]
                conf = conf[idx]
                err = err[idx]
            merged.conf_pairs.conf = conf.contiguous()
            merged.conf_pairs.err = err.contiguous()
        merged.conf_pairs.seen = int(sum(int(s.get("conf_seen", 0)) for s in states))
        return merged

    def finalize(self) -> Dict[str, float]:
        out: Dict[str, float] = {
            "pts3d_valid": self.count,
            "pts3d_l1": self.pts_l1_sum / max(self.count, 1.0),
            "pts3d_l2": self.pts_l2_sum / max(self.count, 1.0),
            "pts3d_median_error": self.pts_l2_med.median(),
            "scale_aligned_pts3d_l1": self.scale_l1_sum / max(self.scale_count, 1.0),
            "scale_aligned_pts3d_l2": self.scale_l2_sum / max(self.scale_count, 1.0),
            "scale_aligned_pts3d_median_error": self.scale_l2_med.median(),
            "depth_absrel": self.depth_absrel_sum / max(self.depth_count, 1.0),
            "depth_rmse": math.sqrt(self.depth_sq_sum / max(self.depth_count, 1.0)),
            "depth_delta_lt_1_25": self.depth_delta_sum / max(self.depth_count, 1.0),
            "depth_valid": self.depth_count,
            "reprojection_error_px": self.reproj_sum / max(self.reproj_count, 1.0),
            "reprojection_median_error_px": self.reproj_med.median(),
            "reprojection_valid": self.reproj_count,
        }

        chamfer_f = self.chamfer_f_sum / max(self.chamfer_f_count, 1.0)
        chamfer_b = self.chamfer_b_sum / max(self.chamfer_b_count, 1.0)
        precision = self.fscore_prec_hits / max(self.chamfer_f_count, 1.0)
        recall = self.fscore_rec_hits / max(self.chamfer_b_count, 1.0)
        fscore = 2.0 * precision * recall / max(precision + recall, 1e-12)
        out.update(
            {
                "chamfer_distance": 0.5 * (chamfer_f + chamfer_b),
                "chamfer_forward": chamfer_f,
                "chamfer_backward": chamfer_b,
                "fscore": fscore,
                "fscore_precision": precision,
                "fscore_recall": recall,
                "fscore_threshold": float(self.args.fscore_threshold),
                "chamfer_pred_points": self.chamfer_f_count,
                "chamfer_gt_points": self.chamfer_b_count,
                "cross_view_consistency_l2": self.cross_sum / max(self.cross_count, 1.0),
                "cross_view_consistency_median": self.cross_med.median(),
                "cross_view_valid": self.cross_count,
            }
        )
        good_thr = (
            float(self.args.confidence_good_threshold)
            if self.args.confidence_good_threshold is not None
            else float(self.args.fscore_threshold)
        )
        out.update(self.conf_pairs.aucs(good_thr))
        return out


def _sample_points(points: torch.Tensor, max_points: int, gen: torch.Generator) -> torch.Tensor:
    if points.shape[0] <= int(max_points):
        return points
    idx = torch.randperm(points.shape[0], device=points.device, generator=gen)[: int(max_points)]
    return points[idx]


def _nearest_distances(src: torch.Tensor, dst: torch.Tensor, chunk_size: int) -> torch.Tensor:
    outs: List[torch.Tensor] = []
    chunk = max(int(chunk_size), 1)
    dst = dst.float()
    for start in range(0, src.shape[0], chunk):
        d = torch.cdist(src[start:start + chunk].float(), dst)
        outs.append(d.min(dim=1).values)
    return torch.cat(outs, dim=0)


def _view_pairs(n_views: int, mode: str) -> List[Tuple[int, int]]:
    if mode == "none":
        return []
    if mode == "adjacent":
        pairs: List[Tuple[int, int]] = []
        for i in range(n_views - 1):
            pairs.append((i, i + 1))
            pairs.append((i + 1, i))
        return pairs
    return [(i, j) for i in range(n_views) for j in range(n_views) if i != j]


def _valid_pointmap_mask(
    pred_ref: torch.Tensor,
    pred_local: torch.Tensor,
    dense_depth: torch.Tensor,
    K_per_frame: torch.Tensor,
    cam2world: torch.Tensor,
    frame_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gt_ref, gt_local, valid = _pointmap_targets_from_depth(
        dense_depth.float(),
        K_per_frame.float(),
        cam2world.float(),
    )
    if pred_ref.shape != gt_ref.shape or pred_local.shape != gt_local.shape:
        raise RuntimeError(
            f"Pointmap shape mismatch: pred_ref={tuple(pred_ref.shape)} "
            f"pred_local={tuple(pred_local.shape)} gt_ref={tuple(gt_ref.shape)} "
            f"gt_local={tuple(gt_local.shape)}."
        )
    if frame_mask is not None:
        valid = valid & frame_mask.to(device=valid.device, dtype=torch.bool).view(
            valid.shape[0], valid.shape[1], 1, 1
        )
    return gt_ref, gt_local, valid


def _json_safe(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _build_eval_loader(args: argparse.Namespace, dataset):
    if not args.distributed:
        return _build_loader(args, dataset, train=False), len(dataset)
    rank = misc.get_rank()
    world_size = misc.get_world_size()
    indices = list(range(rank, len(dataset), world_size))
    shard = Subset(dataset, indices)
    loader_args = copy.copy(args)
    loader_args.distributed = False
    return _build_loader(loader_args, shard, train=False), len(indices)


def _gather_metric_states(local_state: Dict, args: argparse.Namespace) -> Optional[List[Dict]]:
    if not args.distributed:
        return [local_state]
    world_size = misc.get_world_size()
    if hasattr(dist, "gather_object"):
        gathered = [None for _ in range(world_size)] if misc.is_main_process() else None
        dist.gather_object(local_state, object_gather_list=gathered, dst=0)
        return gathered if misc.is_main_process() else None
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_state)
    return gathered if misc.is_main_process() else None


@torch.no_grad()
def evaluate_pointmap(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    args: argparse.Namespace,
    log_progress: bool = True,
) -> Tuple[Dict, int]:
    model.eval()
    accum = PointmapMetricAccumulator(args)
    t0 = time.time()
    n_batches = 0

    for step, batch in enumerate(loader):
        if args.max_batches > 0 and step >= args.max_batches:
            break
        with _amp_context(device, args.amp):
            out = _model_forward(model, batch, device, args)

        required = ("pointmap_pts3d", "pointmap_pts3d_local")
        missing = [k for k in required if k not in out]
        if missing:
            raise RuntimeError(f"Model output is missing pointmap keys: {missing}")

        pred_ref = out["pointmap_pts3d"].float()
        pred_local = out["pointmap_pts3d_local"].float()
        pred_conf = out.get("pointmap_conf")
        if pred_conf is not None:
            pred_conf = pred_conf.float()

        dense_depth = batch["dense_depth"].to(device=device, dtype=torch.float32, non_blocking=True)
        if "K_per_frame" in batch:
            K_per_frame = batch["K_per_frame"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
        else:
            K_per_frame = torch.stack(
                [
                    v["camera_intrinsics"].to(
                        device=device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )
                    for v in batch["views"]
                ],
                dim=1,
            )
        cam2world = _stack_cam2world_from_views(batch, device=device)
        frame_mask = batch.get("dense_depth_frame_mask")
        if frame_mask is not None:
            frame_mask = frame_mask.to(device=device, non_blocking=True)

        gt_ref, _gt_local, valid = _valid_pointmap_mask(
            pred_ref,
            pred_local,
            dense_depth,
            K_per_frame,
            cam2world,
            frame_mask,
        )

        accum.update_point_errors(pred_ref, gt_ref, valid, pred_conf)
        accum.update_scale_aligned(pred_ref, gt_ref, valid)
        accum.update_depth(pred_local, dense_depth, valid)
        accum.update_reprojection(pred_local, K_per_frame, valid)
        accum.update_chamfer(pred_ref, gt_ref, valid)
        accum.update_cross_view(pred_ref, pred_local, K_per_frame, cam2world, valid)
        n_batches += 1

        if log_progress and args.print_freq > 0 and (step + 1) % args.print_freq == 0:
            elapsed = str(_datetime.timedelta(seconds=int(time.time() - t0)))
            stats = accum.finalize()
            print(
                f"[{step + 1}/{len(loader)}] "
                f"pts3d_l2={stats['pts3d_l2']:.4f} "
                f"depth_absrel={stats['depth_absrel']:.4f} "
                f"chamfer={stats['chamfer_distance']:.4f} "
                f"elapsed={elapsed}"
            )

    state = accum.state_dict()
    state["n_batches"] = float(n_batches)
    return state, n_batches


def main() -> None:
    args = get_args_parser().parse_args()
    misc.init_distributed_mode(args)
    toggle_memory_efficient_attention(enabled=False)
    register_legacy_checkpoint_modules()

    ckpt_path = _resolve_ckpt_path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _fill_args_from_checkpoint(args, ckpt.get("args", {}))

    cudnn.benchmark = True
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    model = _build_pointmap_model(args, device)
    missing, unexpected = _load_pointmap_weights(model, ckpt)
    model.eval()

    val_dataset = _build_dataset(args, "val")
    val_loader, rank_samples = _build_eval_loader(args, val_dataset)

    backbone_hash = _state_dict_hash(model.backbone.state_dict())
    if misc.is_main_process():
        print(f"[pointmap-eval] ckpt={ckpt_path}")
        print(
            f"[pointmap-eval] device={device} amp={args.amp} "
            f"val_samples={len(val_dataset)} world_size={misc.get_world_size()} "
            f"rank0_samples={rank_samples}"
        )
        print(
            f"[pointmap-eval] load_state missing={len(missing)} "
            f"unexpected={len(unexpected)} backbone_hash={backbone_hash}"
        )

    t0 = time.time()
    local_state, _local_batches = evaluate_pointmap(
        model,
        val_loader,
        device,
        args,
        log_progress=misc.is_main_process(),
    )
    gathered_states = _gather_metric_states(local_state, args)

    if args.distributed:
        dist.barrier()

    if not misc.is_main_process():
        return

    assert gathered_states is not None
    merged_accum = PointmapMetricAccumulator.from_states(args, gathered_states)
    metrics = merged_accum.finalize()
    metrics["n_batches"] = float(sum(float(s.get("n_batches", 0.0)) for s in gathered_states))
    elapsed = str(_datetime.timedelta(seconds=int(time.time() - t0)))

    result = {
        "ckpt": str(ckpt_path),
        "elapsed": elapsed,
        "amp": args.amp,
        "batch_size": int(args.batch_size),
        "world_size": int(misc.get_world_size()),
        "val_samples": int(len(val_dataset)),
        "chamfer_max_points": int(args.chamfer_max_points),
        "cross_view_pairs": args.cross_view_pairs,
        "metrics": metrics,
    }

    result_safe = _json_safe(result)
    print(json.dumps(result_safe, indent=2, sort_keys=True))
    if args.output_json:
        out_path = Path(args.output_json)
    else:
        out_path = ckpt_path.parent / "pointmap_quality_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result_safe, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[pointmap-eval] wrote {out_path}")


if __name__ == "__main__":
    main()
