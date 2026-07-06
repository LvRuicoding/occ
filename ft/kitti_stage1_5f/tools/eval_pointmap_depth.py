"""Depth-only evaluation for Stage-1 depth/pointmap checkpoints.

Examples:
  python -m ft.kitti_stage1_5f.tools.eval_pointmap_depth \
    --ckpt output/kitti_stage1_5f_4gpu_pointmap_original

  python -m ft.kitti_stage1_5f.tools.eval_pointmap_depth \
    --ckpt output/kitti_stage1_5f_4gpu_pointmap_postfusion_only

  python -m ft.kitti_stage1_5f.tools.eval_pointmap_depth \
    --ckpt output/kitti_ddad_stage1_5f_4gpu_depth_original/checkpoint-last.pth \
    --eval_dataset ddad

Only ``--ckpt`` is required. Dataset/model settings are restored from the
checkpoint args. By default this script evaluates model ``dense_depth`` output;
legacy pointmap-z evaluation requires ``--depth_source pointmap``.
"""
from __future__ import annotations

try:
    from .. import _paths  # noqa: F401  (must run before project imports)
except ImportError:  # Allows direct `python ft/.../eval_pointmap_depth.py`.
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from ft.kitti_stage1_5f import _paths  # noqa: F401

import argparse
import copy
import contextlib
import datetime as _datetime
import json
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

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
    Stage1DepthOriginalModel,
    Stage1DepthPostFusionOnlyModel,
    Stage1DepthPromptFusionOnlyModel,
    Stage1PointmapOriginalModel,
    Stage1PointmapPostFusionOnlyModel,
    Stage1SSCBEVDetOccLidarPointmapDenseDepthModel,
)
from ft.kitti_stage1_5f.tools.train import (
    _build_ddad_dataset,
    _build_kitti_dataset,
    _build_loader,
    _model_forward,
    _state_dict_hash,
)


SUPPORTED_EXPS = (
    "depth_original",
    "depth_postfusion_only",
    "depth_promptfusion_only",
    "pointmap_original",
    "pointmap_postfusion_only",
    "bevdetocc_lidar_pointmap_dense_depth",
)
DEPTH_BINS: Tuple[Tuple[float, float], ...] = (
    (0.0, 10.0),
    (10.0, 20.0),
    (20.0, 40.0),
    (40.0, 80.0),
)


def get_args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Depth-only evaluation for Stage-1 pointmap checkpoints")
    p.add_argument(
        "--ckpt",
        required=True,
        type=str,
        help="Checkpoint file, or an experiment directory containing checkpoint-last.pth.",
    )
    p.add_argument(
        "--eval_dataset",
        choices=["kitti", "ddad"],
        default="kitti",
        help="Validation dataset to evaluate. Paths are restored from checkpoint args unless overridden.",
    )
    p.add_argument("--processed_root", default=None, type=str, help="Override KITTI processed root.")
    p.add_argument("--ddad_processed_root", default=None, type=str, help="Override DDAD processed root.")
    p.add_argument("--ddad_raw_root", default=None, type=str, help="Override DDAD raw root.")
    p.add_argument("--occany_ckpt", default=None, type=str, help="Override OccAny backbone checkpoint path.")
    p.add_argument("--velodyne_root", default=None, type=str, help="Override/deprecated KITTI velodyne root.")
    p.add_argument("--device", default="auto", type=str)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--amp", choices=["bf16", "fp16", "none"], default=None)
    p.add_argument("--min_depth", type=float, default=1e-3)
    p.add_argument("--max_depth", type=float, default=80.0)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--print_freq", type=int, default=20)
    p.add_argument("--reservoir_size", type=int, default=2_000_000)
    p.add_argument("--output_json", default=None, type=str)
    p.add_argument(
        "--depth_source",
        choices=["auto", "dense_depth", "pointmap"],
        default="auto",
        help=(
            "Depth prediction to evaluate. auto/dense_depth require model output "
            "'dense_depth'; pointmap explicitly evaluates pointmap_pts3d_local[..., 2]."
        ),
    )
    p.add_argument(
        "--target_frame_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, evaluate only frame index 0 instead of all input frames.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--world_size", default=1, type=int)
    p.add_argument("--local_rank", default=-1, type=int)
    p.add_argument("--dist_url", default="env://", type=str)
    p.add_argument(
        "--nodist",
        action="store_true",
        help="Disable distributed mode even under torchrun.",
    )
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
    args.exp = _ckpt_arg(ckpt_args, "exp", None)
    if args.exp not in SUPPORTED_EXPS:
        raise ValueError(
            f"eval_pointmap_depth supports {SUPPORTED_EXPS}, got checkpoint exp={args.exp!r}."
        )

    args.processed_root = _override_or_ckpt(args, ckpt_args, "processed_root", None)
    args.ddad_processed_root = _override_or_ckpt(args, ckpt_args, "ddad_processed_root", None)
    args.ddad_raw_root = _override_or_ckpt(args, ckpt_args, "ddad_raw_root", None)
    args.occany_ckpt = _override_or_ckpt(args, ckpt_args, "occany_ckpt", None)
    args.velodyne_root = _override_or_ckpt(args, ckpt_args, "velodyne_root", None)
    args.width = int(_ckpt_arg(ckpt_args, "width", 512))
    args.height = int(_ckpt_arg(ckpt_args, "height", 160))
    args.num_frames = int(_ckpt_arg(ckpt_args, "num_frames", 5))
    args.frame_stride = int(_ckpt_arg(ckpt_args, "frame_stride", 4))
    args.c_lift = int(_ckpt_arg(ckpt_args, "c_lift", 64))
    args.token_dim = int(_ckpt_arg(ckpt_args, "token_dim", 768))
    args.patch_size = int(_ckpt_arg(ckpt_args, "patch_size", 16))
    args.backbone = _ckpt_arg(ckpt_args, "backbone", "must3r")
    args.dense_depth_features = int(_ckpt_arg(ckpt_args, "dense_depth_features", 128))
    args.prompt_depth_scale = _ckpt_arg(ckpt_args, "prompt_depth_scale", "log")
    args.prompt_depth_min = float(_ckpt_arg(ckpt_args, "prompt_depth_min", 1e-3))
    args.prompt_depth_max = float(_ckpt_arg(ckpt_args, "prompt_depth_max", 120.0))
    args.max_points_per_sweep = int(_ckpt_arg(ckpt_args, "max_points_per_sweep", 0))
    args.freeze_backbone = bool(_ckpt_arg(ckpt_args, "freeze_backbone", True))
    args.batch_size = int(_override_or_ckpt(args, ckpt_args, "batch_size", 1))
    args.num_workers = int(_override_or_ckpt(args, ckpt_args, "num_workers", 4))
    args.amp = args.amp or _ckpt_arg(ckpt_args, "amp", "bf16")

    # Consumed by train.py helpers.
    args.multi_dataset = False
    args.depth_supervision = False
    args.dense_depth_supervision = False
    args.pointmap_supervision = False

    if not args.processed_root:
        raise ValueError("Checkpoint args do not contain processed_root.")
    if args.eval_dataset == "ddad" and not args.ddad_processed_root:
        raise ValueError(
            "DDAD evaluation requires ddad_processed_root in the checkpoint or "
            "an explicit --ddad_processed_root."
        )
    if args.eval_dataset == "ddad" and args.exp not in (
        "depth_original",
        "depth_postfusion_only",
    ):
        raise ValueError(
            "--eval_dataset ddad is supported only for depth_original or "
            "depth_postfusion_only checkpoints."
        )
    if not args.occany_ckpt:
        raise ValueError("Checkpoint args do not contain occany_ckpt.")


def _build_depth_eval_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if args.amp == "bf16" and device.type == "cuda":
        backbone_dtype = torch.bfloat16
    elif args.amp == "fp16" and device.type == "cuda":
        backbone_dtype = torch.float16
    else:
        backbone_dtype = torch.float32

    if args.exp == "depth_original":
        model_cls = Stage1DepthOriginalModel
    elif args.exp == "depth_postfusion_only":
        model_cls = Stage1DepthPostFusionOnlyModel
    elif args.exp == "depth_promptfusion_only":
        model_cls = Stage1DepthPromptFusionOnlyModel
    elif args.exp == "pointmap_original":
        model_cls = Stage1PointmapOriginalModel
    elif args.exp == "pointmap_postfusion_only":
        model_cls = Stage1PointmapPostFusionOnlyModel
    elif args.exp == "bevdetocc_lidar_pointmap_dense_depth":
        model_cls = Stage1SSCBEVDetOccLidarPointmapDenseDepthModel
    else:
        raise ValueError(f"Unsupported exp={args.exp!r}.")
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
    if args.exp in (
        "depth_original",
        "depth_postfusion_only",
        "depth_promptfusion_only",
        "bevdetocc_lidar_pointmap_dense_depth",
    ):
        model_kwargs["dense_depth_features"] = args.dense_depth_features
    if args.exp == "depth_promptfusion_only":
        model_kwargs["prompt_depth_scale"] = args.prompt_depth_scale
        model_kwargs["prompt_depth_min"] = args.prompt_depth_min
        model_kwargs["prompt_depth_max"] = args.prompt_depth_max
    if args.exp in (
        "pointmap_postfusion_only",
        "depth_postfusion_only",
        "bevdetocc_lidar_pointmap_dense_depth",
    ):
        model_kwargs["fusion_attn_type"] = "cross"
    if args.exp in ("depth_original", "depth_postfusion_only"):
        model_kwargs["backbone"] = args.backbone
    return model_cls(**model_kwargs).to(device)


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state):
        return state
    return {k.removeprefix("module."): v for k, v in state.items()}


def _load_eval_weights(model: torch.nn.Module, ckpt: Dict) -> Tuple[int, int]:
    if "model" not in ckpt:
        raise KeyError("Checkpoint must contain a 'model' state_dict.")
    state = _strip_module_prefix(ckpt["model"])
    status = model.load_state_dict(state, strict=False)
    critical_missing = [
        k for k in status.missing_keys
        if not k.startswith("backbone.") and not k.endswith("num_batches_tracked")
    ]
    if critical_missing:
        preview = ", ".join(critical_missing[:10])
        raise RuntimeError(
            f"Checkpoint is missing non-backbone model keys "
            f"({len(critical_missing)}): {preview}"
        )
    return len(status.missing_keys), len(status.unexpected_keys)


def _amp_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


class Reservoir:
    """Approximate streaming sample buffer for median-like depth statistics."""

    def __init__(self, max_size: int, seed: int = 0) -> None:
        self.max_size = int(max_size)
        self.values: Optional[torch.Tensor] = None
        self.seen = 0
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(int(seed))

    def add(self, values: torch.Tensor) -> None:
        v = values.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        n = int(v.numel())
        if n == 0:
            return
        if self.max_size <= 0:
            self.seen += n
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

    def state_dict(self) -> Dict:
        return {
            "values": self.values,
            "seen": int(self.seen),
        }

    @classmethod
    def from_states(cls, states, max_size: int, seed: int) -> "Reservoir":
        merged = cls(max_size=max_size, seed=seed)
        merged.seen = int(sum(int(s.get("seen", 0)) for s in states))
        values = [
            s["values"].reshape(-1).to(device="cpu", dtype=torch.float32)
            for s in states
            if s.get("values") is not None and int(s["values"].numel()) > 0
        ]
        if not values or int(max_size) <= 0:
            return merged
        out = torch.cat(values, dim=0)
        if out.numel() > int(max_size):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            idx = torch.randperm(out.numel(), generator=gen)[: int(max_size)]
            out = out[idx]
        merged.values = out.contiguous()
        return merged


class DepthStats:
    """Streaming scalar depth metrics for one split/bin/frame group."""

    def __init__(self, reservoir_size: int = 0, seed: int = 0) -> None:
        self.count = 0.0
        self.abs_sum = 0.0
        self.abs_rel_sum = 0.0
        self.sq_sum = 0.0
        self.sq_rel_sum = 0.0
        self.log_sum = 0.0
        self.log_sq_sum = 0.0
        self.log10_abs_sum = 0.0
        self.delta1_sum = 0.0
        self.delta2_sum = 0.0
        self.delta3_sum = 0.0
        self.inv_abs_sum = 0.0
        self.inv_sq_sum = 0.0
        self.pred_sum = 0.0
        self.gt_sum = 0.0
        self.pred_reservoir = Reservoir(reservoir_size, seed + 1) if reservoir_size > 0 else None
        self.gt_reservoir = Reservoir(reservoir_size, seed + 2) if reservoir_size > 0 else None
        self.abs_reservoir = Reservoir(reservoir_size, seed + 3) if reservoir_size > 0 else None
        self.abs_rel_reservoir = Reservoir(reservoir_size, seed + 4) if reservoir_size > 0 else None

    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        pred = pred.detach().reshape(-1).float()
        gt = gt.detach().reshape(-1).float()
        if pred.numel() == 0:
            return
        diff = pred - gt
        abs_err = diff.abs()
        log_diff = torch.log(pred) - torch.log(gt)
        log10_abs = (torch.log10(pred) - torch.log10(gt)).abs()
        ratio = torch.maximum(pred / gt.clamp(min=1e-12), gt / pred.clamp(min=1e-12))
        inv_diff = (1.0 / pred.clamp(min=1e-12)) - (1.0 / gt.clamp(min=1e-12))
        n = float(pred.numel())

        self.count += n
        self.abs_sum += float(abs_err.sum().item())
        self.abs_rel_sum += float((abs_err / gt.clamp(min=1e-12)).sum().item())
        self.sq_sum += float(diff.square().sum().item())
        self.sq_rel_sum += float((diff.square() / gt.clamp(min=1e-12)).sum().item())
        self.log_sum += float(log_diff.sum().item())
        self.log_sq_sum += float(log_diff.square().sum().item())
        self.log10_abs_sum += float(log10_abs.sum().item())
        self.delta1_sum += float((ratio < 1.25).float().sum().item())
        self.delta2_sum += float((ratio < 1.25 ** 2).float().sum().item())
        self.delta3_sum += float((ratio < 1.25 ** 3).float().sum().item())
        self.inv_abs_sum += float(inv_diff.abs().sum().item())
        self.inv_sq_sum += float(inv_diff.square().sum().item())
        self.pred_sum += float(pred.sum().item())
        self.gt_sum += float(gt.sum().item())

        if self.pred_reservoir is not None:
            self.pred_reservoir.add(pred)
            self.gt_reservoir.add(gt)
            self.abs_reservoir.add(abs_err)
            self.abs_rel_reservoir.add(abs_err / gt.clamp(min=1e-12))

    def finalize(self) -> Dict[str, float]:
        n = max(self.count, 1.0)
        log_mean = self.log_sum / n
        log_sq_mean = self.log_sq_sum / n
        out = {
            "valid_pixels": self.count,
            "abs_rel": self.abs_rel_sum / n,
            "sq_rel": self.sq_rel_sum / n,
            "rmse": math.sqrt(self.sq_sum / n),
            "rmse_log": math.sqrt(self.log_sq_sum / n),
            "mae": self.abs_sum / n,
            "log10": self.log10_abs_sum / n,
            "silog": math.sqrt(max(log_sq_mean - log_mean * log_mean, 0.0)) * 100.0,
            "delta1": self.delta1_sum / n,
            "delta2": self.delta2_sum / n,
            "delta3": self.delta3_sum / n,
            "imae": (self.inv_abs_sum / n) * 1000.0,
            "irmse": math.sqrt(self.inv_sq_sum / n) * 1000.0,
            "pred_mean": self.pred_sum / n,
            "gt_mean": self.gt_sum / n,
        }
        if self.pred_reservoir is not None:
            out.update(
                {
                    "pred_median": self.pred_reservoir.median(),
                    "gt_median": self.gt_reservoir.median(),
                    "abs_error_median": self.abs_reservoir.median(),
                    "abs_rel_median": self.abs_rel_reservoir.median(),
                }
            )
        return out

    def state_dict(self) -> Dict:
        scalar_names = (
            "count",
            "abs_sum",
            "abs_rel_sum",
            "sq_sum",
            "sq_rel_sum",
            "log_sum",
            "log_sq_sum",
            "log10_abs_sum",
            "delta1_sum",
            "delta2_sum",
            "delta3_sum",
            "inv_abs_sum",
            "inv_sq_sum",
            "pred_sum",
            "gt_sum",
        )
        state = {name: float(getattr(self, name)) for name in scalar_names}
        if self.pred_reservoir is not None:
            state.update(
                {
                    "pred_reservoir": self.pred_reservoir.state_dict(),
                    "gt_reservoir": self.gt_reservoir.state_dict(),
                    "abs_reservoir": self.abs_reservoir.state_dict(),
                    "abs_rel_reservoir": self.abs_rel_reservoir.state_dict(),
                }
            )
        return state

    @classmethod
    def from_states(
        cls,
        states,
        reservoir_size: int = 0,
        seed: int = 0,
    ) -> "DepthStats":
        merged = cls(reservoir_size=reservoir_size, seed=seed)
        scalar_names = (
            "count",
            "abs_sum",
            "abs_rel_sum",
            "sq_sum",
            "sq_rel_sum",
            "log_sum",
            "log_sq_sum",
            "log10_abs_sum",
            "delta1_sum",
            "delta2_sum",
            "delta3_sum",
            "inv_abs_sum",
            "inv_sq_sum",
            "pred_sum",
            "gt_sum",
        )
        for name in scalar_names:
            setattr(merged, name, float(sum(float(s.get(name, 0.0)) for s in states)))
        if int(reservoir_size) > 0:
            merged.pred_reservoir = Reservoir.from_states(
                [s.get("pred_reservoir", {}) for s in states],
                reservoir_size,
                seed + 1001,
            )
            merged.gt_reservoir = Reservoir.from_states(
                [s.get("gt_reservoir", {}) for s in states],
                reservoir_size,
                seed + 1002,
            )
            merged.abs_reservoir = Reservoir.from_states(
                [s.get("abs_reservoir", {}) for s in states],
                reservoir_size,
                seed + 1003,
            )
            merged.abs_rel_reservoir = Reservoir.from_states(
                [s.get("abs_rel_reservoir", {}) for s in states],
                reservoir_size,
                seed + 1004,
            )
        return merged


class DepthMetricAccumulator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.global_stats = DepthStats(
            reservoir_size=int(args.reservoir_size),
            seed=int(args.seed) + 100,
        )
        self.frame_stats = [
            DepthStats(seed=int(args.seed) + 200 + i)
            for i in range(int(args.num_frames))
        ]
        self.bin_stats = {
            self._bin_name(lo, hi): DepthStats(seed=int(args.seed) + 300 + i)
            for i, (lo, hi) in enumerate(DEPTH_BINS)
        }
        self.valid_frames = 0.0
        self.total_frames = 0.0

    @staticmethod
    def _bin_name(lo: float, hi: float) -> str:
        return f"{lo:g}_{hi:g}m"

    def update(
        self,
        pred_depth: torch.Tensor,
        dense_depth: torch.Tensor,
        frame_mask: Optional[torch.Tensor],
    ) -> None:
        pred = pred_depth.float()
        gt = dense_depth.to(device=pred.device, dtype=torch.float32)
        if bool(self.args.target_frame_only):
            pred = pred[:, :1]
            gt = gt[:, :1]
            if frame_mask is not None:
                frame_mask = frame_mask[:, :1]

        min_depth = float(self.args.min_depth)
        max_depth = float(self.args.max_depth)
        valid = (
            torch.isfinite(pred)
            & torch.isfinite(gt)
            & (gt >= min_depth)
            & (gt <= max_depth)
            & (gt > 0.0)
        )
        if frame_mask is not None:
            fm = frame_mask.to(device=pred.device, dtype=torch.bool).view(
                pred.shape[0], pred.shape[1], 1, 1
            )
            valid = valid & fm

        self.total_frames += float(pred.shape[0] * pred.shape[1])
        self.valid_frames += float(valid.view(pred.shape[0], pred.shape[1], -1).any(dim=-1).sum().item())
        if not bool(valid.any().item()):
            return

        pred_eval = pred.clamp(min=min_depth, max=max_depth)
        self.global_stats.update(pred_eval[valid], gt[valid])

        for frame_idx in range(pred.shape[1]):
            frame_valid = valid[:, frame_idx]
            if bool(frame_valid.any().item()):
                self.frame_stats[frame_idx].update(
                    pred_eval[:, frame_idx][frame_valid],
                    gt[:, frame_idx][frame_valid],
                )

        for lo, hi in DEPTH_BINS:
            bin_valid = valid & (gt >= max(lo, min_depth)) & (gt < min(hi, max_depth))
            if bool(bin_valid.any().item()):
                self.bin_stats[self._bin_name(lo, hi)].update(
                    pred_eval[bin_valid],
                    gt[bin_valid],
                )

    def finalize(self) -> Dict:
        metrics = self.global_stats.finalize()
        metrics["valid_frames"] = self.valid_frames
        metrics["total_frames"] = self.total_frames
        metrics["frame_valid_ratio"] = self.valid_frames / max(self.total_frames, 1.0)
        metrics["inverse_depth_unit"] = "1/km"
        return {
            "overall": metrics,
            "by_frame_index": {
                str(i): stats.finalize()
                for i, stats in enumerate(self.frame_stats)
                if stats.count > 0
            },
            "by_gt_depth_range": {
                name: stats.finalize()
                for name, stats in self.bin_stats.items()
                if stats.count > 0
            },
        }

    def state_dict(self) -> Dict:
        return {
            "global_stats": self.global_stats.state_dict(),
            "frame_stats": [stats.state_dict() for stats in self.frame_stats],
            "bin_stats": {name: stats.state_dict() for name, stats in self.bin_stats.items()},
            "valid_frames": float(self.valid_frames),
            "total_frames": float(self.total_frames),
        }

    @classmethod
    def from_states(cls, args: argparse.Namespace, states) -> "DepthMetricAccumulator":
        merged = cls(args)
        reservoir_size = int(args.reservoir_size)
        seed = int(args.seed)
        merged.global_stats = DepthStats.from_states(
            [s["global_stats"] for s in states],
            reservoir_size=reservoir_size,
            seed=seed + 1100,
        )
        merged.frame_stats = []
        for frame_idx in range(int(args.num_frames)):
            frame_states = [
                s["frame_stats"][frame_idx]
                for s in states
                if frame_idx < len(s.get("frame_stats", []))
            ]
            merged.frame_stats.append(
                DepthStats.from_states(frame_states, seed=seed + 1200 + frame_idx)
            )
        merged.bin_stats = {}
        for i, (lo, hi) in enumerate(DEPTH_BINS):
            name = cls._bin_name(lo, hi)
            merged.bin_stats[name] = DepthStats.from_states(
                [s["bin_stats"][name] for s in states if name in s.get("bin_stats", {})],
                seed=seed + 1300 + i,
            )
        merged.valid_frames = float(sum(float(s.get("valid_frames", 0.0)) for s in states))
        merged.total_frames = float(sum(float(s.get("total_frames", 0.0)) for s in states))
        return merged


def _json_safe(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _print_metrics(result: Dict) -> None:
    model_output = result.get("model_output", {})
    if model_output:
        print("\nModel output")
        print(f"  output_keys             : {model_output.get('output_keys')}")
        print(f"  has_dense_depth_output  : {model_output.get('has_dense_depth_output')}")
        print(f"  depth_source            : {model_output.get('depth_source')}")
        print(f"  dense_depth_shape       : {model_output.get('dense_depth_shape')}")
        print(f"  pointmap_local_shape    : {model_output.get('pointmap_pts3d_local_shape')}")
        print(f"  pred_depth_shape        : {model_output.get('pred_depth_shape')}")
        print(f"  gt_dense_depth_shape    : {model_output.get('gt_dense_depth_shape')}")
        print(f"  dense_per_pixel_depth   : {model_output.get('pred_depth_is_dense_per_pixel')}")

    overall = result["metrics"]["overall"]
    print("\nDepth metrics (overall)")
    for key in (
        "valid_pixels",
        "valid_frames",
        "abs_rel",
        "sq_rel",
        "rmse",
        "rmse_log",
        "mae",
        "log10",
        "silog",
        "delta1",
        "delta2",
        "delta3",
        "imae",
        "irmse",
        "pred_mean",
        "gt_mean",
        "pred_median",
        "gt_median",
        "abs_error_median",
        "abs_rel_median",
    ):
        if key in overall:
            value = overall[key]
            if isinstance(value, float):
                print(f"  {key:18s}: {value:.6f}")
            else:
                print(f"  {key:18s}: {value}")

    print("\nBy GT depth range")
    for name, stats in result["metrics"]["by_gt_depth_range"].items():
        print(
            f"  {name:8s} pixels={stats['valid_pixels']:.0f} "
            f"abs_rel={stats['abs_rel']:.6f} rmse={stats['rmse']:.6f} "
            f"delta1={stats['delta1']:.6f}"
        )

    print("\nBy frame index")
    for name, stats in result["metrics"]["by_frame_index"].items():
        print(
            f"  frame {name:>2s} pixels={stats['valid_pixels']:.0f} "
            f"abs_rel={stats['abs_rel']:.6f} rmse={stats['rmse']:.6f} "
            f"delta1={stats['delta1']:.6f}"
        )


@torch.no_grad()
def evaluate_depth(model, loader, device: torch.device, args: argparse.Namespace) -> Tuple[Dict, int, Dict]:
    model.eval()
    accum = DepthMetricAccumulator(args)
    t0 = time.time()
    n_batches = 0
    output_info = {}

    for step, batch in enumerate(loader):
        if int(args.max_batches) > 0 and step >= int(args.max_batches):
            break
        with _amp_context(device, args.amp):
            out = _model_forward(model, batch, device, args)
        if args.depth_source in ("auto", "dense_depth"):
            if "dense_depth" not in out:
                raise RuntimeError(
                    "Model output is missing 'dense_depth'. This script evaluates dense-depth "
                    "fine-tuning by default; pass --depth_source pointmap only if you intentionally "
                    "want legacy pointmap-z evaluation."
                )
            pred_depth = out["dense_depth"].float()
            depth_source = "dense_depth"
        elif args.depth_source == "pointmap":
            if "pointmap_pts3d_local" not in out:
                raise RuntimeError("Model output is missing 'pointmap_pts3d_local'.")
            pred_depth = out["pointmap_pts3d_local"][..., 2].float()
            depth_source = "pointmap_pts3d_local[..., 2]"
        else:
            raise ValueError(f"Unsupported depth_source={args.depth_source!r}.")
        dense_depth = batch["dense_depth"].to(device=device, dtype=torch.float32, non_blocking=True)
        if not output_info:
            output_info = {
                "output_keys": sorted(str(k) for k in out.keys()),
                "has_dense_depth_output": "dense_depth" in out,
                "depth_source": depth_source,
                "dense_depth_shape": list(out["dense_depth"].shape) if "dense_depth" in out else None,
                "pointmap_pts3d_local_shape": (
                    list(out["pointmap_pts3d_local"].shape)
                    if "pointmap_pts3d_local" in out
                    else None
                ),
                "pred_depth_shape": list(pred_depth.shape),
                "gt_dense_depth_shape": list(dense_depth.shape),
                "pred_depth_is_dense_per_pixel": bool(pred_depth.shape == dense_depth.shape),
            }
        frame_mask = batch.get("dense_depth_frame_mask")
        if frame_mask is not None:
            frame_mask = frame_mask.to(device=device, non_blocking=True)

        accum.update(pred_depth, dense_depth, frame_mask)
        n_batches += 1

        if int(args.print_freq) > 0 and (step + 1) % int(args.print_freq) == 0:
            elapsed = str(_datetime.timedelta(seconds=int(time.time() - t0)))
            stats = accum.finalize()["overall"]
            print(
                f"[{step + 1}/{len(loader)}] "
                f"abs_rel={stats['abs_rel']:.6f} rmse={stats['rmse']:.6f} "
                f"delta1={stats['delta1']:.6f} elapsed={elapsed}"
            )

    return accum.state_dict(), n_batches, output_info


def _build_eval_dataset(args: argparse.Namespace):
    if args.eval_dataset == "kitti":
        return _build_kitti_dataset(args, "val")
    if args.eval_dataset == "ddad":
        return _build_ddad_dataset(args, "val")
    raise ValueError(f"Unsupported eval_dataset={args.eval_dataset!r}.")


def _build_eval_loader(args: argparse.Namespace, dataset):
    if not bool(getattr(args, "distributed", False)):
        return _build_loader(args, dataset, train=False), len(dataset)
    rank = misc.get_rank()
    world_size = misc.get_world_size()
    indices = list(range(rank, len(dataset), world_size))
    shard = Subset(dataset, indices)
    loader_args = copy.copy(args)
    loader_args.distributed = False
    return _build_loader(loader_args, shard, train=False), len(indices)


def _gather_states(local_state: Dict, args: argparse.Namespace):
    if not bool(getattr(args, "distributed", False)):
        return [local_state]
    world_size = misc.get_world_size()
    if hasattr(dist, "gather_object"):
        gathered = [None for _ in range(world_size)] if misc.is_main_process() else None
        dist.gather_object(local_state, object_gather_list=gathered, dst=0)
        return gathered if misc.is_main_process() else None
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_state)
    return gathered if misc.is_main_process() else None


def main() -> None:
    args = get_args_parser().parse_args()
    misc.init_distributed_mode(args)
    toggle_memory_efficient_attention(enabled=False)
    register_legacy_checkpoint_modules()

    ckpt_path = _resolve_ckpt_path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _fill_args_from_checkpoint(args, ckpt.get("args", {}))

    cudnn.benchmark = True
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    if bool(getattr(args, "distributed", False)):
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    model = _build_depth_eval_model(args, device)
    missing_count, unexpected_count = _load_eval_weights(model, ckpt)
    val_dataset = _build_eval_dataset(args)
    val_loader, rank_samples = _build_eval_loader(args, val_dataset)

    backbone_hash = (
        _state_dict_hash(model.backbone.state_dict())
        if hasattr(model, "backbone") and getattr(args, "backbone", "must3r") == "must3r"
        else None
    )
    if misc.is_main_process():
        print(f"[depth-eval] ckpt={ckpt_path}")
        print(
            f"[depth-eval] exp={args.exp} backbone={getattr(args, 'backbone', 'must3r')} "
            f"device={device} amp={args.amp} "
            f"eval_dataset={args.eval_dataset} "
            f"val_samples={len(val_dataset)} rank0_samples={rank_samples} "
            f"world_size={misc.get_world_size()} batch_size={args.batch_size}"
        )
        print(
            f"[depth-eval] load_state missing={missing_count} "
            f"unexpected={unexpected_count} backbone_hash={backbone_hash}"
        )

    t0 = time.time()
    local_metrics_state, n_batches, output_info = evaluate_depth(model, val_loader, device, args)
    local_state = {
        "metrics_state": local_metrics_state,
        "num_batches": int(n_batches),
        "output_info": output_info,
    }
    gathered_states = _gather_states(local_state, args)
    if bool(getattr(args, "distributed", False)):
        dist.barrier()
    if not misc.is_main_process():
        return

    assert gathered_states is not None
    metrics_accum = DepthMetricAccumulator.from_states(
        args,
        [state["metrics_state"] for state in gathered_states],
    )
    metrics = metrics_accum.finalize()
    n_batches = int(sum(int(state.get("num_batches", 0)) for state in gathered_states))
    output_info = next(
        (state.get("output_info", {}) for state in gathered_states if state.get("output_info")),
        {},
    )
    elapsed = str(_datetime.timedelta(seconds=int(time.time() - t0)))

    result = {
        "ckpt": str(ckpt_path),
        "exp": args.exp,
        "eval_dataset": args.eval_dataset,
        "elapsed": elapsed,
        "amp": args.amp,
        "batch_size": int(args.batch_size),
        "world_size": int(misc.get_world_size()),
        "num_batches": int(n_batches),
        "val_samples": int(len(val_dataset)),
        "min_depth": float(args.min_depth),
        "max_depth": float(args.max_depth),
        "target_frame_only": bool(args.target_frame_only),
        "depth_source": args.depth_source,
        "model_output": output_info,
        "metrics": metrics,
    }
    result_safe = _json_safe(result)
    _print_metrics(result_safe)

    if args.output_json:
        out_path = Path(args.output_json)
    elif args.eval_dataset == "kitti":
        out_path = ckpt_path.parent / "depth_metrics.json"
    else:
        out_path = ckpt_path.parent / f"depth_metrics_{args.eval_dataset}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result_safe, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\n[depth-eval] wrote {out_path}")


if __name__ == "__main__":
    main()
