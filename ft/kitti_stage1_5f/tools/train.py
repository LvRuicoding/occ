"""Training entry for Stage-1 SSC fine-tuning of OccAny on SemanticKITTI.

Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=4 \
      -m ft.kitti_stage1_5f.tools.train \
      --processed_root /home/dataset-local/lr/code/OccAny/data/kitti_processed \
      --kittiodo_root /home/dataset-local/lr/code/OccAny/raw_data/semantickitti \
      --velodyne_root /home/dataset-local/lr/code/OccAny/data/kitti \
      --occany_ckpt /home/dataset-local/lr/code/OccAny/checkpoints/occany_recon.pth \
      --output_dir /home/dataset-local/lr/code/OccAny/output/kitti_stage1_5f_4gpu_monoscene_lidar_selfattn \
      --exp monoscene_lidar \
      --fusion_attn_type self \
      --batch_size 1 \
      --num_workers 6 \
      --amp bf16 \
      --epochs 40 \
      --lr 1e-4
"""
from __future__ import annotations

from .. import _paths  # noqa: F401  (must come first)

import argparse
import datetime
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler

from occany.metrics.ssc import SSCMetrics
from occany.model.must3r_blocks.attention import toggle_memory_efficient_attention
from occany.utils.checkpoint_io import register_legacy_checkpoint_modules

from ft.semantickitti_ft.losses import SSCLoss
from ..datasets import (
    KITTI_SSC_CLASS_NAMES,
    Kitti5FrameStage1Dataset,
    Kitti5FrameStage1LidarDataset,
    Kitti5FrameStage1MonoDataset,
    Kitti5FrameStage1MonoLidarDataset,
    collate_stage1,
    collate_stage1_lidar,
    collate_stage1_mono,
    collate_stage1_mono_lidar,
)
from ..losses_monoscene import MonoSceneSSCLoss
from ..models import (
    Stage1SSCBEVDetOccLidarModel,
    Stage1SSCModel,
    Stage1SSCMonoModel,
    Stage1SSCMonoLidarModel,
)
from ..models.stage1_ssc_bevdetocc_lidar import bevdet_depth_loss, dense_depth_loss


LIDAR_EXPS = ("monoscene_lidar", "bevdetocc_lidar")
MONOSCENE_LOSS_EXPS = ("monoscene", "monoscene_lidar")


def get_args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("OccAny Stage-1 SSC fine-tune", add_help=True)
    p.add_argument("--processed_root", required=True, type=str,
                   help="Path to data/kitti_processed.")
    p.add_argument("--kittiodo_root", default=None, type=str,
                   help="Deprecated; calib.txt is read from processed_root/<split>_<seq>.")
    p.add_argument("--occany_ckpt", required=True, type=str,
                   help="OccAny checkpoint (encoder+decoder sub-dicts).")
    p.add_argument("--output_dir", required=True, type=str)

    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=160)
    p.add_argument("--num_frames", type=int, default=5)
    p.add_argument("--frame_stride", type=int, default=1)

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=1)
    p.add_argument("--min_lr", type=float, default=1e-6)

    p.add_argument("--c_lift", type=int, default=64)
    p.add_argument("--token_dim", type=int, default=768)
    p.add_argument("--patch_size", type=int, default=16)

    p.add_argument("--print_freq", type=int, default=20)
    p.add_argument("--save_freq", type=int, default=2)
    p.add_argument("--keep_freq", type=int, default=5)
    p.add_argument("--eval_freq", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    p.add_argument("--accum_iter", type=int, default=1)
    p.add_argument("--device", default="cuda", type=str)

    p.add_argument("--world_size", default=1, type=int)
    p.add_argument("--local_rank", default=-1, type=int)
    p.add_argument("--dist_url", default="env://", type=str)
    p.add_argument("--resume", default=None, type=str)
    p.add_argument("--eval_only", action="store_true")

    # Which experiment variant to run.
    #   - "light":            existing LightOcc3DUNet head + CE+Lovasz loss.
    #   - "monoscene":        vendored MonoScene UNet3D head (context_prior=True)
    #                         via adapter, + CE+sem_scal+geo_scal+relation_ce
    #                         loss (requires <frame>_1_8.npy under processed_root).
    #   - "monoscene_lidar":  same as monoscene + a LiDAR fusion
    #                         block applied to OccAny's reconstruction tokens
    #                         (post-decoder, pre-lifting). The OccAny backbone
    #                         stays fully frozen; only fusion/lifting/head train.
    #                         Requires --velodyne_root.
    #   - "bevdetocc_lidar": keep the first 2D LiDAR/image cross-attention,
    #                         then use LSS + LiDAR memory + NATTEN + BEVDet-OCC
    #                         3D encoder/head. Requires --velodyne_root.
    p.add_argument(
        "--exp",
        choices=["light", "monoscene", "monoscene_lidar", "bevdetocc_lidar"],
        default="light",
    )
    # LiDAR-fusion-only options.
    p.add_argument("--velodyne_root", default=None, type=str,
                   help="Raw KITTI Odometry root: <velodyne_root>/sequences/<seq>/velodyne/*.bin. "
                        "Required for LiDAR experiments.")
    p.add_argument("--max_points_per_sweep", type=int, default=0,
                   help="If >0, deterministically stride-subsample each LiDAR sweep to this point count.")
    p.add_argument("--depth_supervision", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable BEVDet-style sparse LiDAR depth supervision for --exp=bevdetocc_lidar.")
    p.add_argument("--depth_loss_weight", type=float, default=0.05,
                   help="Weight for BEVDet-style LSS depth loss when --depth_supervision is enabled.")
    p.add_argument("--dense_depth_supervision", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable target-frame dense depth supervision for --exp=bevdetocc_lidar.")
    p.add_argument("--dense_depth_loss_weight", type=float, default=0.3,
                   help="Weight for target-frame continuous dense depth loss.")
    p.add_argument("--dense_depth_si_weight", type=float, default=0.05,
                   help="Scale-invariant log-depth term weight inside dense depth loss.")
    p.add_argument("--dense_depth_min", type=float, default=1.0,
                   help="Minimum valid dense depth in meters.")
    p.add_argument("--dense_depth_max", type=float, default=80.0,
                   help="Maximum valid dense depth in meters.")
    p.add_argument("--shared_geometry_adapter", action=argparse.BooleanOptionalAction, default=None,
                   help="Enable the shared geometry adapter. Defaults to --dense_depth_supervision.")
    p.add_argument("--geometry_channels", type=int, default=256,
                   help="Shared geometry adapter channel width.")
    p.add_argument("--fusion_attn_type", choices=["self", "cross"], default="self",
                   help="LiDAR/image fusion interaction for --exp=monoscene_lidar. "
                        "'self' uses image+voxel window self-attention; 'cross' "
                        "keeps the original image-query/voxel-KV cross-attention.")
    p.add_argument("--fusion3d", action="store_true",
                   help="Enable the extra 3D sorted image/voxel self-attention "
                        "fusion after the 2D LiDAR/image fusion block.")
    p.add_argument("--fusion3d_seq_len", type=int, default=80,
                   help="Fixed chunk length for --fusion3d sorted self-attention.")
    p.add_argument("--fusion3d_num_heads", type=int, default=None,
                   help="Attention heads for --fusion3d. Defaults to the 2D fusion head count.")
    p.add_argument("--fusion3d_ffn_ratio", type=float, default=2.0)
    p.add_argument("--fusion3d_alpha_init", type=float, default=0.0,
                   help="Initial residual gate value for --fusion3d.")
    p.add_argument("--post_lift_lidar", action="store_true",
                   help="Enable dense target-grid LiDAR VFE fusion after lifting. "
                        "This keeps the 2D LiDAR/image fusion path, appends "
                        "configured LiDAR channels plus mask/count, and fuses "
                        "back to the original lifted feature width before the adapter.")
    p.add_argument("--post_lift_lidar_channels", type=int, default=32,
                   help="Dense post-lift LiDAR feature channels before mask/count channels.")
    p.add_argument("--memory_voxel", action="store_true",
                   help="Enable 3D memory voxel fusion: per-frame VFE produces a "
                        "dense voxel volume per frame, historical frames are "
                        "warped to the reference frame and max-pooled, then "
                        "natten 3D cross-attention attends V_post_fuse (Q) to "
                        "the memory volume (KV). Identity at init (alpha=0).")
    p.add_argument("--memory_voxel_kernel", type=int, default=7,
                   help="3D neighborhood kernel size for --memory_voxel NA cross-attn.")
    p.add_argument("--memory_voxel_num_heads", type=int, default=4,
                   help="Attention heads for --memory_voxel NA cross-attn.")
    p.add_argument("--memory_voxel_num_layers", type=int, default=2,
                   help="Number of stacked NA cross-attn blocks in --memory_voxel.")
    p.add_argument("--memory_voxel_ffn_ratio", type=float, default=2.0,
                   help="FFN hidden ratio inside each --memory_voxel NA block.")
    p.add_argument("--memory_voxel_alpha_init", type=float, default=0.0,
                   help="Initial residual gate value for --memory_voxel.")
    p.add_argument("--memory_voxel_d_voxel", type=int, default=128,
                   help="Per-frame VFE point-MLP output dim for --memory_voxel. "
                        "Must equal post_lift_lidar_d_voxel when both are enabled.")
    # Whether to convert BatchNorm layers to SyncBatchNorm under DDP. "auto"
    # (default) turns it on for the monoscene head (which is BN-heavy and
    # otherwise broken at per-GPU bs=1) and leaves the light head alone (it
    # uses GroupNorm). "on" forces it; "off" disables it everywhere.
    p.add_argument("--syncbn", choices=["auto", "on", "off"], default="auto")
    return p


def _build_dataset(args, split: str) -> Kitti5FrameStage1Dataset:
    common = dict(
        processed_root=args.processed_root,
        split=split,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        output_resolution=(args.width, args.height),
        cam_idx=0,
        load_dense_depth=(
            args.exp == "bevdetocc_lidar"
            and bool(getattr(args, "dense_depth_supervision", False))
        ),
    )
    if args.exp == "bevdetocc_lidar":
        if not args.velodyne_root:
            raise ValueError("--velodyne_root is required when --exp=bevdetocc_lidar")
        return Kitti5FrameStage1LidarDataset(
            velodyne_root=args.velodyne_root,
            max_points_per_sweep=args.max_points_per_sweep,
            **common,
        )
    if args.exp == "monoscene_lidar":
        if not args.velodyne_root:
            raise ValueError("--velodyne_root is required when --exp=monoscene_lidar")
        return Kitti5FrameStage1MonoLidarDataset(
            velodyne_root=args.velodyne_root,
            max_points_per_sweep=args.max_points_per_sweep,
            **common,
        )
    if args.exp == "monoscene":
        return Kitti5FrameStage1MonoDataset(**common)
    return Kitti5FrameStage1Dataset(**common)


def _collate_fn(args):
    if args.exp == "bevdetocc_lidar":
        return collate_stage1_lidar
    if args.exp == "monoscene_lidar":
        return collate_stage1_mono_lidar
    if args.exp == "monoscene":
        return collate_stage1_mono
    return collate_stage1


def _build_loader(args, dataset: Kitti5FrameStage1Dataset, train: bool) -> DataLoader:
    if args.distributed:
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=misc.get_world_size(),
            rank=misc.get_rank(),
            shuffle=train,
            drop_last=train,
        )
    else:
        sampler = (
            torch.utils.data.RandomSampler(dataset)
            if train
            else torch.utils.data.SequentialSampler(dataset)
        )
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=train,
        collate_fn=_collate_fn(args),
    )


def _state_dict_hash(state_dict: Dict[str, torch.Tensor]) -> str:
    """Stable short fingerprint of a state_dict (key names + tensor bytes)."""
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        h.update(k.encode("utf-8"))
        if isinstance(v, torch.Tensor):
            h.update(v.detach().cpu().contiguous().view(-1).to(torch.float32).numpy().tobytes())
    return h.hexdigest()[:16]


def _state_dict_without_backbone(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Checkpoint only trainable/non-frozen modules for the BEVDet-OCC branch."""
    return {
        k: v
        for k, v in model.state_dict().items()
        if not k.startswith("backbone.")
    }


def _move_views_to_device(views: List[Dict[str, torch.Tensor]], device: torch.device):
    moved: List[Dict[str, torch.Tensor]] = []
    for v in views:
        d: Dict[str, torch.Tensor] = {}
        for k, val in v.items():
            if isinstance(val, torch.Tensor):
                d[k] = val.to(device, non_blocking=True)
            else:
                d[k] = val
        moved.append(d)
    return moved


def _move_points_to_device(
    points_per_frame: List[List[torch.Tensor]], device: torch.device
) -> List[List[torch.Tensor]]:
    """Recursively move the variable-length LiDAR sweeps onto device."""
    return [
        [pts.to(device, non_blocking=True) for pts in per_sample]
        for per_sample in points_per_frame
    ]


def _model_forward(model: nn.Module, batch: Dict, device: torch.device, args):
    """Dispatch the model forward to match each experiment's signature."""
    views = _move_views_to_device(batch["views"], device)
    T_target_from_refcam = batch["T_target_from_refcam"].to(device, non_blocking=True)
    if args.exp == "bevdetocc_lidar":
        return_dense_depth = bool(getattr(args, "dense_depth_supervision", False))
        return model(
            views,
            T_target_from_refcam,
            _move_points_to_device(batch["points_per_frame"], device),
            batch["T_cam_from_velo"].to(device, non_blocking=True),
            batch["K_per_frame"].to(device, non_blocking=True),
            batch["image_hw"].to(device, non_blocking=True),
            return_depth=bool(getattr(args, "depth_supervision", False)),
            dense_depth_gt=views[0].get("dense_depth") if return_dense_depth else None,
            has_dense_depth=views[0].get("has_dense_depth") if return_dense_depth else None,
            return_dense_depth=return_dense_depth,
        )
    if args.exp == "monoscene_lidar":
        return model(
            views,
            T_target_from_refcam,
            _move_points_to_device(batch["points_per_frame"], device),
            batch["T_cam_from_velo"].to(device, non_blocking=True),
            batch["K_per_frame"].to(device, non_blocking=True),
            batch["image_hw"].to(device, non_blocking=True),
        )
    return model(views, T_target_from_refcam)


def _maybe_add_bevdet_depth_loss(
    loss: torch.Tensor,
    details: Dict[str, float],
    out: Dict,
    args,
) -> tuple[torch.Tensor, Dict[str, float]]:
    if args.exp != "bevdetocc_lidar" or not bool(getattr(args, "depth_supervision", False)):
        return loss, details
    if "depth_logits" not in out or "gt_depth" not in out:
        raise RuntimeError(
            "BEVDetOcc depth supervision expected model output to contain "
            "'depth_logits' and 'gt_depth'."
        )
    depth_weighted, depth_raw, depth_valid = bevdet_depth_loss(
        out["depth_logits"],
        out["gt_depth"],
        depth_start=float(out.get("depth_start", 1.0)),
        depth_step=float(out.get("depth_step", 0.4)),
        loss_weight=float(args.depth_loss_weight),
    )
    details = dict(details)
    details["depth"] = float(depth_raw.detach())
    details["depth_weighted"] = float(depth_weighted.detach())
    details["depth_valid"] = float(depth_valid.detach())
    return loss + depth_weighted, details


def _maybe_add_dense_depth_loss(
    loss: torch.Tensor,
    details: Dict[str, float],
    out: Dict,
    args,
) -> tuple[torch.Tensor, Dict[str, float]]:
    if args.exp != "bevdetocc_lidar" or not bool(getattr(args, "dense_depth_supervision", False)):
        return loss, details
    required = ("pred_dense_depth", "dense_depth_gt", "has_dense_depth")
    missing = [k for k in required if k not in out]
    if missing:
        raise RuntimeError(
            "Dense depth supervision expected model output keys "
            f"{required}, missing={missing}."
        )
    dense_weighted, dense_log, dense_si, dense_valid = dense_depth_loss(
        out["pred_dense_depth"],
        out["dense_depth_gt"],
        has_dense_depth=out["has_dense_depth"],
        min_depth=float(args.dense_depth_min),
        max_depth=float(args.dense_depth_max),
        loss_weight=float(args.dense_depth_loss_weight),
        si_weight=float(args.dense_depth_si_weight),
        target_index=0,
    )
    details = dict(details)
    details["dense_depth_log"] = float(dense_log.detach())
    details["dense_depth_si"] = float(dense_si.detach())
    details["dense_depth_weighted"] = float(dense_weighted.detach())
    details["dense_depth_valid"] = float(dense_valid.detach())
    return loss + dense_weighted, details


def _sanitize_metric_key(name: str) -> str:
    return str(name).replace(" ", "_").replace("-", "_").replace("/", "_")


def _float_list(values) -> List[float]:
    return [float(v) for v in values]


def _per_class_iou_dict(stats: Dict) -> Dict[str, float]:
    names = stats.get("class_names", [])
    values = stats.get("iou_per_class", [])
    return {str(name): float(iou) for name, iou in zip(names, values)}


def _build_log_stats(epoch: int, train_stats: Dict, val_stats: Dict) -> Dict:
    log_stats = dict(
        epoch=epoch,
        **{f"train_{k}": v for k, v in train_stats.items()},
    )
    for k, v in val_stats.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            log_stats[f"val_{k}"] = float(v)

    if "class_names" in val_stats and "iou_per_class" in val_stats:
        per_class_iou = _per_class_iou_dict(val_stats)
        log_stats["val_class_names"] = [str(name) for name in val_stats["class_names"]]
        log_stats["val_iou_per_class"] = _float_list(val_stats["iou_per_class"])
        log_stats["val_iou_per_class_by_name"] = per_class_iou
        for class_name, iou in per_class_iou.items():
            log_stats[f"val_iou_class_{_sanitize_metric_key(class_name)}"] = iou

    return log_stats


def _adjust_lr(optimizer, epoch_f: float, args) -> float:
    if epoch_f < args.warmup_epochs:
        lr = args.lr * epoch_f / max(args.warmup_epochs, 1)
    else:
        progress = (epoch_f - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr * pg.get("lr_scale", 1.0)
    return lr


def train_one_epoch(
    model: Stage1SSCModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_scaler: NativeScaler,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    args,
    log_writer,
):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch [{epoch}]"
    accum = args.accum_iter
    optimizer.zero_grad()

    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    for step, batch in enumerate(metric_logger.log_every(loader, args.print_freq, header)):
        epoch_f = epoch + step / max(len(loader), 1)
        if step % accum == 0:
            _adjust_lr(optimizer, epoch_f, args)

        target = batch["voxel_label"].to(device, non_blocking=True)

        amp_dtype = (
            torch.bfloat16 if args.amp == "bf16" else (torch.float16 if args.amp == "fp16" else None)
        )
        ctx = (
            torch.autocast("cuda", dtype=amp_dtype)
            if amp_dtype is not None
            else torch.autocast("cuda", enabled=False)
        )

        with ctx:
            out = _model_forward(model, batch, device, args)
            if args.exp in MONOSCENE_LOSS_EXPS:
                cp = batch["CP_mega_matrix"].to(device, non_blocking=True)
                loss, details = criterion(out, target, cp)
            elif args.exp == "bevdetocc_lidar":
                loss, details = criterion(out["ssc_logit"], target)
                loss, details = _maybe_add_bevdet_depth_loss(loss, details, out, args)
                loss, details = _maybe_add_dense_depth_loss(loss, details, out, args)
            else:
                loss, details = criterion(out, target)
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss is {loss_value}; details={details}; stopping.")
        loss = loss / accum

        loss_scaler(
            loss,
            optimizer,
            parameters=[p for p in model.parameters() if p.requires_grad],
            update_grad=(step + 1) % accum == 0,
        )
        if (step + 1) % accum == 0:
            optimizer.zero_grad()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss_value, lr=lr, **details)

        if log_writer is not None and (step + 1) % accum == 0:
            it = epoch * len(loader) + step
            log_writer.add_scalar("train/loss", loss_value, it)
            log_writer.add_scalar("train/lr", lr, it)
            for k, v in details.items():
                log_writer.add_scalar(f"train/{k}", v, it)

    metric_logger.synchronize_between_processes()
    print("Train averaged stats:", metric_logger)
    return {k: m.global_avg for k, m in metric_logger.meters.items()}


@torch.no_grad()
def eval_one_epoch(
    model: Stage1SSCModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    args,
    log_writer,
):
    model.eval()
    ssc = SSCMetrics(
        n_classes=20,
        class_names=list(KITTI_SSC_CLASS_NAMES),
        other_class=None,
        ignore_other_class_in_mIoU=False,
        empty_class=0,
    )
    losses_total = 0.0
    details_total: Dict[str, float] = {}
    n_batches = 0
    amp_dtype = (
        torch.bfloat16 if args.amp == "bf16" else (torch.float16 if args.amp == "fp16" else None)
    )
    for batch in loader:
        target = batch["voxel_label"].to(device, non_blocking=True)

        ctx = (
            torch.autocast("cuda", dtype=amp_dtype)
            if amp_dtype is not None
            else torch.autocast("cuda", enabled=False)
        )
        with ctx:
            out = _model_forward(model, batch, device, args)
            if args.exp in MONOSCENE_LOSS_EXPS:
                cp = batch["CP_mega_matrix"].to(device, non_blocking=True)
                loss, _details = criterion(out, target, cp)
                logits = out["ssc_logit"]
            elif args.exp == "bevdetocc_lidar":
                loss, _details = criterion(out["ssc_logit"], target)
                loss, _details = _maybe_add_bevdet_depth_loss(loss, _details, out, args)
                loss, _details = _maybe_add_dense_depth_loss(loss, _details, out, args)
                logits = out["ssc_logit"]
            else:
                loss, _details = criterion(out, target)
                logits = out
        losses_total += float(loss.detach())
        for k, v in _details.items():
            details_total[k] = details_total.get(k, 0.0) + float(v)
        n_batches += 1

        pred = logits.argmax(dim=1).cpu().numpy()
        gt = target.cpu().numpy()
        ssc.add_batch(pred.astype(np.int64), gt.astype(np.int64))

    # DDP: all-reduce loss + SSC counters across ranks before computing stats.
    if getattr(args, "distributed", False):
        import torch.distributed as dist

        def _reduce_np(arr):
            t = torch.as_tensor(arr, device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            return t.cpu().numpy().astype(arr.dtype)

        def _reduce_scalar(x):
            t = torch.tensor([float(x)], device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            return t.item()

        losses_total = _reduce_scalar(losses_total)
        for k in sorted(details_total):
            details_total[k] = _reduce_scalar(details_total[k])
        n_batches = int(_reduce_scalar(n_batches))
        ssc.tps = _reduce_np(ssc.tps)
        ssc.fps = _reduce_np(ssc.fps)
        ssc.fns = _reduce_np(ssc.fns)
        ssc.completion_tp = _reduce_scalar(ssc.completion_tp)
        ssc.completion_fp = _reduce_scalar(ssc.completion_fp)
        ssc.completion_fn = _reduce_scalar(ssc.completion_fn)

    stats = ssc.get_stats()
    loss_avg = losses_total / max(n_batches, 1)
    detail_avgs = {k: v / max(n_batches, 1) for k, v in details_total.items()}
    if log_writer is not None:
        it = 1000 * epoch
        log_writer.add_scalar("val/loss", loss_avg, it)
        for k, v in detail_avgs.items():
            log_writer.add_scalar(f"val/{k}", v, it)
        log_writer.add_scalar("val/iou", stats["iou"], it)
        log_writer.add_scalar("val/mIoU", stats["mIoU"], it)
        log_writer.add_scalar("val/precision", stats["precision"], it)
        log_writer.add_scalar("val/recall", stats["recall"], it)
        for class_name, iou in _per_class_iou_dict(stats).items():
            log_writer.add_scalar(f"val/iou_class/{class_name}", iou, it)
    print(
        f"Val [{epoch}] loss={loss_avg:.4f} IoU={stats['iou']*100:.2f} "
        f"mIoU={stats['mIoU']*100:.2f} P={stats['precision']*100:.2f} "
        f"R={stats['recall']*100:.2f}"
    )
    return dict(loss=loss_avg, **detail_avgs, **stats)


def main():
    args = get_args_parser().parse_args()
    misc.init_distributed_mode_jz(args)
    # xFormers C++ ext is unavailable on this cu126 env; fall back to PyTorch SDPA.
    toggle_memory_efficient_attention(enabled=False)
    register_legacy_checkpoint_modules()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.amp == "bf16":
        backbone_dtype = torch.bfloat16
    elif args.amp == "fp16":
        backbone_dtype = torch.float16
    else:
        backbone_dtype = torch.float32

    if args.exp == "bevdetocc_lidar":
        model_cls = Stage1SSCBEVDetOccLidarModel
    elif args.exp == "monoscene_lidar":
        model_cls = Stage1SSCMonoLidarModel
    elif args.exp == "monoscene":
        model_cls = Stage1SSCMonoModel
    else:
        model_cls = Stage1SSCModel
    model_kwargs = dict(
        occany_ckpt=args.occany_ckpt,
        c_lift=args.c_lift,
        num_classes=20,
        patch_size=args.patch_size,
        token_dim=args.token_dim,
        backbone_img_size=(args.height, args.width),
        backbone_dtype=backbone_dtype,
    )
    if args.exp == "bevdetocc_lidar":
        shared_geometry_enabled = args.shared_geometry_adapter
        if shared_geometry_enabled is None:
            shared_geometry_enabled = bool(args.dense_depth_supervision)
        if bool(args.dense_depth_supervision) and not bool(shared_geometry_enabled):
            raise ValueError(
                "--dense_depth_supervision requires --shared_geometry_adapter; "
                "otherwise dense depth would be a disconnected side branch."
            )
        args.shared_geometry_adapter = bool(shared_geometry_enabled)
        model_kwargs["fusion_attn_type"] = "cross"
        model_kwargs["num_frames"] = args.num_frames
        model_kwargs["use_shared_geometry_adapter"] = args.shared_geometry_adapter
        model_kwargs["geometry_channels"] = args.geometry_channels
    if args.exp == "monoscene_lidar":
        model_kwargs["fusion_attn_type"] = args.fusion_attn_type
        model_kwargs["fusion3d_enabled"] = args.fusion3d
        model_kwargs["fusion3d_seq_len"] = args.fusion3d_seq_len
        model_kwargs["fusion3d_num_heads"] = args.fusion3d_num_heads
        model_kwargs["fusion3d_ffn_ratio"] = args.fusion3d_ffn_ratio
        model_kwargs["fusion3d_alpha_init"] = args.fusion3d_alpha_init
        model_kwargs["post_lift_lidar_enabled"] = args.post_lift_lidar
        model_kwargs["post_lift_lidar_channels"] = args.post_lift_lidar_channels
        model_kwargs["memory_voxel_enabled"] = args.memory_voxel
        model_kwargs["memory_voxel_kernel"] = args.memory_voxel_kernel
        model_kwargs["memory_voxel_num_heads"] = args.memory_voxel_num_heads
        model_kwargs["memory_voxel_num_layers"] = args.memory_voxel_num_layers
        model_kwargs["memory_voxel_ffn_ratio"] = args.memory_voxel_ffn_ratio
        model_kwargs["memory_voxel_alpha_init"] = args.memory_voxel_alpha_init
        model_kwargs["memory_voxel_d_voxel"] = args.memory_voxel_d_voxel
        model_kwargs["num_frames"] = args.num_frames
    model = model_cls(**model_kwargs).to(device)
    print(f"[exp={args.exp}] using {model_cls.__name__}")
    if args.exp == "bevdetocc_lidar":
        print("[fusion] attn_type=cross (forced for BEVDet-OCC LiDAR branch)")
        print("[backend] LSS half-grid -> LiDAR memory -> NATTEN -> BEVDet CustomResNet3D/LSSFPN3D")
        print(
            f"[depth_supervision] enabled={args.depth_supervision} "
            f"weight={args.depth_loss_weight}"
        )
        print(
            f"[dense_depth] enabled={args.dense_depth_supervision} "
            f"weight={args.dense_depth_loss_weight} "
            f"range=[{args.dense_depth_min}, {args.dense_depth_max}] "
            f"shared_geometry_adapter={model_kwargs.get('use_shared_geometry_adapter', False)} "
            f"geometry_channels={args.geometry_channels}"
        )
    if args.exp == "monoscene_lidar":
        print(f"[fusion] attn_type={args.fusion_attn_type}")
        print(
            f"[fusion3d] enabled={args.fusion3d} "
            f"seq_len={args.fusion3d_seq_len} alpha_init={args.fusion3d_alpha_init}"
        )
        print(
            f"[post_lift_lidar] enabled={args.post_lift_lidar} "
            f"channels={args.post_lift_lidar_channels}"
        )
        print(
            f"[memory_voxel] enabled={args.memory_voxel} "
            f"kernel={args.memory_voxel_kernel} "
            f"heads={args.memory_voxel_num_heads} "
            f"layers={args.memory_voxel_num_layers} "
            f"alpha_init={args.memory_voxel_alpha_init}"
        )

    # Freeze the OccAny backbone in every variant (light / monoscene /
    # monoscene_lidar). Trainable params:
    #   light:            lifting + occ_head
    #   monoscene:        lifting + occ_head (incl. monoscene adapter)
    #   monoscene_lidar:  lifting + occ_head + fusion
    #                      (+ optional sorted-3D self-attention / post-lift VFE)
    for p in model.backbone.parameters():
        p.requires_grad = False

    backbone_hash = _state_dict_hash(model.backbone.state_dict())
    print(f"Backbone state_dict hash: {backbone_hash}")

    # Convert BN -> SyncBN before DDP wrap. At per-GPU bs=1 the MonoScene
    # head's BatchNorm3d layers see one-sample stats per forward, which both
    # makes training trivially fit the single sample and accumulates noisy
    # running stats -- exactly the train-down/val-up divergence we observed
    # without this step. Light head uses GroupNorm so it's unaffected.
    syncbn_on = args.distributed and (
        args.syncbn == "on"
        or (args.syncbn == "auto" and args.exp in ("monoscene", "monoscene_lidar", "bevdetocc_lidar"))
    )
    if syncbn_on:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print(f"[syncbn] converted BatchNorm -> SyncBatchNorm (mode={args.syncbn}).")
    else:
        print(f"[syncbn] disabled (mode={args.syncbn}, distributed={args.distributed}).")

    if args.distributed:
        # monoscene_lidar's fusion sub-modules may not fire on a batch with no
        # valid voxel projections, and the per-sample point cloud changes which
        # windows are active — both incompatible with static_graph=True.
        ddp_kwargs = dict(device_ids=[args.gpu])
        if args.exp in LIDAR_EXPS:
            ddp_kwargs.update(find_unused_parameters=True, static_graph=False)
        else:
            ddp_kwargs.update(find_unused_parameters=False, static_graph=True)
        model = nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
        model_to_save = model.module
    else:
        model_to_save = model

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    train_count = sum(p.numel() for p in trainable_params)
    print(f"Trainable params: {train_count:,} / {total_params:,}")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    loss_scaler = NativeScaler(enabled=(args.amp == "fp16"))
    if args.exp in MONOSCENE_LOSS_EXPS:
        criterion = MonoSceneSSCLoss().to(device)
    else:
        criterion = SSCLoss().to(device)

    train_dataset = _build_dataset(args, "train")
    val_dataset = _build_dataset(args, "val")
    print(f"Train samples: {len(train_dataset)}; Val samples: {len(val_dataset)}")
    train_loader = _build_loader(args, train_dataset, train=True)
    val_loader = _build_loader(args, val_dataset, train=False)

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        prev_hash = ckpt.get("backbone_hash", None)
        if prev_hash is not None and prev_hash != backbone_hash:
            raise RuntimeError(
                f"Backbone state_dict hash mismatch on resume: "
                f"checkpoint expected {prev_hash}, current --occany_ckpt gives {backbone_hash}. "
                "Refusing to resume — the frozen backbone has changed."
            )
        if args.exp == "bevdetocc_lidar" and "model" in ckpt:
            status = model_to_save.load_state_dict(ckpt["model"], strict=False)
            print(
                "[resume:bevdetocc_lidar] loaded non-backbone model state: "
                f"missing={len(status.missing_keys)} unexpected={len(status.unexpected_keys)}"
            )
        elif args.exp != "bevdetocc_lidar" and "lifting" in ckpt:
            model_to_save.lifting.load_state_dict(ckpt["lifting"], strict=False)
        if args.exp != "bevdetocc_lidar" and "occ_head" in ckpt:
            model_to_save.occ_head.load_state_dict(ckpt["occ_head"], strict=False)
        if args.exp != "bevdetocc_lidar" and "fusion" in ckpt and hasattr(model_to_save, "fusion"):
            model_to_save.fusion.load_state_dict(ckpt["fusion"], strict=False)
        if (
            "post_lift_lidar" in ckpt
            and getattr(model_to_save, "post_lift_lidar", None) is not None
        ):
            model_to_save.post_lift_lidar.load_state_dict(
                ckpt["post_lift_lidar"], strict=False
            )
        if (
            "post_lift_fuse" in ckpt
            and getattr(model_to_save, "post_lift_fuse", None) is not None
        ):
            model_to_save.post_lift_fuse.load_state_dict(
                ckpt["post_lift_fuse"], strict=False
            )
        if (
            "memory_fusion" in ckpt
            and getattr(model_to_save, "memory_fusion", None) is not None
        ):
            model_to_save.memory_fusion.load_state_dict(
                ckpt["memory_fusion"], strict=False
            )
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print("Optimizer load failed:", e)
        if "scaler" in ckpt:
            loss_scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"Resumed from {args.resume}; epoch={start_epoch}")

    log_writer = (
        SummaryWriter(log_dir=args.output_dir) if misc.is_main_process() else None
    )

    if args.eval_only:
        eval_one_epoch(model, val_loader, criterion, device, start_epoch, args, log_writer)
        return

    print(f"Start training: epochs={args.epochs}, start_epoch={start_epoch}")
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, loss_scaler, criterion, device, epoch, args, log_writer
        )

        val_stats: Dict[str, float] = {}
        if (epoch + 1) % args.eval_freq == 0:
            val_stats = eval_one_epoch(
                model, val_loader, criterion, device, epoch, args, log_writer
            )

        if misc.is_main_process():
            if args.exp == "bevdetocc_lidar":
                ckpt_payload = {
                    "model": _state_dict_without_backbone(model_to_save),
                    "optimizer": optimizer.state_dict(),
                    "scaler": loss_scaler.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                    "backbone_hash": backbone_hash,
                }
            else:
                ckpt_payload = {
                    "lifting": model_to_save.lifting.state_dict(),
                    "occ_head": model_to_save.occ_head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": loss_scaler.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                    "backbone_hash": backbone_hash,
                }
                if args.exp == "monoscene_lidar":
                    ckpt_payload["fusion"] = model_to_save.fusion.state_dict()
                    if model_to_save.post_lift_lidar is not None:
                        ckpt_payload["post_lift_lidar"] = (
                            model_to_save.post_lift_lidar.state_dict()
                        )
                        ckpt_payload["post_lift_fuse"] = (
                            model_to_save.post_lift_fuse.state_dict()
                        )
                    if model_to_save.memory_fusion is not None:
                        ckpt_payload["memory_fusion"] = (
                            model_to_save.memory_fusion.state_dict()
                        )
            if (epoch + 1) % args.save_freq == 0:
                torch.save(ckpt_payload, os.path.join(args.output_dir, "checkpoint-last.pth"))
            if args.keep_freq and (epoch + 1) % args.keep_freq == 0:
                if args.exp == "bevdetocc_lidar":
                    keep_payload = {
                        "model": ckpt_payload["model"],
                        "epoch": epoch,
                        "backbone_hash": backbone_hash,
                    }
                else:
                    keep_payload = {
                        "lifting": ckpt_payload["lifting"],
                        "occ_head": ckpt_payload["occ_head"],
                        "epoch": epoch,
                        "backbone_hash": backbone_hash,
                    }
                    if "fusion" in ckpt_payload:
                        keep_payload["fusion"] = ckpt_payload["fusion"]
                    if "post_lift_lidar" in ckpt_payload:
                        keep_payload["post_lift_lidar"] = ckpt_payload["post_lift_lidar"]
                    if "post_lift_fuse" in ckpt_payload:
                        keep_payload["post_lift_fuse"] = ckpt_payload["post_lift_fuse"]
                    if "memory_fusion" in ckpt_payload:
                        keep_payload["memory_fusion"] = ckpt_payload["memory_fusion"]
                torch.save(
                    keep_payload,
                    os.path.join(args.output_dir, f"checkpoint-{epoch}.pth"),
                )
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(_build_log_stats(epoch, train_stats, val_stats)) + "\n")

    dt = str(datetime.timedelta(seconds=int(time.time() - t0)))
    print(f"Done. Total time: {dt}")


if __name__ == "__main__":
    main()
