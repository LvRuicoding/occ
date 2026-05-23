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
    Kitti5FrameStage1MonoDataset,
    Kitti5FrameStage1MonoLidarDataset,
    collate_stage1,
    collate_stage1_mono,
    collate_stage1_mono_lidar,
)
from ..losses_monoscene import MonoSceneSSCLoss
from ..models import Stage1SSCModel, Stage1SSCMonoModel, Stage1SSCMonoLidarModel


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
    p.add_argument("--exp", choices=["light", "monoscene", "monoscene_lidar"], default="light")
    # LiDAR-fusion-only options.
    p.add_argument("--velodyne_root", default=None, type=str,
                   help="Raw KITTI Odometry root: <velodyne_root>/sequences/<seq>/velodyne/*.bin. "
                        "Required when --exp=monoscene_lidar.")
    p.add_argument("--max_points_per_sweep", type=int, default=0,
                   help="If >0, deterministically stride-subsample each LiDAR sweep to this point count.")
    p.add_argument("--fusion_attn_type", choices=["self", "cross"], default="self",
                   help="LiDAR/image fusion interaction for --exp=monoscene_lidar. "
                        "'self' uses image+voxel window self-attention; 'cross' "
                        "keeps the original image-query/voxel-KV cross-attention.")
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
            if args.exp in ("monoscene", "monoscene_lidar"):
                cp = batch["CP_mega_matrix"].to(device, non_blocking=True)
                loss, details = criterion(out, target, cp)
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
            if args.exp in ("monoscene", "monoscene_lidar"):
                cp = batch["CP_mega_matrix"].to(device, non_blocking=True)
                loss, _details = criterion(out, target, cp)
                logits = out["ssc_logit"]
            else:
                loss, _details = criterion(out, target)
                logits = out
        losses_total += float(loss.detach())
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
        n_batches = int(_reduce_scalar(n_batches))
        ssc.tps = _reduce_np(ssc.tps)
        ssc.fps = _reduce_np(ssc.fps)
        ssc.fns = _reduce_np(ssc.fns)
        ssc.completion_tp = _reduce_scalar(ssc.completion_tp)
        ssc.completion_fp = _reduce_scalar(ssc.completion_fp)
        ssc.completion_fn = _reduce_scalar(ssc.completion_fn)

    stats = ssc.get_stats()
    loss_avg = losses_total / max(n_batches, 1)
    if log_writer is not None:
        it = 1000 * epoch
        log_writer.add_scalar("val/loss", loss_avg, it)
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
    return dict(loss=loss_avg, **stats)


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

    if args.exp == "monoscene_lidar":
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
    if args.exp == "monoscene_lidar":
        model_kwargs["fusion_attn_type"] = args.fusion_attn_type
    model = model_cls(**model_kwargs).to(device)
    print(f"[exp={args.exp}] using {model_cls.__name__}")
    if args.exp == "monoscene_lidar":
        print(f"[fusion] attn_type={args.fusion_attn_type}")

    # Freeze the OccAny backbone in every variant (light / monoscene /
    # monoscene_lidar). Trainable params:
    #   light:            lifting + occ_head
    #   monoscene:        lifting + occ_head (incl. monoscene adapter)
    #   monoscene_lidar:  lifting + occ_head + fusion (VFE + W-MSA/SW-MSA)
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
        or (args.syncbn == "auto" and args.exp in ("monoscene", "monoscene_lidar"))
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
        if args.exp == "monoscene_lidar":
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
    if args.exp in ("monoscene", "monoscene_lidar"):
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
        if "lifting" in ckpt:
            model_to_save.lifting.load_state_dict(ckpt["lifting"], strict=False)
        if "occ_head" in ckpt:
            model_to_save.occ_head.load_state_dict(ckpt["occ_head"], strict=False)
        if "fusion" in ckpt and hasattr(model_to_save, "fusion"):
            model_to_save.fusion.load_state_dict(ckpt["fusion"], strict=False)
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
            if (epoch + 1) % args.save_freq == 0:
                torch.save(ckpt_payload, os.path.join(args.output_dir, "checkpoint-last.pth"))
            if args.keep_freq and (epoch + 1) % args.keep_freq == 0:
                keep_payload = {
                    "lifting": ckpt_payload["lifting"],
                    "occ_head": ckpt_payload["occ_head"],
                    "epoch": epoch,
                    "backbone_hash": backbone_hash,
                }
                if "fusion" in ckpt_payload:
                    keep_payload["fusion"] = ckpt_payload["fusion"]
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
