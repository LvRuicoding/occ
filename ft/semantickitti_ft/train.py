"""Training entry for SemanticKITTI SSC fine-tuning of OccAny.

Loads a frozen OccAny (encoder + decoder + raymap_encoder + gen_decoder),
trains only a MonoSceneOccupancyHead on top of last-frame stereo recon tokens
plus N novel-pose render tokens.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler

from occany.metrics.ssc import SSCMetrics
from occany.model.model_must3r import (
    Dust3rEncoder,
    Must3rDecoder,
    RaymapEncoderDiT,
)
from occany.model.must3r_blocks.attention import toggle_memory_efficient_attention
from occany.model.must3r_blocks.head import ActivationType  # noqa: F401
from occany.utils.checkpoint_io import register_legacy_checkpoint_modules

from ft.semantickitti_ft.dataset import (
    KITTI_SSC_CLASS_NAMES,
    SemanticKittiSSCDataset,
    collate_ssc,
)
from ft.semantickitti_ft.losses import SSCLoss
from ft.semantickitti_ft.heads import available_heads
from ft.semantickitti_ft.lifting import available_lifts
from ft.semantickitti_ft.models import OccAnyOccHead


def get_args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("OccAny SSC fine-tune", add_help=True)
    p.add_argument("--semkitti_root", required=True, type=str)
    p.add_argument("--kittiodo_root", required=True, type=str)
    p.add_argument("--remap_lut_path", default=None, type=str,
                   help="Path to occany/datasets/semantic_kitti.yaml; default=auto.")
    p.add_argument("--occany_ckpt", required=True, type=str,
                   help="Merged OccAny checkpoint (encoder+decoder+raymap+gen).")
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=1)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--frame_stride", type=int, default=5)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=160)
    p.add_argument("--no_render", action="store_true",
                   help="Disable novel-view render-token branch.")
    p.add_argument("--lift", choices=available_lifts(), default="occany_render_tokens",
                   help="Feature lifting module to use.")
    p.add_argument("--head", choices=available_heads(), default="monoscene",
                   help="SSC head module to use.")
    p.add_argument("--n_render_views", type=int, default=4)
    p.add_argument("--n_decoder_feature_layers", type=int, default=4)
    p.add_argument("--occ_feature", type=int, default=64)
    p.add_argument("--occ_project_scale", type=int, default=2)
    p.add_argument("--print_freq", type=int, default=20)
    p.add_argument("--save_freq", type=int, default=1)
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
    return p


def _resolve_remap_lut_path(arg_path: str | None) -> str:
    if arg_path is not None:
        return arg_path
    here = Path(__file__).resolve().parent
    repo = here.parent.parent
    return str(repo / "occany" / "datasets" / "semantic_kitti.yaml")


def _build_dataset(args, split: str) -> SemanticKittiSSCDataset:
    return SemanticKittiSSCDataset(
        semkitti_root=args.semkitti_root,
        kittiodo_root=args.kittiodo_root,
        remap_lut_path=_resolve_remap_lut_path(args.remap_lut_path),
        split=split,
        frame_stride=args.frame_stride,
        output_resolution=(args.width, args.height),
    )


def _build_loader(args, dataset: SemanticKittiSSCDataset, train: bool) -> DataLoader:
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
        collate_fn=collate_ssc,
    )


def _load_backbone(args, device: torch.device):
    img_encoder = Dust3rEncoder()
    decoder = Must3rDecoder(
        img_size=(args.width, args.width),
        enc_embed_dim=1024,
        embed_dim=768,
        pointmaps_activation=ActivationType.LINEAR,
        pred_sam_features=True,
        feedback_type="single_mlp",
        memory_mode="kv",
        ray_map_encoder_depth=6,
        use_multitask_token=True,
    )
    raymap_encoder = (
        None
        if args.no_render
        else RaymapEncoderDiT(use_time_cond=False)
    )
    gen_decoder = (
        None
        if args.no_render
        else Must3rDecoder(
            img_size=(args.width, args.width),
            enc_embed_dim=1024,
            embed_dim=768,
            pointmaps_activation=ActivationType.LINEAR,
            pred_sam_features=True,
            feedback_type="single_mlp",
            memory_mode="kv",
            ray_map_encoder_depth=6,
            use_multitask_token=True,
        )
    )
    img_encoder.to(device)
    decoder.to(device)
    if raymap_encoder is not None:
        raymap_encoder.to(device)
    if gen_decoder is not None:
        gen_decoder.to(device)

    print(f"Loading OccAny checkpoint: {args.occany_ckpt}")
    ckpt = torch.load(args.occany_ckpt, map_location="cpu", weights_only=False)
    enc_status = img_encoder.load_state_dict(ckpt.get("encoder", {}), strict=False)
    print("encoder load status:", enc_status)
    dec_status = decoder.load_state_dict(ckpt.get("decoder", {}), strict=False)
    print("decoder load status:", dec_status)
    if raymap_encoder is not None and "raymap_encoder" in ckpt:
        rs = raymap_encoder.load_state_dict(ckpt["raymap_encoder"], strict=False)
        print("raymap_encoder load status:", rs)
    if gen_decoder is not None and "gen_decoder" in ckpt:
        gs = gen_decoder.load_state_dict(ckpt["gen_decoder"], strict=False)
        print("gen_decoder load status:", gs)
    elif gen_decoder is not None:
        # Fall back to recon decoder weights for the gen decoder if a separate
        # gen_decoder is not stored in the checkpoint.
        gs = gen_decoder.load_state_dict(ckpt.get("decoder", {}), strict=False)
        print("gen_decoder (fallback) load status:", gs)
    del ckpt
    return img_encoder, decoder, raymap_encoder, gen_decoder


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


def _adjust_lr(optimizer, epoch_f: float, args) -> float:
    if epoch_f < args.warmup_epochs:
        lr = args.lr * epoch_f / max(args.warmup_epochs, 1)
    else:
        progress = (epoch_f - args.warmup_epochs) / max(
            args.epochs - args.warmup_epochs, 1
        )
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (
            1 + math.cos(math.pi * progress)
        )
    for pg in optimizer.param_groups:
        pg["lr"] = lr * pg.get("lr_scale", 1.0)
    return lr


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_scaler: NativeScaler,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    args,
    log_writer: SummaryWriter | None,
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

        views = _move_views_to_device(batch["views"], device)
        anchor_pose = batch["anchor_pose"].to(device, non_blocking=True)
        lidar_to_world = batch["lidar_to_world"].to(device, non_blocking=True)
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
            logits = model(views, anchor_pose=anchor_pose, lidar_to_world=lidar_to_world)
            loss, details = criterion(logits, target)
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}; details={details}; stopping.")
            sys.exit(1)
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
            it = int(epoch_f * 1000)
            log_writer.add_scalar("train/loss", loss_value, it)
            log_writer.add_scalar("train/lr", lr, it)
            for k, v in details.items():
                log_writer.add_scalar(f"train/{k}", v, it)

    metric_logger.synchronize_between_processes()
    print("Train averaged stats:", metric_logger)
    return {k: m.global_avg for k, m in metric_logger.meters.items()}


@torch.no_grad()
def eval_one_epoch(
    model,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    args,
    log_writer: SummaryWriter | None,
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
    for batch in loader:
        views = _move_views_to_device(batch["views"], device)
        anchor_pose = batch["anchor_pose"].to(device, non_blocking=True)
        lidar_to_world = batch["lidar_to_world"].to(device, non_blocking=True)
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
            logits = model(views, anchor_pose=anchor_pose, lidar_to_world=lidar_to_world)
            loss, _details = criterion(logits, target)
        losses_total += float(loss.detach())
        n_batches += 1

        pred = logits.argmax(dim=1).cpu().numpy()
        gt = target.cpu().numpy()
        ssc.add_batch(pred.astype(np.int64), gt.astype(np.int64))

    stats = ssc.get_stats()
    loss_avg = losses_total / max(n_batches, 1)
    if log_writer is not None:
        it = 1000 * epoch
        log_writer.add_scalar("val/loss", loss_avg, it)
        log_writer.add_scalar("val/iou", stats["iou"], it)
        log_writer.add_scalar("val/mIoU", stats["mIoU"], it)
        log_writer.add_scalar("val/precision", stats["precision"], it)
        log_writer.add_scalar("val/recall", stats["recall"], it)
    print(
        f"Val [{epoch}] loss={loss_avg:.4f} IoU={stats['iou']*100:.2f} "
        f"mIoU={stats['mIoU']*100:.2f} P={stats['precision']*100:.2f} "
        f"R={stats['recall']*100:.2f}"
    )
    return dict(loss=loss_avg, **stats)


def main():
    args = get_args_parser().parse_args()
    misc.init_distributed_mode_jz(args)
    toggle_memory_efficient_attention(enabled=True)
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

    img_encoder, decoder, raymap_encoder, gen_decoder = _load_backbone(args, device)

    point_cloud_range = (0.0, -25.6, -2.0, 51.2, 25.6, 4.4)
    voxel_size = (0.2, 0.2, 0.2)

    model = OccAnyOccHead(
        img_encoder=img_encoder,
        decoder=decoder,
        raymap_encoder=raymap_encoder,
        gen_decoder=gen_decoder,
        num_classes=20,
        feature=args.occ_feature,
        project_scale=args.occ_project_scale,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        n_render_views=(0 if args.no_render else args.n_render_views),
        n_decoder_feature_layers=args.n_decoder_feature_layers,
        last_frame_view_indices=(4, 5),
        token_dim=768,
        patch_size=16,
        pointmaps_activation=ActivationType.LINEAR,
        lift_type=args.lift,
        head_type=args.head,
    ).to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu]
        )
        model_to_save = model.module
    else:
        model_to_save = model

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(
        f"Trainable params: {sum(p.numel() for p in trainable_params):,} / "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    loss_scaler = NativeScaler()
    criterion = SSCLoss().to(device)

    train_dataset = _build_dataset(args, "train")
    val_dataset = _build_dataset(args, "val")
    print(f"Train samples: {len(train_dataset)}; Val samples: {len(val_dataset)}")
    train_loader = _build_loader(args, train_dataset, train=True)
    val_loader = _build_loader(args, val_dataset, train=False)

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_to_save.load_state_dict(ckpt["model"], strict=False)
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
            if (epoch + 1) % args.save_freq == 0:
                torch.save(
                    {
                        "model": model_to_save.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": loss_scaler.state_dict(),
                        "epoch": epoch,
                        "args": vars(args),
                    },
                    os.path.join(args.output_dir, "checkpoint-last.pth"),
                )
            if args.keep_freq and (epoch + 1) % args.keep_freq == 0:
                torch.save(
                    {
                        "model": model_to_save.state_dict(),
                        "epoch": epoch,
                    },
                    os.path.join(args.output_dir, f"checkpoint-{epoch}.pth"),
                )
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(
                    json.dumps(
                        dict(
                            epoch=epoch,
                            **{f"train_{k}": v for k, v in train_stats.items()},
                            **{f"val_{k}": float(v) if isinstance(v, (int, float, np.floating)) else None
                               for k, v in val_stats.items() if not isinstance(v, (list, np.ndarray))},
                        )
                    )
                    + "\n"
                )

    dt = str(datetime.timedelta(seconds=int(time.time() - t0)))
    print(f"Done. Total time: {dt}")


if __name__ == "__main__":
    main()
