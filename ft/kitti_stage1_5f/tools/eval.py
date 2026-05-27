"""Distributed evaluation entry for Stage-1 SSC checkpoints.

Example:
  torchrun --nproc_per_node=4 -m ft.kitti_stage1_5f.tools.eval \
    --ckpt /path/to/model_param_dir/checkpoint-last.pth \
    --model_type monoscene_lidar
"""
from __future__ import annotations

from .. import _paths  # noqa: F401  (must come first)

import argparse
import datetime
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc

from occany.model.must3r_blocks.attention import toggle_memory_efficient_attention
from occany.utils.checkpoint_io import register_legacy_checkpoint_modules

from ft.semantickitti_ft.losses import SSCLoss
from ..losses_monoscene import MonoSceneSSCLoss
from ..models import Stage1SSCModel, Stage1SSCMonoLidarModel, Stage1SSCMonoModel
from .train import (
    _build_dataset,
    _build_loader,
    _build_log_stats,
    _state_dict_hash,
    eval_one_epoch,
)


def get_args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("OccAny Stage-1 SSC eval", add_help=True)
    p.add_argument(
        "--ckpt",
        required=True,
        type=str,
        help="Fine-tuned checkpoint file, or a directory containing checkpoint-last.pth.",
    )
    p.add_argument(
        "--model_type",
        "--exp",
        dest="exp",
        choices=["light", "monoscene", "monoscene_lidar"],
        default=None,
        help="Model variant. If omitted, read from checkpoint args.",
    )

    p.add_argument("--processed_root", default=None, type=str)
    p.add_argument("--kittiodo_root", default=None, type=str)
    p.add_argument("--occany_ckpt", default=None, type=str)
    p.add_argument("--velodyne_root", default=None, type=str)
    p.add_argument("--fusion_attn_type", choices=["self", "cross"], default=None)
    p.add_argument("--fusion3d", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--fusion3d_seq_len", type=int, default=None)
    p.add_argument("--fusion3d_num_heads", type=int, default=None)
    p.add_argument("--fusion3d_ffn_ratio", type=float, default=None)
    p.add_argument("--fusion3d_alpha_init", type=float, default=None)
    p.add_argument("--post_lift_lidar", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--post_lift_lidar_channels", type=int, default=None)
    p.add_argument("--max_points_per_sweep", type=int, default=None)

    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--num_frames", type=int, default=None)
    p.add_argument("--frame_stride", type=int, default=None)
    p.add_argument("--c_lift", type=int, default=None)
    p.add_argument("--token_dim", type=int, default=None)
    p.add_argument("--patch_size", type=int, default=None)

    # Default batch size is per process/GPU under torchrun.
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--amp", choices=["bf16", "fp16", "none"], default=None)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--print_freq", type=int, default=20)

    p.add_argument("--world_size", default=1, type=int)
    p.add_argument("--local_rank", default=-1, type=int)
    p.add_argument("--dist_url", default="env://", type=str)
    p.add_argument("--nodist", action="store_true")

    p.add_argument(
        "--log_name",
        default="eval_log.txt",
        type=str,
        help="JSON-lines log filename written under the checkpoint directory.",
    )
    return p


def _ckpt_arg(ckpt_args: Dict, name: str, default):
    return ckpt_args.get(name, default) if isinstance(ckpt_args, dict) else default


def _override_or_ckpt(args: argparse.Namespace, ckpt_args: Dict, name: str, default):
    value = getattr(args, name)
    return value if value is not None else _ckpt_arg(ckpt_args, name, default)


def _resolve_ckpt_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_dir():
        path = path / "checkpoint-last.pth"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _fill_args_from_checkpoint(args: argparse.Namespace, ckpt_args: Dict) -> None:
    args.exp = args.exp or _ckpt_arg(ckpt_args, "exp", "light")
    args.processed_root = _override_or_ckpt(args, ckpt_args, "processed_root", None)
    args.kittiodo_root = _override_or_ckpt(args, ckpt_args, "kittiodo_root", None)
    args.occany_ckpt = _override_or_ckpt(args, ckpt_args, "occany_ckpt", None)
    args.velodyne_root = _override_or_ckpt(args, ckpt_args, "velodyne_root", None)
    args.fusion_attn_type = _override_or_ckpt(
        args, ckpt_args, "fusion_attn_type", "self"
    )
    args.fusion3d = bool(_override_or_ckpt(args, ckpt_args, "fusion3d", False))
    args.fusion3d_seq_len = int(
        _override_or_ckpt(args, ckpt_args, "fusion3d_seq_len", 80)
    )
    args.fusion3d_num_heads = _override_or_ckpt(
        args, ckpt_args, "fusion3d_num_heads", None
    )
    if args.fusion3d_num_heads is not None:
        args.fusion3d_num_heads = int(args.fusion3d_num_heads)
    args.fusion3d_ffn_ratio = float(
        _override_or_ckpt(args, ckpt_args, "fusion3d_ffn_ratio", 2.0)
    )
    args.fusion3d_alpha_init = float(
        _override_or_ckpt(args, ckpt_args, "fusion3d_alpha_init", 0.0)
    )
    args.post_lift_lidar = bool(
        _override_or_ckpt(args, ckpt_args, "post_lift_lidar", False)
    )
    args.post_lift_lidar_channels = int(
        _override_or_ckpt(args, ckpt_args, "post_lift_lidar_channels", 32)
    )
    args.max_points_per_sweep = int(
        _override_or_ckpt(args, ckpt_args, "max_points_per_sweep", 0)
    )
    args.width = int(_override_or_ckpt(args, ckpt_args, "width", 512))
    args.height = int(_override_or_ckpt(args, ckpt_args, "height", 160))
    args.num_frames = int(_override_or_ckpt(args, ckpt_args, "num_frames", 5))
    args.frame_stride = int(_override_or_ckpt(args, ckpt_args, "frame_stride", 1))
    args.c_lift = int(_override_or_ckpt(args, ckpt_args, "c_lift", 64))
    args.token_dim = int(_override_or_ckpt(args, ckpt_args, "token_dim", 768))
    args.patch_size = int(_override_or_ckpt(args, ckpt_args, "patch_size", 16))
    args.num_workers = int(_override_or_ckpt(args, ckpt_args, "num_workers", 4))
    args.amp = args.amp or _ckpt_arg(ckpt_args, "amp", "bf16")

    if not args.processed_root:
        raise ValueError("--processed_root is required when checkpoint args do not contain it.")
    if not args.occany_ckpt:
        raise ValueError("--occany_ckpt is required when checkpoint args do not contain it.")
    if args.exp == "monoscene_lidar" and not args.velodyne_root:
        raise ValueError("--velodyne_root is required for --model_type monoscene_lidar.")


def _build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.amp == "bf16" and device.type == "cuda":
        backbone_dtype = torch.bfloat16
    elif args.amp == "fp16" and device.type == "cuda":
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
        model_kwargs["fusion3d_enabled"] = args.fusion3d
        model_kwargs["fusion3d_seq_len"] = args.fusion3d_seq_len
        model_kwargs["fusion3d_num_heads"] = args.fusion3d_num_heads
        model_kwargs["fusion3d_ffn_ratio"] = args.fusion3d_ffn_ratio
        model_kwargs["fusion3d_alpha_init"] = args.fusion3d_alpha_init
        model_kwargs["post_lift_lidar_enabled"] = args.post_lift_lidar
        model_kwargs["post_lift_lidar_channels"] = args.post_lift_lidar_channels
        model_kwargs["num_frames"] = args.num_frames
    model = model_cls(**model_kwargs).to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False
    return model


def _load_stage1_weights(model: nn.Module, ckpt: Dict, args: argparse.Namespace) -> None:
    if "lifting" not in ckpt or "occ_head" not in ckpt:
        raise KeyError("Checkpoint must contain 'lifting' and 'occ_head' state_dicts.")
    model.lifting.load_state_dict(ckpt["lifting"], strict=True)
    model.occ_head.load_state_dict(ckpt["occ_head"], strict=True)
    if args.exp == "monoscene_lidar":
        if "fusion" not in ckpt:
            raise KeyError("monoscene_lidar checkpoint must contain a 'fusion' state_dict.")
        model.fusion.load_state_dict(ckpt["fusion"], strict=True)
        if args.post_lift_lidar:
            if "post_lift_lidar" not in ckpt:
                raise KeyError(
                    "post-lift LiDAR checkpoint must contain a 'post_lift_lidar' state_dict."
                )
            model.post_lift_lidar.load_state_dict(ckpt["post_lift_lidar"], strict=True)
            if "post_lift_fuse" not in ckpt:
                raise KeyError(
                    "post-lift LiDAR checkpoint must contain a 'post_lift_fuse' state_dict."
                )
            model.post_lift_fuse.load_state_dict(ckpt["post_lift_fuse"], strict=True)


def _log_path_for_ckpt(ckpt_path: Path, log_name: str) -> Path:
    return ckpt_path.parent / log_name


def main() -> None:
    args = get_args_parser().parse_args()
    misc.init_distributed_mode(args)
    toggle_memory_efficient_attention(enabled=False)
    register_legacy_checkpoint_modules()

    ckpt_path = _resolve_ckpt_path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    _fill_args_from_checkpoint(args, ckpt_args)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = _build_model(args, device)
    _load_stage1_weights(model, ckpt, args)
    model.eval()

    backbone_hash = _state_dict_hash(model.backbone.state_dict())
    if misc.is_main_process():
        print(f"[eval] ckpt={ckpt_path}")
        print(f"[eval] model_type={args.exp}; batch_size_per_gpu={args.batch_size}")
        print(f"[eval] backbone_hash={backbone_hash}")

    val_dataset = _build_dataset(args, "val")
    val_loader = _build_loader(args, val_dataset, train=False)
    if misc.is_main_process():
        print(f"[eval] Val samples: {len(val_dataset)}")

    criterion = (
        MonoSceneSSCLoss().to(device)
        if args.exp in ("monoscene", "monoscene_lidar")
        else SSCLoss().to(device)
    )

    t0 = time.time()
    val_stats = eval_one_epoch(model, val_loader, criterion, device, 0, args, None)
    elapsed = str(datetime.timedelta(seconds=int(time.time() - t0)))

    if misc.is_main_process():
        log_stats = _build_log_stats(epoch=0, train_stats={}, val_stats=val_stats)
        log_stats.update(
            {
                "ckpt": str(ckpt_path),
                "model_type": args.exp,
                "batch_size_per_gpu": int(args.batch_size),
                "world_size": int(misc.get_world_size()),
                "backbone_hash": backbone_hash,
                "elapsed": elapsed,
            }
        )
        log_path = _log_path_for_ckpt(ckpt_path, args.log_name)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats, ensure_ascii=False) + "\n")
        print(f"[eval] wrote {log_path}")
        print(f"[eval] done in {elapsed}")


if __name__ == "__main__":
    main()
