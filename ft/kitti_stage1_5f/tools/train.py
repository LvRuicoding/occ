"""Training entry for Stage-1 SSC fine-tuning of OccAny on SemanticKITTI.

Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=4 \
      -m ft.kitti_stage1_5f.tools.train \
      --processed_root /home/dataset-local/lr/code/OccAny/data/kitti_processed \
      --occany_ckpt /home/dataset-local/lr/code/OccAny/checkpoints/occany_recon.pth \
      --output_dir /home/dataset-local/lr/code/OccAny/output/kitti_stage1_5f_4gpu_monoscene_lidar_selfattn \
      --exp monoscene_lidar \
      --fusion_attn_type self \
      --batch_size 1 \
      --num_workers 6 \
      --amp bf16 \
      --epochs 20 \
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
import shlex
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter

import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc
from dust3r.losses import L21
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler

from occany.loss.losses_multiview import ConfLoss_multiview, Regr3D_multiview
from occany.metrics.ssc import SSCMetrics
from occany.model.must3r_blocks.attention import toggle_memory_efficient_attention
from occany.utils.checkpoint_io import register_legacy_checkpoint_modules

from ft.semantickitti_ft.losses import SSCLoss
from ..datasets import (
    DDAD5FrameStage1DenseDepthDataset,
    DDAD5FrameStage1LidarDenseDepthDataset,
    KITTI_SSC_CLASS_NAMES,
    KITTI_OBJECT_CLASS_NAMES,
    UNIFIED_SSC_CLASS_NAMES,
    Kitti5FrameStage1DenseDepthDataset,
    Kitti5FrameStage1Dataset,
    Kitti5FrameStage1LidarDenseDepthDataset,
    Kitti5FrameStage1LidarDataset,
    Kitti5FrameStage1MonoDataset,
    Kitti5FrameStage1MonoLidarDataset,
    KittiObject5FrameDetDataset,
    NuScenes5FrameStage1LidarDataset,
    collate_kitti_object_det,
    collate_stage1,
    collate_stage1_dense_depth,
    collate_stage1_lidar_dense_depth,
    collate_stage1_lidar,
    collate_stage1_mono,
    collate_stage1_mono_lidar,
    collate_stage1_nuscenes_lidar,
    evaluate_lidar_det_ap40,
)
from ..datasets.unified_occ import NUSCENES_TO_UNIFIED
from ..datasets.kitti_object_det import (
    KITTI_OBJECT_DET_DEPTH_BOUND,
    KITTI_OBJECT_DET_PC_RANGE,
    make_kitti_object_det_grid_config,
)
from ..losses_monoscene import MonoSceneSSCLoss
from ..models import (
    Stage1DepthOriginalModel,
    Stage1DepthPostFusionOnlyModel,
    Stage1DepthPromptFusionOnlyModel,
    Stage1DetOriginalModel,
    Stage1DetPostFusionOnlyModel,
    Stage1SSCBEVDetOccLidarDenseDepthModel,
    Stage1SSCBEVDetOccLidarModel,
    Stage1SSCBEVDetOccLidarPointmapModel,
    Stage1PointmapOriginalModel,
    Stage1PointmapPostFusionOnlyModel,
    Stage1SSCBEVDetOccLidarPointmapDenseDepthModel,
    Stage1SSCModel,
    Stage1SSCMonoModel,
    Stage1SSCMonoLidarModel,
)
from ..models.stage1_ssc_bevdetocc_lidar_dense_depth import dense_metric_depth_loss
from ..models.stage1_ssc_bevdetocc_lidar import bevdet_depth_loss
from ..models.stage1_ssc_bevdetocc_lidar_pointmap import _pointmap_targets_from_depth
from ..pointmap_metrics import (
    PointmapMetricAccumulator,
    add_pointmap_metric_args,
    update_pointmap_metrics_from_batch,
)


BEVDETOCC_LIDAR_EXPS = (
    "bevdetocc_lidar",
    "bevdetocc_lidar_dense_depth",
    "bevdetocc_lidar_pointmap",
    "bevdetocc_lidar_pointmap_dense_depth",
)
POINTMAP_ONLY_EXPS = (
    "pointmap_postfusion_only",
    "pointmap_original",
)
DEPTH_ONLY_EXPS = (
    "depth_postfusion_only",
    "depth_promptfusion_only",
    "depth_original",
)
DET_EXPS = (
    "det_original",
    "det_postfusion_only",
)
POINTMAP_QUALITY_EXPS = (
    *POINTMAP_ONLY_EXPS,
    "bevdetocc_lidar_pointmap",
    "bevdetocc_lidar_pointmap_dense_depth",
)
BEVDETOCC_DENSE_DEPTH_DATA_EXPS = (
    "bevdetocc_lidar_dense_depth",
    "bevdetocc_lidar_pointmap",
    "bevdetocc_lidar_pointmap_dense_depth",
    "pointmap_postfusion_only",
    "depth_postfusion_only",
    "depth_promptfusion_only",
)
POINTMAP_DENSE_DEPTH_DATA_EXPS = (
    "pointmap_original",
    "depth_original",
)
LIDAR_EXPS = (
    "monoscene_lidar",
    *BEVDETOCC_LIDAR_EXPS,
    "pointmap_postfusion_only",
    "depth_postfusion_only",
    "depth_promptfusion_only",
    "det_postfusion_only",
)
MONOSCENE_LOSS_EXPS = ("monoscene", "monoscene_lidar")


def get_args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("OccAny Stage-1 SSC fine-tune", add_help=True)
    p.add_argument("--processed_root", default=None, type=str,
                   help="Path to data/kitti_processed.")
    p.add_argument(
        "--kitti_det_root",
        default="/home/dataset-local/lr/code/OccAny/raw_data/OpenDataLab___KITTI_Object",
        type=str,
        help="Path to the KITTI Object root for detection experiments.",
    )
    p.add_argument("--kittiodo_root", default=None, type=str,
                   help="Deprecated; calib.txt is read from processed_root/<split>_<seq>.")
    p.add_argument("--occany_ckpt", required=True, type=str,
                   help="OccAny checkpoint (encoder+decoder sub-dicts).")
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument(
        "--backbone",
        choices=["must3r", "da3"],
        default="must3r",
        help="Reconstruction backbone for supported Stage-1 variants.",
    )

    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=160)
    p.add_argument("--num_frames", type=int, default=5)
    p.add_argument("--frame_stride", type=int, default=4)

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
    p.add_argument(
        "--freeze_backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Freeze the OccAny reconstruction encoder/decoder. When "
            "--freeze_backbone_epochs > 0, freeze only for the first N epochs "
            "and then train the full model."
        ),
    )
    p.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=0,
        help=(
            "Number of initial epochs to freeze the OccAny backbone. The "
            "pointmap head remains trainable during this phase. If 0 and "
            "--freeze_backbone is set, the backbone stays frozen for all epochs."
        ),
    )

    # Which experiment variant to run.
    #   - "light":            existing LightOcc3DUNet head + CE+Lovasz loss.
    #   - "monoscene":        vendored MonoScene UNet3D head (context_prior=True)
    #                         via adapter, + CE+sem_scal+geo_scal+relation_ce
    #                         loss (requires <frame>_1_8.npy under processed_root).
    #   - "monoscene_lidar":  same as monoscene + a LiDAR fusion
    #                         block applied to OccAny's reconstruction tokens
    #                         (post-decoder, pre-lifting). The OccAny backbone
    #                         stays fully frozen; only fusion/lifting/head train.
    #                         Reads LiDAR from processed_root.
    #   - "bevdetocc_lidar": keep the first 2D LiDAR/image cross-attention,
    #                         then use LSS + LiDAR memory + NATTEN + BEVDet-OCC
    #                         3D encoder/head. Reads LiDAR from processed_root.
    #   - "bevdetocc_lidar_dense_depth": same as bevdetocc_lidar plus a
    #                         post-2D-fusion DPT-style dense depth head.
    #   - "bevdetocc_lidar_pointmap": same as bevdetocc_lidar plus a cloned
    #                         OccAny pointmap head after the 2D LiDAR fusion.
    #   - "pointmap_postfusion_only": 2D LiDAR fusion + pointmap head only.
    #   - "pointmap_original": original OccAny pointmap output only.
    #   - "bevdetocc_lidar_pointmap_dense_depth": current pointmap model plus
    #                         a post-2D-fusion dense depth auxiliary head.
    #   - "depth_original": original OccAny reconstruction tokens + DPT dense
    #                         depth head only; no fusion, pointmap loss, or SSC.
    #   - "depth_postfusion_only": 2D LiDAR fusion + DPT dense depth head only.
    #   - "depth_promptfusion_only": original reconstruction tokens + LiDAR
    #                         sparse-depth prompt injected into the DPT decoder.
    #   - "det_original": reliable-history KITTI Object detection from target
    #                         frame OccAny tokens + BEVDet/CenterHead backend.
    #   - "det_postfusion_only": same detection backend after 2D LiDAR fusion.
    p.add_argument(
        "--exp",
        choices=[
            "light",
            "monoscene",
            "monoscene_lidar",
            "bevdetocc_lidar",
            "bevdetocc_lidar_dense_depth",
            "bevdetocc_lidar_pointmap",
            "pointmap_postfusion_only",
            "pointmap_original",
            "bevdetocc_lidar_pointmap_dense_depth",
            "depth_original",
            "depth_postfusion_only",
            "depth_promptfusion_only",
            "det_original",
            "det_postfusion_only",
        ],
        default="light",
    )
    # LiDAR-fusion-only options.
    p.add_argument("--velodyne_root", default=None, type=str,
                   help="Deprecated compatibility option; KITTI LiDAR is read from "
                        "<processed_root>/<split>_<seq>/lidar/*.bin.")
    p.add_argument("--nuscenes_processed_root", default=None, type=str,
                   help="Path to data/nuscenes_processed for KITTI+nuScenes multi-dataset training.")
    p.add_argument("--nuscenes_raw_root", default=None, type=str,
                   help="Deprecated compatibility option; nuScenes calibration/pose is read from "
                        "<nuscenes_processed_root>/<split>_scene-*/meta/*.npz.")
    p.add_argument("--ddad_processed_root", default=None, type=str,
                   help="Path to data/ddad_processed for KITTI+DDAD depth-only multi-dataset training.")
    p.add_argument("--ddad_raw_root", default=None, type=str,
                   help="Path to raw DDAD ddad_train_val root; required for DDAD LiDAR poses.")
    p.add_argument("--multi_dataset", action="store_true",
                   help="Enable KITTI+nuScenes occupancy or KITTI+DDAD depth-only multi-dataset training.")
    p.add_argument("--dataset_ratio", type=str, default="1:1",
                   help="KITTI:nuScenes sampling ratio for --multi_dataset. Default: 1:1.")
    p.add_argument("--nuscenes_frame_stride", type=int, default=1,
                   help="Temporal stride for nuScenes samples under --multi_dataset.")
    p.add_argument("--iters_per_epoch", type=int, default=0,
                   help="If >0, cap/define training iterations per epoch.")
    p.add_argument("--val_iters", type=int, default=0,
                   help="If >0, cap validation iterations under --multi_dataset.")
    p.add_argument("--init_from", default=None, type=str,
                   help="Warm-start model weights from a checkpoint without restoring optimizer/epoch.")
    p.add_argument("--head_lr", type=float, default=1e-4,
                   help="LR for BEV/SSC head modules when using grouped optimizer.")
    p.add_argument("--classifier_lr", type=float, default=2e-4,
                   help="LR for final classifier when using grouped optimizer.")
    p.add_argument("--base_lr", type=float, default=5e-5,
                   help="LR for already-initialized non-head modules under --multi_dataset.")
    p.add_argument("--class_weights_path", default=None, type=str,
                   help="Optional JSON/list/tensor path for unified 27-class loss weights.")
    p.add_argument("--max_points_per_sweep", type=int, default=0,
                   help="If >0, deterministically stride-subsample each LiDAR sweep to this point count.")
    p.add_argument("--det_score_threshold", type=float, default=0.05,
                   help="Score threshold used by the local CenterHead decoder for detection eval.")
    p.add_argument("--det_pc_range", nargs=6, type=float, default=KITTI_OBJECT_DET_PC_RANGE,
                   metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
                   help="KITTI Object DET-only LiDAR pc_range. Other experiments keep their dataset grid.")
    p.add_argument("--det_depth_bound", nargs=3, type=float, default=KITTI_OBJECT_DET_DEPTH_BOUND,
                   metavar=("START", "END", "STEP"),
                   help="KITTI Object DET-only LSS depth bins used for lifting/depth supervision.")
    p.add_argument("--depth_supervision", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable BEVDet-style sparse LiDAR depth supervision for BEVDet-OCC LiDAR and DET variants.")
    p.add_argument("--depth_loss_weight", type=float, default=0.05,
                   help="Weight for BEVDet-style LSS depth loss when --depth_supervision is enabled.")
    p.add_argument("--dense_depth_supervision", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable dense metric depth auxiliary supervision for --exp=bevdetocc_lidar_dense_depth.")
    p.add_argument("--dense_depth_loss_weight", type=float, default=0.1,
                   help="Weight for dense metric depth auxiliary loss.")
    p.add_argument("--dense_depth_features", type=int, default=128,
                   help="Feature width for the single-scale DPT-style dense depth head.")
    p.add_argument("--prompt_depth_scale", choices=["log", "linear", "per_frame_max"], default="log",
                   help="Scale used for depth_promptfusion_only LiDAR prompt depth values.")
    p.add_argument("--prompt_depth_min", type=float, default=1e-3,
                   help="Minimum metric depth used to normalize/filter LiDAR prompt depth.")
    p.add_argument("--prompt_depth_max", type=float, default=120.0,
                   help="Maximum metric depth used to normalize/filter LiDAR prompt depth.")
    p.add_argument("--pointmap_supervision", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable post-2D-fusion pointmap supervision for --exp=bevdetocc_lidar_pointmap.")
    p.add_argument("--pointmap_loss_weight", type=float, default=0.1,
                   help="Weight for OccAny-style pointmap auxiliary loss.")
    p.add_argument("--pointmap_conf_alpha", type=float, default=0.2,
                   help="Confidence regularization alpha for OccAny-style pointmap loss.")
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
    add_pointmap_metric_args(p)
    return p


class BalancedMultiDataset(Dataset):
    """Deterministic wrapper over KITTI and nuScenes datasets."""

    def __init__(
        self,
        kitti: Dataset,
        nuscenes: Dataset,
        *,
        kitti_weight: int = 1,
        nuscenes_weight: int = 1,
        length: int | None = None,
        train: bool = True,
    ) -> None:
        self.kitti = kitti
        self.nuscenes = nuscenes
        self.pattern = ["kitti"] * int(kitti_weight) + ["nuscenes"] * int(nuscenes_weight)
        if not self.pattern:
            raise ValueError("dataset ratio must include at least one dataset.")
        self.all_frames = length is None or length <= 0
        base = len(kitti) + len(nuscenes) if self.all_frames else int(length)
        self._length = base
        self.train = bool(train)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int):
        if self.all_frames:
            if index < len(self.kitti):
                return self.kitti[index]
            return self.nuscenes[index - len(self.kitti)]

        dataset_name = self.pattern[index % len(self.pattern)]
        cycle = index // len(self.pattern)
        if dataset_name == "kitti":
            return self.kitti[cycle % len(self.kitti)]
        return self.nuscenes[cycle % len(self.nuscenes)]


class FixedLengthDataset(Dataset):
    """Repeat one dataset to a fixed number of iterations per epoch."""

    def __init__(self, dataset: Dataset, length: int | None = None) -> None:
        self.dataset = dataset
        self._length = int(length) if length is not None and length > 0 else len(dataset)
        if len(dataset) <= 0:
            raise ValueError("FixedLengthDataset requires a non-empty dataset.")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int):
        return self.dataset[index % len(self.dataset)]


def _parse_dataset_ratio(value: str) -> tuple[int, int]:
    parts = str(value).split(":")
    if len(parts) != 2:
        raise ValueError("--dataset_ratio must have form KITTI:NUSCENES, e.g. 1:1")
    a, b = int(parts[0]), int(parts[1])
    if a < 0 or b < 0 or a + b <= 0:
        raise ValueError("--dataset_ratio values must be non-negative and not both zero.")
    return a, b


def _multi_dataset_with_ddad(args) -> bool:
    return bool(getattr(args, "multi_dataset", False) and getattr(args, "ddad_processed_root", None))


def _multi_dataset_with_nuscenes(args) -> bool:
    return bool(
        getattr(args, "multi_dataset", False)
        and getattr(args, "nuscenes_processed_root", None)
        and not _multi_dataset_with_ddad(args)
    )


def _build_kitti_dataset(args, split: str) -> Kitti5FrameStage1Dataset:
    if args.exp in DET_EXPS:
        return KittiObject5FrameDetDataset(
            root=args.kitti_det_root,
            split=split,
            num_frames=args.num_frames,
            frame_stride=args.frame_stride,
            output_resolution=(args.width, args.height),
            max_points_per_sweep=args.max_points_per_sweep,
            grid_config=make_kitti_object_det_grid_config(tuple(args.det_pc_range)),
        )
    common = dict(
        processed_root=args.processed_root,
        split=split,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        output_resolution=(args.width, args.height),
        cam_idx=0,
    )
    if args.exp in BEVDETOCC_DENSE_DEPTH_DATA_EXPS:
        return Kitti5FrameStage1LidarDenseDepthDataset(
            velodyne_root=args.velodyne_root,
            max_points_per_sweep=args.max_points_per_sweep,
            **common,
        )
    if args.exp in POINTMAP_DENSE_DEPTH_DATA_EXPS:
        return Kitti5FrameStage1DenseDepthDataset(**common)
    if args.exp == "bevdetocc_lidar":
        return Kitti5FrameStage1LidarDataset(
            velodyne_root=args.velodyne_root,
            max_points_per_sweep=args.max_points_per_sweep,
            **common,
        )
    if args.exp == "monoscene_lidar":
        return Kitti5FrameStage1MonoLidarDataset(
            velodyne_root=args.velodyne_root,
            max_points_per_sweep=args.max_points_per_sweep,
            **common,
        )
    if args.exp == "monoscene":
        return Kitti5FrameStage1MonoDataset(**common)
    return Kitti5FrameStage1Dataset(**common)


def _build_nuscenes_dataset(args, split: str) -> NuScenes5FrameStage1LidarDataset:
    if not args.nuscenes_processed_root:
        raise ValueError("--nuscenes_processed_root is required for nuScenes training/eval.")
    return NuScenes5FrameStage1LidarDataset(
        processed_root=args.nuscenes_processed_root,
        split=split,
        num_frames=args.num_frames,
        frame_stride=args.nuscenes_frame_stride,
        output_resolution=(args.width, args.height),
        max_points_per_sweep=args.max_points_per_sweep,
    )


def _build_ddad_dataset(args, split: str) -> Dataset:
    if not args.ddad_processed_root:
        raise ValueError("--ddad_processed_root is required for DDAD training/eval.")
    common = dict(
        processed_root=args.ddad_processed_root,
        raw_root=args.ddad_raw_root,
        split=split,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        output_resolution=(args.width, args.height),
        cam_idx=0,
        max_points_per_sweep=args.max_points_per_sweep,
    )
    if args.exp == "depth_original":
        return DDAD5FrameStage1DenseDepthDataset(**common)
    if args.exp == "depth_postfusion_only":
        return DDAD5FrameStage1LidarDenseDepthDataset(**common)
    raise ValueError(
        "--multi_dataset with --ddad_processed_root supports only "
        "--exp=depth_original or --exp=depth_postfusion_only."
    )


def _limit_dataset(dataset: Dataset, length: int) -> Dataset:
    length = int(length)
    if length <= 0 or len(dataset) <= length:
        return dataset
    return Subset(dataset, range(length))


def _build_dataset(args, split: str) -> Dataset:
    if not bool(getattr(args, "multi_dataset", False)):
        return _build_kitti_dataset(args, split)

    if args.exp in DET_EXPS:
        raise ValueError("Detection experiments do not support --multi_dataset.")

    if _multi_dataset_with_ddad(args):
        if args.exp not in ("depth_original", "depth_postfusion_only"):
            raise ValueError(
                "--multi_dataset with --ddad_processed_root supports only "
                "--exp=depth_original or --exp=depth_postfusion_only."
            )
        if int(args.batch_size) != 1:
            raise ValueError("--multi_dataset with DDAD requires --batch_size=1.")
        if args.nuscenes_processed_root:
            raise ValueError("Use either --ddad_processed_root or --nuscenes_processed_root, not both.")
        kitti = _build_kitti_dataset(args, split)
        ddad = _build_ddad_dataset(args, split)
        dataset = ConcatDataset([kitti, ddad])
        length = (
            int(args.iters_per_epoch)
            if split == "train" and int(args.iters_per_epoch) > 0
            else None
        )
        return FixedLengthDataset(dataset, length=length) if length is not None else dataset

    if args.exp not in (
        "bevdetocc_lidar",
        "bevdetocc_lidar_pointmap",
        "pointmap_postfusion_only",
    ):
        raise ValueError(
            "--multi_dataset currently supports --exp=bevdetocc_lidar or "
            "--exp=bevdetocc_lidar_pointmap or --exp=pointmap_postfusion_only only."
        )
    if int(args.batch_size) != 1:
        raise ValueError("--multi_dataset requires --batch_size=1 because grid shapes differ.")

    kitti_weight, nuscenes_weight = _parse_dataset_ratio(args.dataset_ratio)
    if kitti_weight == 0:
        nuscenes = _build_nuscenes_dataset(args, split)
        length = (
            int(args.iters_per_epoch)
            if split == "train" and int(args.iters_per_epoch) > 0
            else None
        )
        return FixedLengthDataset(nuscenes, length=length)
    if nuscenes_weight == 0:
        kitti = _build_kitti_dataset(args, split)
        length = (
            int(args.iters_per_epoch)
            if split == "train" and int(args.iters_per_epoch) > 0
            else None
        )
        return FixedLengthDataset(kitti, length=length)

    kitti = _build_kitti_dataset(args, split)
    nuscenes = _build_nuscenes_dataset(args, split)
    if split != "train":
        val_length = int(args.val_iters) if int(args.val_iters) > 0 else None
        return BalancedMultiDataset(
            kitti,
            nuscenes,
            kitti_weight=kitti_weight,
            nuscenes_weight=nuscenes_weight,
            length=val_length,
            train=False,
        )

    train_length = int(args.iters_per_epoch) if int(args.iters_per_epoch) > 0 else None
    return BalancedMultiDataset(
        kitti,
        nuscenes,
        kitti_weight=kitti_weight,
        nuscenes_weight=nuscenes_weight,
        length=train_length,
        train=True,
    )


def _eval_valid_classes(dataset_name: str, args) -> List[int] | None:
    if not bool(getattr(args, "multi_dataset", False)):
        return None
    if dataset_name == "kitti":
        return list(range(1, len(KITTI_SSC_CLASS_NAMES)))
    if dataset_name == "nuscenes":
        return sorted({int(v) for v in NUSCENES_TO_UNIFIED.tolist() if int(v) != 0})
    return None


def _build_eval_loaders(args) -> Dict[str, DataLoader]:
    if not bool(getattr(args, "multi_dataset", False)):
        dataset = _build_dataset(args, "val")
        return {"kitti": _build_loader(args, dataset, train=False)}

    if _multi_dataset_with_ddad(args):
        val_limit = int(args.val_iters) if int(args.val_iters) > 0 else 0
        datasets = {
            "kitti": _build_kitti_dataset(args, "val"),
            "ddad": _build_ddad_dataset(args, "val"),
        }
        return {
            name: _build_loader(args, _limit_dataset(dataset, val_limit), train=False)
            for name, dataset in datasets.items()
        }

    val_limit = int(args.val_iters) if int(args.val_iters) > 0 else 0
    kitti_weight, nuscenes_weight = _parse_dataset_ratio(args.dataset_ratio)
    datasets = {}
    if kitti_weight > 0:
        datasets["kitti"] = _build_kitti_dataset(args, "val")
    if nuscenes_weight > 0:
        datasets["nuscenes"] = _build_nuscenes_dataset(args, "val")
    return {
        name: _build_loader(args, _limit_dataset(dataset, val_limit), train=False)
        for name, dataset in datasets.items()
    }


def _collate_fn(args):
    if args.exp in DET_EXPS:
        return collate_kitti_object_det
    if bool(getattr(args, "multi_dataset", False)):
        if _multi_dataset_with_ddad(args):
            if args.exp == "depth_original":
                return collate_stage1_dense_depth
            if args.exp == "depth_postfusion_only":
                return collate_stage1_lidar_dense_depth
            raise ValueError(
                "KITTI+DDAD multi-dataset collate supports only depth_original "
                "or depth_postfusion_only."
            )

        def _collate_multi(batch):
            names = [b.get("dataset_name", "kitti") for b in batch]
            if any(n != names[0] for n in names):
                raise RuntimeError(
                    "Mixed dataset batch contains multiple grid shapes; use --batch_size=1."
                )
            if names[0] == "nuscenes":
                return collate_stage1_nuscenes_lidar(batch)
            if args.exp in BEVDETOCC_DENSE_DEPTH_DATA_EXPS:
                return collate_stage1_lidar_dense_depth(batch)
            return collate_stage1_lidar(batch)
        return _collate_multi
    if args.exp in BEVDETOCC_DENSE_DEPTH_DATA_EXPS:
        return collate_stage1_lidar_dense_depth
    if args.exp in POINTMAP_DENSE_DEPTH_DATA_EXPS:
        return collate_stage1_dense_depth
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
            drop_last=False,
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
        drop_last=False,
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


def _filter_state_dict_without_backbone(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {
        k: v
        for k, v in state_dict.items()
        if not k.removeprefix("module.").startswith("backbone.")
        or _is_backbone_pointmap_head_key(k.removeprefix("module."))
    }


def _state_dict_without_backbone(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Checkpoint only trainable/non-frozen modules when OccAny is frozen."""
    return _filter_state_dict_without_backbone(model.state_dict())


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def _expand_20_to_27_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Expand BEVDet classifier rows from KITTI-20 to unified-27 when present."""
    out = dict(state_dict)
    semantic_sources = {
        20: 5,   # other <- other-vehicle
        21: 14,  # barrier <- fence
        22: 5,   # bus <- other-vehicle
        23: 5,   # construction-vehicle <- other-vehicle
        24: 19,  # traffic-cone <- traffic-sign
        25: 5,   # trailer <- other-vehicle
        26: 13,  # manmade <- building
    }
    for weight_key in ("occ_head.predicter.2.weight", "module.occ_head.predicter.2.weight"):
        bias_key = weight_key.replace("weight", "bias")
        if weight_key not in out:
            continue
        w = out[weight_key]
        if not (isinstance(w, torch.Tensor) and w.ndim == 2 and w.shape[0] == 20):
            continue
        new_w = w.new_empty((27, w.shape[1]))
        new_w[:20].copy_(w)
        for dst, src in semantic_sources.items():
            new_w[dst].copy_(w[src])
        new_w[20:].add_(torch.randn_like(new_w[20:]) * 1e-3)
        out[weight_key] = new_w
        if bias_key in out and isinstance(out[bias_key], torch.Tensor) and out[bias_key].shape[0] == 20:
            b = out[bias_key]
            new_b = b.new_empty((27,))
            new_b[:20].copy_(b)
            for dst, src in semantic_sources.items():
                new_b[dst].copy_(b[src])
            new_b[20:].add_(torch.randn_like(new_b[20:]) * 1e-3)
            out[bias_key] = new_b
    return out


def _load_class_weights(args) -> torch.Tensor:
    if args.class_weights_path:
        path = Path(args.class_weights_path)
        if path.suffix.lower() == ".json":
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                values = [float(data[name]) for name in UNIFIED_SSC_CLASS_NAMES]
            else:
                values = [float(v) for v in data]
            weights = torch.tensor(values, dtype=torch.float32)
        else:
            weights = torch.as_tensor(torch.load(path, map_location="cpu"), dtype=torch.float32)
        if weights.numel() != len(UNIFIED_SSC_CLASS_NAMES):
            raise ValueError(
                f"class weights must have {len(UNIFIED_SSC_CLASS_NAMES)} values, got {weights.numel()}"
            )
        return weights.view(-1).clamp(0.25, 5.0)

    return torch.ones(len(UNIFIED_SSC_CLASS_NAMES), dtype=torch.float32)


def _load_init_from(model: nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        state = ckpt["model"]
    else:
        state = {
            **{f"lifting.{k}": v for k, v in ckpt.get("lifting", {}).items()},
            **{f"occ_head.{k}": v for k, v in ckpt.get("occ_head", {}).items()},
        }
        if "fusion" in ckpt:
            state.update({f"fusion.{k}": v for k, v in ckpt["fusion"].items()})
    model_state = model.state_dict()
    state = _strip_module_prefix(_filter_state_dict_without_backbone(state))
    wants_unified_head = any(
        k.endswith("occ_head.predicter.2.weight")
        and isinstance(v, torch.Tensor)
        and v.ndim >= 1
        and v.shape[0] == len(UNIFIED_SSC_CLASS_NAMES)
        for k, v in model_state.items()
    )
    if wants_unified_head:
        state = _expand_20_to_27_state_dict(state)
    compatible = {
        k: v
        for k, v in state.items()
        if k in model_state and isinstance(v, torch.Tensor) and tuple(v.shape) == tuple(model_state[k].shape)
    }
    missing_shape = [
        k for k, v in state.items()
        if k in model_state and isinstance(v, torch.Tensor) and tuple(v.shape) != tuple(model_state[k].shape)
    ]
    status = model.load_state_dict(compatible, strict=False)
    print(
        f"[init_from] loaded {len(compatible)} tensors from {ckpt_path}; "
        f"skipped_shape={len(missing_shape)} missing={len(status.missing_keys)} "
        f"unexpected={len(status.unexpected_keys)}"
    )


def _trainable_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _is_backbone_pointmap_head_key(name: str) -> bool:
    return (
        name == "backbone.decoder.pts3d_task_token"
        or name.startswith("backbone.decoder.head_dec.")
        or name.startswith("backbone.decoder._head_wrapper.")
    )


def _is_occany_pointmap_head_param(name: str) -> bool:
    return (
        name == "decoder.pts3d_task_token"
        or name.startswith("decoder.head_dec.")
        or name.startswith("decoder._head_wrapper.")
    )


def _set_occany_backbone_frozen(
    model: nn.Module,
    freeze: bool,
    *,
    keep_pointmap_head_trainable: bool = True,
) -> bool:
    """Set OccAny backbone freeze state without touching external heads."""
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return False
    freeze = bool(freeze)
    if hasattr(backbone, "set_frozen") and not hasattr(backbone, "decoder"):
        backbone.set_frozen(freeze)
        if hasattr(model, "freeze_backbone"):
            model.freeze_backbone = freeze
        return freeze

    freeze_forward = freeze and not keep_pointmap_head_trainable
    if hasattr(backbone, "freeze"):
        backbone.freeze = freeze_forward
    if hasattr(backbone, "_capturer"):
        backbone._capturer.detach_output = freeze_forward

    for name, param in backbone.named_parameters():
        is_head = keep_pointmap_head_trainable and _is_occany_pointmap_head_param(name)
        param.requires_grad = (not freeze) or is_head

    if hasattr(model, "freeze_backbone"):
        model.freeze_backbone = freeze
    if freeze:
        if hasattr(backbone, "encoder"):
            backbone.encoder.eval()
        if hasattr(backbone, "decoder"):
            backbone.decoder.eval()
    else:
        backbone.train(model.training)
    return freeze


def _freeze_occany_backbone(model: nn.Module) -> bool:
    return _set_occany_backbone_frozen(model, True)


def _apply_backbone_freeze_state(model: nn.Module, args, epoch: int) -> bool:
    if (
        args.exp in DEPTH_ONLY_EXPS
        and getattr(args, "backbone", "must3r") == "must3r"
        and bool(args.freeze_backbone)
        and int(args.freeze_backbone_epochs) == 0
    ):
        args.freeze_backbone = False
        print("[backbone] depth-only experiments fine-tune OccAny; forcing --no-freeze_backbone.")
    if not bool(args.freeze_backbone):
        return _set_occany_backbone_frozen(model, False)
    freeze_epochs = int(args.freeze_backbone_epochs)
    freeze = True if freeze_epochs == 0 else int(epoch) < freeze_epochs
    return _set_occany_backbone_frozen(
        model,
        freeze,
        keep_pointmap_head_trainable=(args.exp == "pointmap_original"),
    )


def _make_optimizer(model: nn.Module, args) -> torch.optim.Optimizer | None:
    named_trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    trainable_params = [p for _, p in named_trainable]
    if not trainable_params:
        return None
    if bool(getattr(args, "multi_dataset", False)) and not _multi_dataset_with_ddad(args):
        base_params = []
        head_params = []
        classifier_params = []
        for name, param in named_trainable:
            clean = name.removeprefix("module.")
            if clean.startswith("occ_head.predicter.2."):
                classifier_params.append(param)
            elif clean.startswith("occ_head.") or clean.startswith("pointmap_head."):
                head_params.append(param)
            else:
                base_params.append(param)
        groups = []
        if base_params:
            groups.append({"params": base_params, "lr": float(args.base_lr), "lr_scale": 1.0})
        if head_params:
            groups.append({"params": head_params, "lr": float(args.head_lr), "lr_scale": 1.0})
        if classifier_params:
            groups.append(
                {"params": classifier_params, "lr": float(args.classifier_lr), "lr_scale": 1.0}
            )
        return torch.optim.AdamW(
            groups,
            lr=float(args.base_lr),
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
    return torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )


def _optimizer_param_count(optimizer: torch.optim.Optimizer | None) -> int:
    if optimizer is None:
        return 0
    return sum(p.numel() for group in optimizer.param_groups for p in group["params"])


def _ensure_optimizer_matches_trainable(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    args,
) -> torch.optim.Optimizer | None:
    trainable_count = _trainable_param_count(model)
    if trainable_count == 0:
        if optimizer is not None:
            print("[optimizer] no trainable parameters; dropping optimizer.")
        return None
    if optimizer is None or _optimizer_param_count(optimizer) != trainable_count:
        optimizer = _make_optimizer(model, args)
        print(f"[optimizer] rebuilt with trainable params={trainable_count:,}.")
    return optimizer


def _ddp_kwargs(args) -> Dict:
    kwargs = dict(device_ids=[args.gpu])
    if args.exp in (*LIDAR_EXPS, *DEPTH_ONLY_EXPS, *DET_EXPS) or int(args.freeze_backbone_epochs) > 0:
        kwargs.update(find_unused_parameters=True, static_graph=False)
    else:
        kwargs.update(find_unused_parameters=False, static_graph=True)
    return kwargs


def _wrap_distributed_if_needed(
    model: nn.Module,
    args,
) -> tuple[nn.Module, nn.Module]:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.distributed and trainable_params:
        wrapped = nn.parallel.DistributedDataParallel(model, **_ddp_kwargs(args))
        return wrapped, wrapped.module
    if args.distributed and not trainable_params:
        print("[ddp] skipped DDP wrapping because no parameters require grad.")
    return model, model


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


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _model_forward(model: nn.Module, batch: Dict, device: torch.device, args):
    """Dispatch the model forward to match each experiment's signature."""
    views = _move_views_to_device(batch["views"], device)
    T_target_from_refcam = batch["T_target_from_refcam"].to(device, non_blocking=True)
    grid_config = {
        key: batch[key].to(device, non_blocking=True)
        for key in (
            "grid_size",
            "voxel_origin",
            "voxel_size",
            "half_grid_size",
            "half_voxel_origin",
            "half_voxel_size",
            "fusion_vox_origin",
            "fusion_vox_size",
            "fusion_vox_grid",
        )
        if key in batch and isinstance(batch[key], torch.Tensor)
    }
    if args.exp == "pointmap_postfusion_only":
        return model(
            views,
            T_target_from_refcam,
            _move_points_to_device(batch["points_per_frame"], device),
            batch["T_cam_from_velo"].to(device, non_blocking=True),
            batch["K_per_frame"].to(device, non_blocking=True),
            batch["image_hw"].to(device, non_blocking=True),
            grid_config=grid_config,
        )
    if args.exp == "depth_postfusion_only":
        return model(
            views,
            T_target_from_refcam,
            _move_points_to_device(batch["points_per_frame"], device),
            batch["T_cam_from_velo"].to(device, non_blocking=True),
            batch["K_per_frame"].to(device, non_blocking=True),
            batch["image_hw"].to(device, non_blocking=True),
            grid_config=grid_config,
        )
    if args.exp == "depth_promptfusion_only":
        return model(
            views,
            T_target_from_refcam,
            _move_points_to_device(batch["points_per_frame"], device),
            batch["T_cam_from_velo"].to(device, non_blocking=True),
            batch["K_per_frame"].to(device, non_blocking=True),
            batch["image_hw"].to(device, non_blocking=True),
            grid_config=grid_config,
        )
    if args.exp in DET_EXPS:
        return model(
            views,
            T_target_from_refcam,
            _move_points_to_device(batch["points_per_frame"], device),
            batch["T_cam_from_velo"].to(device, non_blocking=True),
            batch["K_per_frame"].to(device, non_blocking=True),
            batch["image_hw"].to(device, non_blocking=True),
            return_depth=bool(getattr(args, "depth_supervision", False)),
            grid_config=grid_config,
        )
    if args.exp == "pointmap_original":
        return model(views, T_target_from_refcam)
    if args.exp == "depth_original":
        return model(views, T_target_from_refcam)
    if args.exp in BEVDETOCC_LIDAR_EXPS:
        return model(
            views,
            T_target_from_refcam,
            _move_points_to_device(batch["points_per_frame"], device),
            batch["T_cam_from_velo"].to(device, non_blocking=True),
            batch["K_per_frame"].to(device, non_blocking=True),
            batch["image_hw"].to(device, non_blocking=True),
            return_depth=bool(getattr(args, "depth_supervision", False)),
            grid_config=grid_config,
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


def _dense_depth_only_loss(
    out: Dict,
    batch: Dict,
    args,
) -> tuple[torch.Tensor, Dict[str, float]]:
    if "dense_depth" not in out:
        raise RuntimeError("Depth-only training expected model output key 'dense_depth'.")
    dense_gt = batch["dense_depth"].to(device=out["dense_depth"].device, non_blocking=True)
    frame_mask = batch["dense_depth_frame_mask"].to(
        device=out["dense_depth"].device,
        non_blocking=True,
    )
    dense_weighted, dense_raw, dense_valid, dense_frames = dense_metric_depth_loss(
        out["dense_depth"],
        dense_gt,
        frame_mask=frame_mask,
        loss_weight=float(args.dense_depth_loss_weight),
    )
    return dense_weighted, {
        "dense_depth": float(dense_raw.detach()),
        "dense_depth_weighted": float(dense_weighted.detach()),
        "dense_depth_valid": float(dense_valid.detach()),
        "dense_depth_frames": float(dense_frames.detach()),
    }


def _maybe_add_bevdet_depth_loss(
    loss: torch.Tensor,
    details: Dict[str, float],
    out: Dict,
    args,
) -> tuple[torch.Tensor, Dict[str, float]]:
    if args.exp not in (*BEVDETOCC_LIDAR_EXPS, *DET_EXPS) or not bool(
        getattr(args, "depth_supervision", False)
    ):
        return loss, details
    if "depth_logits" not in out or "gt_depth" not in out:
        raise RuntimeError(
            "BEVDet-style depth supervision expected model output to contain "
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
    batch: Dict,
    args,
) -> tuple[torch.Tensor, Dict[str, float]]:
    if args.exp not in (
        "bevdetocc_lidar_dense_depth",
        "bevdetocc_lidar_pointmap_dense_depth",
    ) or not bool(
        getattr(args, "dense_depth_supervision", False)
    ):
        return loss, details
    if "dense_depth" not in out:
        raise RuntimeError("Dense depth supervision expected model output key 'dense_depth'.")
    dense_gt = batch["dense_depth"].to(device=out["dense_depth"].device, non_blocking=True)
    frame_mask = batch["dense_depth_frame_mask"].to(
        device=out["dense_depth"].device,
        non_blocking=True,
    )
    dense_weighted, dense_raw, dense_valid, dense_frames = dense_metric_depth_loss(
        out["dense_depth"],
        dense_gt,
        frame_mask=frame_mask,
        loss_weight=float(args.dense_depth_loss_weight),
    )
    details = dict(details)
    details["dense_depth"] = float(dense_raw.detach())
    details["dense_depth_weighted"] = float(dense_weighted.detach())
    details["dense_depth_valid"] = float(dense_valid.detach())
    details["dense_depth_frames"] = float(dense_frames.detach())
    return loss + dense_weighted, details


def _stack_cam2world_from_views(batch: Dict, device: torch.device) -> torch.Tensor:
    return torch.stack(
        [
            v["cam2world"].to(device=device, dtype=torch.float32, non_blocking=True)
            for v in batch["views"]
        ],
        dim=1,
    )


_POINTMAP_OCCANY_CRITERIA: Dict[tuple, nn.Module] = {}


def _pointmap_K_per_frame(batch: Dict, device: torch.device) -> torch.Tensor:
    if "K_per_frame" in batch:
        return batch["K_per_frame"].to(device=device, dtype=torch.float32, non_blocking=True)
    return torch.stack(
        [
            v["camera_intrinsics"].to(device=device, dtype=torch.float32, non_blocking=True)
            for v in batch["views"]
        ],
        dim=1,
    )


def _occany_pointmap_criterion(args, device: torch.device, *, is_train: bool) -> nn.Module:
    key = (
        str(device),
        bool(is_train),
        float(args.pointmap_conf_alpha),
    )
    criterion = _POINTMAP_OCCANY_CRITERIA.get(key)
    if criterion is None:
        pixel_loss = Regr3D_multiview(
            L21,
            norm_mode="?avg_dis",
            loss_in_log=False,
            gt_scale=not is_train,
        )
        if is_train:
            criterion = ConfLoss_multiview(
                pixel_loss,
                alpha=float(args.pointmap_conf_alpha),
            )
        else:
            criterion = pixel_loss
        criterion = criterion.to(device)
        _POINTMAP_OCCANY_CRITERIA[key] = criterion
    return criterion


def _occany_pointmap_gt_pred(
    out: Dict,
    batch: Dict,
    device: torch.device,
) -> tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor], torch.Tensor]:
    required = ("pointmap_pts3d", "pointmap_pts3d_local", "pointmap_conf")
    missing = [k for k in required if k not in out]
    if missing:
        raise RuntimeError(f"Pointmap supervision expected model output keys {missing}.")
    if "dense_depth" not in batch:
        raise RuntimeError(
            "Pointmap supervision requires batch['dense_depth']; use the dense-depth "
            "dataset/collate path."
        )

    pred_pts3d = out["pointmap_pts3d"].float()
    pred_pts3d_local = out["pointmap_pts3d_local"].float()
    if pred_pts3d.shape != pred_pts3d_local.shape:
        raise RuntimeError(
            f"pred global/local pointmap shapes differ: {tuple(pred_pts3d.shape)} vs "
            f"{tuple(pred_pts3d_local.shape)}."
        )
    if pred_pts3d.ndim != 5 or pred_pts3d.shape[-1] != 3:
        raise RuntimeError(f"pred pointmaps must be (B,N,H,W,3), got {tuple(pred_pts3d.shape)}.")

    dense_depth = batch["dense_depth"].to(device=device, dtype=torch.float32, non_blocking=True)
    K_per_frame = _pointmap_K_per_frame(batch, device)
    cam2world_per_frame = _stack_cam2world_from_views(batch, device)
    gt_ref, _gt_local, valid = _pointmap_targets_from_depth(
        dense_depth,
        K_per_frame,
        cam2world_per_frame,
    )
    if pred_pts3d.shape != gt_ref.shape:
        raise RuntimeError(
            f"pred pointmap shape {tuple(pred_pts3d.shape)} does not match GT "
            f"{tuple(gt_ref.shape)}."
        )

    frame_mask = batch.get("dense_depth_frame_mask")
    if frame_mask is not None:
        fm = frame_mask.to(device=device, dtype=torch.bool, non_blocking=True).view(
            dense_depth.shape[0], dense_depth.shape[1], 1, 1
        )
        valid = valid & fm

    conf = out["pointmap_conf"].to(device=device, dtype=torch.float32)
    if conf.ndim == pred_pts3d.ndim and conf.shape[-1] == 1:
        conf = conf[..., 0]
    if conf.shape != valid.shape:
        raise RuntimeError(
            f"pred_conf shape {tuple(conf.shape)} does not match valid mask "
            f"{tuple(valid.shape)}."
        )

    T_ref_from_world = torch.linalg.inv(cam2world_per_frame[:, 0])
    T_ref_from_cam = T_ref_from_world[:, None] @ cam2world_per_frame
    is_metric_scale = torch.ones(
        dense_depth.shape[0],
        device=device,
        dtype=torch.bool,
    )
    gt_views = [
        {
            "pts3d": gt_ref[:, view_idx],
            "valid_mask": valid[:, view_idx],
            "camera_pose": T_ref_from_cam[:, view_idx],
            "is_metric_scale": is_metric_scale,
        }
        for view_idx in range(dense_depth.shape[1])
    ]
    pred = {
        "pts3d": pred_pts3d,
        "pts3d_local": pred_pts3d_local,
        "conf": conf,
    }
    return gt_views, pred, valid


def _pointmap_occany_loss(
    out: Dict,
    batch: Dict,
    args,
    *,
    is_train: bool,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    device = out["pointmap_pts3d"].device
    with torch.amp.autocast(device_type=device.type, enabled=False):
        gt_views, pred, valid = _occany_pointmap_gt_pred(out, batch, device)
        valid_count = valid.sum()
        if not bool(valid_count.item()):
            zero = pred["pts3d"].sum() * 0.0
            return zero, zero, {
                "pointmap": 0.0,
                "pointmap_weighted": 0.0,
                "pointmap_pts3d": 0.0,
                "pointmap_pts3d_local": 0.0,
                "pointmap_conf_loss_g": 0.0,
                "pointmap_conf_loss_l": 0.0,
                "pointmap_valid": 0.0,
            }

        criterion = _occany_pointmap_criterion(args, device, is_train=is_train)
        raw_loss, occany_details = criterion(gt_views, pred)
        weighted = float(args.pointmap_loss_weight) * raw_loss

    pointmap_details = {
        "pointmap": float(raw_loss.detach()),
        "pointmap_weighted": float(weighted.detach()),
        "pointmap_valid": float(valid_count.detach()),
    }
    for key, value in occany_details.items():
        if key == "Regr3D_multiview_pts3d":
            pointmap_details["pointmap_pts3d"] = float(value)
        elif key == "Regr3D_multiview_pts3d_local":
            pointmap_details["pointmap_pts3d_local"] = float(value)
        elif key in ("conf_loss_g", "conf_loss_l"):
            pointmap_details[f"pointmap_{key}"] = float(value)
        else:
            pointmap_details[f"pointmap_{key}"] = float(value)
    return weighted, raw_loss, pointmap_details


def _maybe_add_pointmap_loss(
    loss: torch.Tensor,
    details: Dict[str, float],
    out: Dict,
    batch: Dict,
    args,
    *,
    is_train: bool,
) -> tuple[torch.Tensor, Dict[str, float]]:
    if args.exp not in (
        "bevdetocc_lidar_pointmap",
        "bevdetocc_lidar_pointmap_dense_depth",
    ) or not bool(
        getattr(args, "pointmap_supervision", False)
    ):
        return loss, details
    pointmap_weighted, _pointmap_raw, pointmap_details = _pointmap_occany_loss(
        out,
        batch,
        args,
        is_train=is_train,
    )
    prefix = "pointmap_train" if is_train else "pointmap_eval"
    details = dict(details)
    for key, value in pointmap_details.items():
        suffix = "loss" if key == "pointmap" else key.removeprefix("pointmap_")
        details[f"{prefix}_{suffix}"] = value
    return loss + pointmap_weighted, details


def _pointmap_only_loss(
    out: Dict,
    batch: Dict,
    args,
    *,
    is_train: bool,
) -> tuple[torch.Tensor, Dict[str, float]]:
    weighted, raw, details = _pointmap_occany_loss(
        out,
        batch,
        args,
        is_train=is_train,
    )
    prefix = "pointmap_train" if is_train else "pointmap_eval"
    out_details = {
        f"{prefix}_loss": float(raw.detach()),
        f"{prefix}_weighted": float(weighted.detach()),
    }
    for key, value in details.items():
        suffix = "loss" if key == "pointmap" else key.removeprefix("pointmap_")
        out_details[f"{prefix}_{suffix}"] = value
    return weighted, out_details


def _sanitize_metric_key(name: str) -> str:
    return str(name).replace(" ", "_").replace("-", "_").replace("/", "_")


def _float_list(values) -> List[float]:
    return [float(v) for v in values]


def _jsonable_metric_value(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        return _jsonable_metric_value(value.tolist())
    if isinstance(value, torch.Tensor):
        return _jsonable_metric_value(value.detach().cpu().tolist())
    if isinstance(value, (list, tuple)):
        return [_jsonable_metric_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable_metric_value(v) for k, v in value.items()}
    return None



def _jsonable_config_value(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable_config_value(value.tolist())
    if isinstance(value, torch.Tensor):
        return _jsonable_config_value(value.detach().cpu().tolist())
    if isinstance(value, (list, tuple)):
        return [_jsonable_config_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable_config_value(v) for k, v in value.items()}
    return str(value)


def _optimizer_config(optimizer: torch.optim.Optimizer | None) -> List[Dict] | None:
    if optimizer is None:
        return None
    groups = []
    for group in optimizer.param_groups:
        groups.append(
            {
                str(k): _jsonable_config_value(v)
                for k, v in group.items()
                if k != "params"
            }
        )
    return groups


def _save_training_config(
    args,
    *,
    model_cls_name: str,
    model: nn.Module,
    backbone_hash: str | None,
    initial_backbone_frozen: bool,
    start_backbone_frozen: bool,
    start_epoch: int,
    syncbn_on: bool,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    train_dataset: Dataset,
    eval_loaders: Dict[str, DataLoader],
) -> None:
    config = {
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "argv": list(sys.argv),
        "command": " ".join(shlex.quote(x) for x in sys.argv),
        "args": {k: _jsonable_config_value(v) for k, v in vars(args).items()},
        "model": {
            "class": model_cls_name,
            "total_params": int(sum(p.numel() for p in model.parameters())),
            "trainable_params": int(_trainable_param_count(model)),
            "backbone_hash": backbone_hash,
            "initial_backbone_frozen": bool(initial_backbone_frozen),
            "start_backbone_frozen": bool(start_backbone_frozen),
            "sync_batchnorm": bool(syncbn_on),
        },
        "optimizer": {
            "class": optimizer.__class__.__name__ if optimizer is not None else None,
            "param_groups": _optimizer_config(optimizer),
        },
        "criterion": criterion.__class__.__name__,
        "runtime": {
            "start_epoch": int(start_epoch),
            "distributed": bool(getattr(args, "distributed", False)),
            "rank": int(getattr(args, "rank", 0)),
            "world_size": int(getattr(args, "world_size", 1)),
            "gpu": int(getattr(args, "gpu", -1)),
        },
        "data": {
            "train_samples": int(len(train_dataset)),
            "eval_samples": {
                str(name): int(len(loader.dataset))
                for name, loader in eval_loaders.items()
            },
        },
    }
    out_path = Path(args.output_dir) / "training_config.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[config] wrote {out_path}")


def _gather_pointmap_metric_states(local_state: Dict, args) -> List[Dict] | None:
    if not getattr(args, "distributed", False):
        return [local_state]
    import torch.distributed as dist

    world_size = misc.get_world_size()
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_state)
    return gathered


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
        converted = _jsonable_metric_value(v)
        if converted is not None:
            log_stats[f"val_{k}"] = converted

    if "class_names" in val_stats and "iou_per_class" in val_stats:
        per_class_iou = _per_class_iou_dict(val_stats)
        log_stats["val_class_names"] = [str(name) for name in val_stats["class_names"]]
        log_stats["val_iou_per_class"] = _float_list(val_stats["iou_per_class"])
        log_stats["val_iou_per_class_by_name"] = per_class_iou
        for class_name, iou in per_class_iou.items():
            log_stats[f"val_iou_class_{_sanitize_metric_key(class_name)}"] = iou

    return log_stats


def _flatten_eval_stats(eval_stats: Dict[str, Dict]) -> Dict:
    flat: Dict = {}
    for dataset_name, stats in eval_stats.items():
        prefix = f"{dataset_name}_"
        for key, value in stats.items():
            flat[f"{prefix}{key}"] = value
        for class_name, iou in _per_class_iou_dict(stats).items():
            flat[f"{prefix}iou_class_{_sanitize_metric_key(class_name)}"] = float(iou)
    return flat


def _adjust_lr(optimizer, epoch_f: float, args) -> float:
    use_multidataset_base_lr = bool(getattr(args, "multi_dataset", False)) and not _multi_dataset_with_ddad(args)
    base_lr = float(getattr(args, "base_lr", args.lr)) if use_multidataset_base_lr else float(args.lr)
    if epoch_f < args.warmup_epochs:
        lr = base_lr * epoch_f / max(args.warmup_epochs, 1)
    else:
        progress = (epoch_f - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        lr = args.min_lr + 0.5 * (base_lr - args.min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        initial = float(pg.get("initial_lr", pg.get("lr", lr)))
        if "initial_lr" not in pg:
            pg["initial_lr"] = initial
        scale = initial / max(base_lr, 1e-12)
        pg["lr"] = lr * scale * pg.get("lr_scale", 1.0)
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
    if optimizer is not None:
        optimizer.zero_grad()

    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    for step, batch in enumerate(metric_logger.log_every(loader, args.print_freq, header)):
        epoch_f = epoch + step / max(len(loader), 1)
        if optimizer is not None and step % accum == 0:
            _adjust_lr(optimizer, epoch_f, args)

        target = None
        if args.exp not in (*POINTMAP_ONLY_EXPS, *DEPTH_ONLY_EXPS, *DET_EXPS):
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
            if args.exp in POINTMAP_ONLY_EXPS:
                loss, details = _pointmap_only_loss(out, batch, args, is_train=True)
            elif args.exp in DEPTH_ONLY_EXPS:
                loss, details = _dense_depth_only_loss(out, batch, args)
            elif args.exp in DET_EXPS:
                gt_boxes = [b.to(device=device, non_blocking=True) for b in batch["gt_bboxes_3d"]]
                gt_labels = [l.to(device=device, non_blocking=True) for l in batch["gt_labels_3d"]]
                loss, details = _unwrap_model(model).det_loss(
                    out["det_preds"],
                    gt_boxes,
                    gt_labels,
                )
                loss, details = _maybe_add_bevdet_depth_loss(loss, details, out, args)
            elif args.exp in MONOSCENE_LOSS_EXPS:
                cp = batch["CP_mega_matrix"].to(device, non_blocking=True)
                loss, details = criterion(out, target, cp)
            elif args.exp in BEVDETOCC_LIDAR_EXPS:
                loss, details = criterion(out["ssc_logit"], target)
                loss, details = _maybe_add_bevdet_depth_loss(loss, details, out, args)
                loss, details = _maybe_add_dense_depth_loss(loss, details, out, batch, args)
                loss, details = _maybe_add_pointmap_loss(
                    loss, details, out, batch, args, is_train=True
                )
            else:
                loss, details = criterion(out, target)
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss is {loss_value}; details={details}; stopping.")
        loss = loss / accum

        trainable_now = [p for p in model.parameters() if p.requires_grad]
        if optimizer is not None and loss.requires_grad and trainable_now:
            loss_scaler(
                loss,
                optimizer,
                parameters=trainable_now,
                update_grad=(step + 1) % accum == 0,
            )
            if (step + 1) % accum == 0:
                optimizer.zero_grad()

        lr = optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0
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
    dataset_name: str = "val",
):
    model.eval()
    eval_ssc = args.exp not in (*POINTMAP_ONLY_EXPS, *DEPTH_ONLY_EXPS, *DET_EXPS)
    ssc = None
    if eval_ssc:
        metric_names = (
            list(UNIFIED_SSC_CLASS_NAMES)
            if bool(getattr(args, "multi_dataset", False))
            else list(KITTI_SSC_CLASS_NAMES)
        )
        ssc = SSCMetrics(
            n_classes=len(metric_names),
            class_names=metric_names,
            other_class=None,
            ignore_other_class_in_mIoU=False,
            empty_class=0,
            valid_classes=_eval_valid_classes(dataset_name, args),
        )
    pointmap_accum = PointmapMetricAccumulator(args) if args.exp in POINTMAP_QUALITY_EXPS else None
    det_gt_boxes: List[torch.Tensor] = []
    det_gt_labels: List[torch.Tensor] = []
    det_pred_boxes: List[torch.Tensor] = []
    det_pred_scores: List[torch.Tensor] = []
    det_pred_labels: List[torch.Tensor] = []
    det_sample_ids: List[int] = []
    losses_total = 0.0
    details_total: Dict[str, float] = {}
    n_batches = 0
    amp_dtype = (
        torch.bfloat16 if args.amp == "bf16" else (torch.float16 if args.amp == "fp16" else None)
    )
    for batch in loader:
        target = None
        if eval_ssc:
            target = batch["voxel_label"].to(device, non_blocking=True)

        ctx = (
            torch.autocast("cuda", dtype=amp_dtype)
            if amp_dtype is not None
            else torch.autocast("cuda", enabled=False)
        )
        with ctx:
            out = _model_forward(model, batch, device, args)
            if args.exp in POINTMAP_ONLY_EXPS:
                loss, _details = _pointmap_only_loss(out, batch, args, is_train=False)
                logits = None
            elif args.exp in DEPTH_ONLY_EXPS:
                loss, _details = _dense_depth_only_loss(out, batch, args)
                logits = None
            elif args.exp in DET_EXPS:
                gt_boxes = [b.to(device=device, non_blocking=True) for b in batch["gt_bboxes_3d"]]
                gt_labels = [l.to(device=device, non_blocking=True) for l in batch["gt_labels_3d"]]
                loss, _details = _unwrap_model(model).det_loss(
                    out["det_preds"],
                    gt_boxes,
                    gt_labels,
                )
                loss, _details = _maybe_add_bevdet_depth_loss(loss, _details, out, args)
                logits = None
            elif args.exp in MONOSCENE_LOSS_EXPS:
                cp = batch["CP_mega_matrix"].to(device, non_blocking=True)
                loss, _details = criterion(out, target, cp)
                logits = out["ssc_logit"]
            elif args.exp in BEVDETOCC_LIDAR_EXPS:
                loss, _details = criterion(out["ssc_logit"], target)
                loss, _details = _maybe_add_bevdet_depth_loss(loss, _details, out, args)
                loss, _details = _maybe_add_dense_depth_loss(loss, _details, out, batch, args)
                loss, _details = _maybe_add_pointmap_loss(
                    loss, _details, out, batch, args, is_train=False
                )
                logits = out["ssc_logit"]
            else:
                loss, _details = criterion(out, target)
                logits = out
        if pointmap_accum is not None:
            update_pointmap_metrics_from_batch(
                pointmap_accum,
                out,
                batch,
                device,
                _stack_cam2world_from_views(batch, device),
            )
        if args.exp in DET_EXPS:
            decoded = _unwrap_model(model).det_decode(out["det_preds"])
            for i, pred in enumerate(decoded):
                det_sample_ids.append(int(batch["sample_id"][i]))
                det_gt_boxes.append(batch["gt_bboxes_3d"][i].cpu())
                det_gt_labels.append(batch["gt_labels_3d"][i].cpu())
                det_pred_boxes.append(pred["boxes_3d"].detach().cpu())
                det_pred_scores.append(pred["scores_3d"].detach().cpu())
                det_pred_labels.append(pred["labels_3d"].detach().cpu())
        losses_total += float(loss.detach())
        for k, v in _details.items():
            details_total[k] = details_total.get(k, 0.0) + float(v)
        n_batches += 1

        if eval_ssc:
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
        if eval_ssc:
            ssc.tps = _reduce_np(ssc.tps)
            ssc.fps = _reduce_np(ssc.fps)
            ssc.fns = _reduce_np(ssc.fns)
            ssc.completion_tp = _reduce_scalar(ssc.completion_tp)
            ssc.completion_fp = _reduce_scalar(ssc.completion_fp)
            ssc.completion_fn = _reduce_scalar(ssc.completion_fn)

    stats = ssc.get_stats() if eval_ssc else {}
    if args.exp in DET_EXPS:
        if getattr(args, "distributed", False):
            import torch.distributed as dist

            local_state = {
                "gt_boxes": det_gt_boxes,
                "gt_labels": det_gt_labels,
                "pred_boxes": det_pred_boxes,
                "pred_scores": det_pred_scores,
                "pred_labels": det_pred_labels,
                "sample_ids": det_sample_ids,
            }
            gathered = [None for _ in range(misc.get_world_size())]
            dist.all_gather_object(gathered, local_state)
            det_gt_boxes = []
            det_gt_labels = []
            det_pred_boxes = []
            det_pred_scores = []
            det_pred_labels = []
            det_sample_ids = []
            for item in gathered:
                det_gt_boxes.extend(item["gt_boxes"])
                det_gt_labels.extend(item["gt_labels"])
                det_pred_boxes.extend(item["pred_boxes"])
                det_pred_scores.extend(item["pred_scores"])
                det_pred_labels.extend(item["pred_labels"])
                det_sample_ids.extend(item["sample_ids"])
        if det_sample_ids:
            keep_indices = []
            seen_ids = set()
            for i, sid in enumerate(det_sample_ids):
                if int(sid) in seen_ids:
                    continue
                seen_ids.add(int(sid))
                keep_indices.append(i)
            det_gt_boxes = [det_gt_boxes[i] for i in keep_indices]
            det_gt_labels = [det_gt_labels[i] for i in keep_indices]
            det_pred_boxes = [det_pred_boxes[i] for i in keep_indices]
            det_pred_scores = [det_pred_scores[i] for i in keep_indices]
            det_pred_labels = [det_pred_labels[i] for i in keep_indices]
        stats.update(
            evaluate_lidar_det_ap40(
                det_gt_boxes,
                det_gt_labels,
                det_pred_boxes,
                det_pred_scores,
                det_pred_labels,
            )
        )
    if pointmap_accum is not None:
        local_state = pointmap_accum.state_dict()
        gathered = _gather_pointmap_metric_states(local_state, args)
        if getattr(args, "distributed", False):
            import torch.distributed as dist

            dist.barrier()
        assert gathered is not None
        pm_metrics = PointmapMetricAccumulator.from_states(args, gathered).finalize()
        stats.update({f"pointmap_quality_{k}": v for k, v in pm_metrics.items()})
    loss_avg = losses_total / max(n_batches, 1)
    detail_avgs = {k: v / max(n_batches, 1) for k, v in details_total.items()}
    if log_writer is not None:
        it = 1000 * epoch
        tb_prefix = "val" if dataset_name == "val" else f"val/{dataset_name}"
        log_writer.add_scalar(f"{tb_prefix}/loss", loss_avg, it)
        for k, v in detail_avgs.items():
            log_writer.add_scalar(f"{tb_prefix}/{k}", v, it)
        if eval_ssc:
            log_writer.add_scalar(f"{tb_prefix}/iou", stats["iou"], it)
            log_writer.add_scalar(f"{tb_prefix}/mIoU", stats["mIoU"], it)
            log_writer.add_scalar(f"{tb_prefix}/precision", stats["precision"], it)
            log_writer.add_scalar(f"{tb_prefix}/recall", stats["recall"], it)
            for class_name, iou in _per_class_iou_dict(stats).items():
                log_writer.add_scalar(f"{tb_prefix}/iou_class/{class_name}", iou, it)
        for k, v in stats.items():
            if k.startswith("pointmap_quality_") and isinstance(v, (int, float)):
                log_writer.add_scalar(f"{tb_prefix}/{k}", v, it)
            if k.startswith("det_") and isinstance(v, (int, float)):
                log_writer.add_scalar(f"{tb_prefix}/{k}", v, it)
    if eval_ssc:
        print(
            f"Val {dataset_name} [{epoch}] loss={loss_avg:.4f} IoU={stats['iou']*100:.2f} "
            f"mIoU={stats['mIoU']*100:.2f} P={stats['precision']*100:.2f} "
            f"R={stats['recall']*100:.2f}"
        )
    else:
        if args.exp in DET_EXPS:
            print(
                f"Val {dataset_name} [{epoch}] loss={loss_avg:.4f} "
                f"mAP_bev={stats.get('det_map_bev', 0.0)*100:.2f} "
                f"mAP_3d={stats.get('det_map_3d', 0.0)*100:.2f} "
                f"gt={stats.get('det_gt_count', 0.0):.0f} "
                f"pred={stats.get('det_pred_count', 0.0):.0f}"
            )
        elif args.exp in DEPTH_ONLY_EXPS:
            dense = detail_avgs.get("dense_depth", float("nan"))
            valid = detail_avgs.get("dense_depth_valid", float("nan"))
            frames = detail_avgs.get("dense_depth_frames", float("nan"))
            print(
                f"Val {dataset_name} [{epoch}] loss={loss_avg:.4f} "
                f"dense_depth={dense:.4f} valid={valid:.0f} frames={frames:.0f}"
            )
        else:
            pm_l2 = stats.get("pointmap_quality_pts3d_l2", float("nan"))
            pm_depth = stats.get("pointmap_quality_depth_absrel", float("nan"))
            pm_chamfer = stats.get("pointmap_quality_chamfer_distance", float("nan"))
            print(
                f"Val {dataset_name} [{epoch}] loss={loss_avg:.4f} "
                f"pointmap_l2={pm_l2:.4f} depth_absrel={pm_depth:.4f} "
                f"chamfer={pm_chamfer:.4f}"
            )
    return dict(loss=loss_avg, **detail_avgs, **stats)


def main():
    args = get_args_parser().parse_args()
    args.freeze_backbone_epochs = int(args.freeze_backbone_epochs)
    if args.freeze_backbone_epochs < 0:
        raise ValueError("--freeze_backbone_epochs must be >= 0.")
    if args.exp in DET_EXPS:
        if bool(getattr(args, "multi_dataset", False)):
            raise ValueError("Detection experiments do not support --multi_dataset.")
        if not args.kitti_det_root:
            raise ValueError("--kitti_det_root is required for detection experiments.")
    elif not args.processed_root:
        raise ValueError("--processed_root is required for non-detection experiments.")
    if args.ddad_processed_root and not args.multi_dataset:
        raise ValueError("--ddad_processed_root requires --multi_dataset.")
    if _multi_dataset_with_ddad(args):
        if args.nuscenes_processed_root:
            raise ValueError("Use either --ddad_processed_root or --nuscenes_processed_root, not both.")
        if args.exp not in ("depth_original", "depth_postfusion_only"):
            raise ValueError(
                "--multi_dataset with --ddad_processed_root supports only "
                "--exp=depth_original or --exp=depth_postfusion_only."
            )
        if int(args.batch_size) != 1:
            raise ValueError("--multi_dataset with DDAD requires --batch_size=1.")
        if args.exp == "depth_postfusion_only" and not args.ddad_raw_root:
            raise ValueError("--ddad_raw_root is required for DDAD depth_postfusion_only.")
    if args.backbone == "da3" and args.exp not in ("depth_original", "depth_postfusion_only"):
        raise ValueError(
            "--backbone da3 is currently wired for --exp depth_original "
            "or --exp depth_postfusion_only only."
        )
    if args.backbone == "da3" and (
        int(args.height) % int(args.patch_size) != 0
        or int(args.width) % int(args.patch_size) != 0
    ):
        raise ValueError("--backbone da3 requires height/width divisible by --patch_size.")

    misc.init_distributed_mode(args)
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

    if args.exp == "pointmap_original":
        model_cls = Stage1PointmapOriginalModel
    elif args.exp == "pointmap_postfusion_only":
        model_cls = Stage1PointmapPostFusionOnlyModel
    elif args.exp == "depth_original":
        model_cls = Stage1DepthOriginalModel
    elif args.exp == "depth_postfusion_only":
        model_cls = Stage1DepthPostFusionOnlyModel
    elif args.exp == "depth_promptfusion_only":
        model_cls = Stage1DepthPromptFusionOnlyModel
    elif args.exp == "det_original":
        model_cls = Stage1DetOriginalModel
    elif args.exp == "det_postfusion_only":
        model_cls = Stage1DetPostFusionOnlyModel
    elif args.exp == "bevdetocc_lidar_pointmap_dense_depth":
        model_cls = Stage1SSCBEVDetOccLidarPointmapDenseDepthModel
    elif args.exp == "bevdetocc_lidar_pointmap":
        model_cls = Stage1SSCBEVDetOccLidarPointmapModel
    elif args.exp == "bevdetocc_lidar_dense_depth":
        model_cls = Stage1SSCBEVDetOccLidarDenseDepthModel
    elif args.exp == "bevdetocc_lidar":
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
        num_classes=(len(UNIFIED_SSC_CLASS_NAMES) if bool(getattr(args, "multi_dataset", False)) else 20),
        patch_size=args.patch_size,
        token_dim=args.token_dim,
        backbone_img_size=(args.height, args.width),
        backbone_dtype=backbone_dtype,
    )
    if args.exp in (*BEVDETOCC_LIDAR_EXPS, "pointmap_postfusion_only", "depth_postfusion_only"):
        model_kwargs["fusion_attn_type"] = "cross"
        model_kwargs["num_frames"] = args.num_frames
        model_kwargs["freeze_backbone"] = args.freeze_backbone
        if args.exp == "depth_postfusion_only":
            model_kwargs["backbone"] = args.backbone
    if args.exp in DET_EXPS:
        model_kwargs["num_frames"] = args.num_frames
        model_kwargs["freeze_backbone"] = args.freeze_backbone
        model_kwargs["backbone"] = args.backbone
        model_kwargs["det_score_threshold"] = args.det_score_threshold
        model_kwargs["det_pc_range"] = tuple(args.det_pc_range)
        model_kwargs["depth_bound"] = tuple(args.det_depth_bound)
        if args.exp == "det_postfusion_only":
            model_kwargs["fusion_attn_type"] = "cross"
    if args.exp in ("pointmap_original", "depth_original", "depth_promptfusion_only"):
        model_kwargs["num_frames"] = args.num_frames
        model_kwargs["freeze_backbone"] = args.freeze_backbone
        if args.exp == "depth_original":
            model_kwargs["backbone"] = args.backbone
    if args.exp in (
        "bevdetocc_lidar_dense_depth",
        "bevdetocc_lidar_pointmap_dense_depth",
        *DEPTH_ONLY_EXPS,
    ):
        model_kwargs["dense_depth_features"] = args.dense_depth_features
    if args.exp == "depth_promptfusion_only":
        model_kwargs["prompt_depth_scale"] = args.prompt_depth_scale
        model_kwargs["prompt_depth_min"] = args.prompt_depth_min
        model_kwargs["prompt_depth_max"] = args.prompt_depth_max
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
    if args.exp in ("depth_original", "depth_postfusion_only"):
        print(f"[backbone] selected={args.backbone}")
    if args.exp in POINTMAP_ONLY_EXPS:
        print(
            f"[pointmap-only] weight={args.pointmap_loss_weight} "
            f"conf_alpha={args.pointmap_conf_alpha}"
        )
        if args.exp == "pointmap_postfusion_only":
            print("[fusion] attn_type=cross (pointmap-only post-fusion branch)")
    if args.exp in DEPTH_ONLY_EXPS:
        print(
            f"[depth-only] dense metric depth loss weight={args.dense_depth_loss_weight} "
            f"features={args.dense_depth_features}"
        )
        if _multi_dataset_with_ddad(args):
            print(
                "[multi_dataset] KITTI+DDAD depth concat; "
                f"ddad_processed_root={args.ddad_processed_root}"
            )
        if args.exp == "depth_postfusion_only":
            print("[fusion] attn_type=cross (depth-only post-fusion branch)")
        if args.exp == "depth_promptfusion_only":
            print(
                "[fusion] prompt_depth=lidar_sparse+mask "
                f"scale={args.prompt_depth_scale} "
                f"min={args.prompt_depth_min:g} max={args.prompt_depth_max:g} "
                "(PromptDA-style decoder injection)"
            )
    if args.exp in BEVDETOCC_LIDAR_EXPS:
        print("[fusion] attn_type=cross (forced for BEVDet-OCC LiDAR branch)")
        print("[backend] LSS half-grid -> LiDAR memory -> NATTEN -> BEVDet CustomResNet3D/LSSFPN3D")
        print(
            f"[depth_supervision] enabled={args.depth_supervision} "
            f"weight={args.depth_loss_weight}"
        )
        if args.exp in ("bevdetocc_lidar_dense_depth", "bevdetocc_lidar_pointmap_dense_depth"):
            print(
                f"[dense_depth] enabled={args.dense_depth_supervision} "
                f"weight={args.dense_depth_loss_weight} "
                f"features={args.dense_depth_features}"
            )
        if args.exp in ("bevdetocc_lidar_pointmap", "bevdetocc_lidar_pointmap_dense_depth"):
            print(
                f"[pointmap] enabled={args.pointmap_supervision} "
                f"weight={args.pointmap_loss_weight} "
                f"conf_alpha={args.pointmap_conf_alpha}"
            )
    if args.exp in DET_EXPS:
        print(
            "[det] KITTI Object reliable-history detection "
            f"classes={KITTI_OBJECT_CLASS_NAMES} root={args.kitti_det_root}"
        )
        print(
            f"[det] frame_stride={args.frame_stride} num_frames={args.num_frames} "
            f"score_thr={args.det_score_threshold}"
        )
        print(
            f"[det] pc_range={tuple(float(v) for v in args.det_pc_range)} "
            f"half_grid={model.half_grid_size} depth_bound={tuple(float(v) for v in args.det_depth_bound)}"
        )
        print(
            f"[depth_supervision] enabled={args.depth_supervision} "
            f"weight={args.depth_loss_weight}"
        )
        if args.exp == "det_postfusion_only":
            print("[fusion] attn_type=cross (detection post-fusion branch)")
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

    # Apply initial OccAny backbone freeze state before DDP/optimizer creation.
    # External heads stay trainable; for pointmap_original the original OccAny
    # decoder pointmap head is also kept trainable during frozen epochs.
    backbone_frozen = _apply_backbone_freeze_state(model, args, 0)
    backbone_hash = (
        _state_dict_hash(model.backbone.state_dict())
        if hasattr(model, "backbone") and getattr(args, "backbone", "must3r") == "must3r"
        else None
    )
    print(f"[backbone] initial freeze={backbone_frozen} (reconstruction backbone)")
    if args.freeze_backbone and args.freeze_backbone_epochs:
        print(
            f"[backbone] freeze_backbone_epochs={args.freeze_backbone_epochs} "
            "then full-model training."
        )
    elif args.freeze_backbone:
        print("[backbone] frozen for all epochs.")
    else:
        print("[backbone] trainable from epoch 0.")
    if backbone_hash is not None:
        print(f"Backbone state_dict hash: {backbone_hash}")

    if args.init_from:
        _load_init_from(model, args.init_from)

    # Convert BN -> SyncBN before DDP wrap. At per-GPU bs=1 the MonoScene
    # head's BatchNorm3d layers see one-sample stats per forward, which both
    # makes training trivially fit the single sample and accumulates noisy
    # running stats -- exactly the train-down/val-up divergence we observed
    # without this step. Light head uses GroupNorm so it's unaffected.
    syncbn_on = args.distributed and (
        args.syncbn == "on"
        or (
            args.syncbn == "auto"
            and args.exp in (
                "monoscene",
                "monoscene_lidar",
                *BEVDETOCC_LIDAR_EXPS,
                *POINTMAP_ONLY_EXPS,
                *DEPTH_ONLY_EXPS,
                *DET_EXPS,
            )
        )
    )
    if syncbn_on:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print(f"[syncbn] converted BatchNorm -> SyncBatchNorm (mode={args.syncbn}).")
    else:
        print(f"[syncbn] disabled (mode={args.syncbn}, distributed={args.distributed}).")

    model, model_to_save = _wrap_distributed_if_needed(model, args)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    train_count = sum(p.numel() for p in trainable_params)
    print(f"Trainable params: {train_count:,} / {total_params:,}")

    optimizer = _make_optimizer(model, args)
    if optimizer is None:
        print("[optimizer] no trainable parameters; running eval/log-only epochs.")
    loss_scaler = NativeScaler(enabled=(args.amp == "fp16"))
    if args.exp in DET_EXPS:
        criterion = nn.Identity().to(device)
    elif args.exp in MONOSCENE_LOSS_EXPS:
        criterion = MonoSceneSSCLoss().to(device)
    elif bool(getattr(args, "multi_dataset", False)):
        criterion = SSCLoss(class_weights=_load_class_weights(args)).to(device)
    else:
        criterion = SSCLoss().to(device)

    train_dataset = _build_dataset(args, "train")
    eval_loaders = _build_eval_loaders(args)
    eval_sample_summary = ", ".join(
        f"{name}={len(loader.dataset)}" for name, loader in eval_loaders.items()
    )
    print(f"Train samples: {len(train_dataset)}; Val samples: {eval_sample_summary}")
    train_loader = _build_loader(args, train_dataset, train=True)

    start_epoch = 0
    optimizer_state_to_load = None
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        prev_hash = ckpt.get("backbone_hash", None)
        ckpt_args = ckpt.get("args", {})
        if isinstance(ckpt_args, dict):
            ckpt_freeze_backbone = bool(ckpt_args.get("freeze_backbone", True))
        else:
            ckpt_freeze_backbone = bool(getattr(ckpt_args, "freeze_backbone", True))
        frozen_for_all_epochs = bool(args.freeze_backbone) and int(args.freeze_backbone_epochs) == 0
        ckpt_frozen_for_all_epochs = ckpt_freeze_backbone and int(
            ckpt_args.get("freeze_backbone_epochs", 0)
            if isinstance(ckpt_args, dict)
            else getattr(ckpt_args, "freeze_backbone_epochs", 0)
        ) == 0
        if (
            frozen_for_all_epochs
            and ckpt_frozen_for_all_epochs
            and prev_hash is not None
            and prev_hash != backbone_hash
        ):
            raise RuntimeError(
                f"Backbone state_dict hash mismatch on resume: "
                f"checkpoint expected {prev_hash}, current --occany_ckpt gives {backbone_hash}. "
                "Refusing to resume — the frozen backbone has changed."
            )
        if args.exp in (*BEVDETOCC_LIDAR_EXPS, *POINTMAP_ONLY_EXPS, *DEPTH_ONLY_EXPS, *DET_EXPS) and "model" in ckpt:
            if frozen_for_all_epochs and ckpt_frozen_for_all_epochs:
                model_state = _filter_state_dict_without_backbone(ckpt["model"])
            else:
                model_state = ckpt["model"]
            status = model_to_save.load_state_dict(model_state, strict=False)
            print(
                f"[resume:{args.exp}] loaded model state: "
                f"missing={len(status.missing_keys)} unexpected={len(status.unexpected_keys)}"
            )
        elif args.exp not in (*BEVDETOCC_LIDAR_EXPS, *POINTMAP_ONLY_EXPS, *DEPTH_ONLY_EXPS, *DET_EXPS) and "lifting" in ckpt:
            model_to_save.lifting.load_state_dict(ckpt["lifting"], strict=False)
        if args.exp not in (*BEVDETOCC_LIDAR_EXPS, *POINTMAP_ONLY_EXPS, *DEPTH_ONLY_EXPS, *DET_EXPS) and "occ_head" in ckpt:
            model_to_save.occ_head.load_state_dict(ckpt["occ_head"], strict=False)
        if (
            args.exp not in (*BEVDETOCC_LIDAR_EXPS, *POINTMAP_ONLY_EXPS, *DEPTH_ONLY_EXPS, *DET_EXPS)
            and "fusion" in ckpt
            and hasattr(model_to_save, "fusion")
        ):
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
        optimizer_state_to_load = ckpt.get("optimizer", None)
        if "scaler" in ckpt:
            loss_scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"Resumed from {args.resume}; epoch={start_epoch}")

    resumed_backbone_frozen = _apply_backbone_freeze_state(model_to_save, args, start_epoch)
    optimizer = _ensure_optimizer_matches_trainable(model, optimizer, args)
    if optimizer is not None and optimizer_state_to_load is not None:
        try:
            optimizer.load_state_dict(optimizer_state_to_load)
        except Exception as e:
            print("Optimizer load failed:", e)
    print(
        f"[backbone] start_epoch={start_epoch} freeze={resumed_backbone_frozen} "
        f"trainable={_trainable_param_count(model_to_save):,}"
    )
    if misc.is_main_process():
        _save_training_config(
            args,
            model_cls_name=model_cls.__name__,
            model=model_to_save,
            backbone_hash=backbone_hash,
            initial_backbone_frozen=backbone_frozen,
            start_backbone_frozen=resumed_backbone_frozen,
            start_epoch=start_epoch,
            syncbn_on=syncbn_on,
            optimizer=optimizer,
            criterion=criterion,
            train_dataset=train_dataset,
            eval_loaders=eval_loaders,
        )

    log_writer = (
        SummaryWriter(log_dir=args.output_dir) if misc.is_main_process() else None
    )

    if args.eval_only:
        _apply_backbone_freeze_state(model_to_save, args, start_epoch)
        for eval_name, eval_loader in eval_loaders.items():
            eval_one_epoch(
                model,
                eval_loader,
                criterion,
                device,
                start_epoch,
                args,
                log_writer,
                dataset_name=eval_name,
            )
        return

    print(f"Start training: epochs={args.epochs}, start_epoch={start_epoch}")
    t0 = time.time()
    last_backbone_frozen = None
    for epoch in range(start_epoch, args.epochs):
        backbone_frozen = _apply_backbone_freeze_state(model_to_save, args, epoch)
        if backbone_frozen != last_backbone_frozen:
            if args.distributed:
                if isinstance(model, nn.parallel.DistributedDataParallel):
                    model = model.module
                model, model_to_save = _wrap_distributed_if_needed(model, args)
            print(
                f"[backbone] epoch={epoch} freeze={backbone_frozen} "
                f"trainable={_trainable_param_count(model_to_save):,}"
            )
            optimizer = _ensure_optimizer_matches_trainable(model, optimizer, args)
        last_backbone_frozen = backbone_frozen

        train_stats = train_one_epoch(
            model, train_loader, optimizer, loss_scaler, criterion, device, epoch, args, log_writer
        )

        val_stats: Dict[str, Dict] = {}
        if (epoch + 1) % args.eval_freq == 0:
            for eval_name, eval_loader in eval_loaders.items():
                val_stats[eval_name] = eval_one_epoch(
                    model,
                    eval_loader,
                    criterion,
                    device,
                    epoch,
                    args,
                    log_writer,
                    dataset_name=eval_name,
                )

        if misc.is_main_process():
            if args.exp in (*BEVDETOCC_LIDAR_EXPS, *POINTMAP_ONLY_EXPS, *DEPTH_ONLY_EXPS, *DET_EXPS):
                if args.freeze_backbone and int(args.freeze_backbone_epochs) == 0:
                    model_state = _state_dict_without_backbone(model_to_save)
                else:
                    model_state = model_to_save.state_dict()
                ckpt_payload = {
                    "model": model_state,
                    "scaler": loss_scaler.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                    "backbone_hash": backbone_hash,
                }
            else:
                ckpt_payload = {
                    "lifting": model_to_save.lifting.state_dict(),
                    "occ_head": model_to_save.occ_head.state_dict(),
                    "scaler": loss_scaler.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                    "backbone_hash": backbone_hash,
                }
            if optimizer is not None:
                ckpt_payload["optimizer"] = optimizer.state_dict()
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
                if args.exp in (*BEVDETOCC_LIDAR_EXPS, *POINTMAP_ONLY_EXPS, *DEPTH_ONLY_EXPS, *DET_EXPS):
                    keep_payload = {
                        "model": ckpt_payload["model"],
                        "epoch": epoch,
                        "args": vars(args),
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
                f.write(
                    json.dumps(
                        _build_log_stats(
                            epoch,
                            train_stats,
                            _flatten_eval_stats(val_stats),
                        )
                    )
                    + "\n"
                )

    dt = str(datetime.timedelta(seconds=int(time.time() - t0)))
    print(f"Done. Total time: {dt}")


if __name__ == "__main__":
    main()
