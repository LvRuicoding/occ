"""Visualize the three unified Stage-1 experiment checkpoints on val frames.

Examples:
  python -m ft.kitti_stage1_5f.tools.visualize_three_dataset_experiments \
    --dataset both --kitti_sample_idx 0 --nuscenes_sample_idx 0

  python -m ft.kitti_stage1_5f.tools.visualize_three_dataset_experiments \
    --dataset nuscenes --sample_idx 0 --device cuda:0
"""
from __future__ import annotations

from .. import _paths  # noqa: F401

import argparse
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from occany.metrics.ssc import SSCMetrics
from occany.model.must3r_blocks.attention import toggle_memory_efficient_attention
from occany.utils.checkpoint_io import register_legacy_checkpoint_modules
from occany.utils.image_util import convert_images_to_uint8_hwc

from ..datasets import (
    Kitti5FrameStage1LidarDataset,
    NuScenes5FrameStage1LidarDataset,
    UNIFIED_SSC_CLASS_NAMES,
    collate_stage1_lidar,
    collate_stage1_nuscenes_lidar,
)
from ..datasets.unified_occ import GRID_CONFIGS, NUSCENES_TO_UNIFIED
from ..models import (
    Stage1SSCBEVDetOccLidarDenseDepthModel,
    Stage1SSCBEVDetOccLidarModel,
    Stage1SSCBEVDetOccLidarPointmapDenseDepthModel,
    Stage1SSCBEVDetOccLidarPointmapModel,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "visuals" / "three_dataset_experiments_compare"
DEFAULT_CKPTS = (
    REPO_ROOT / "output" / "three_dataset_experiments" / "01_unified_kitti_only" / "checkpoint-last.pth",
    REPO_ROOT / "output" / "three_dataset_experiments" / "02_unified_nuscenes_only" / "checkpoint-last.pth",
    REPO_ROOT / "output" / "three_dataset_experiments" / "03_unified_kitti_nuscenes_all_frames" / "checkpoint-last.pth",
)
DEFAULT_LABELS = (
    "01_kitti_only",
    "02_nuscenes_only",
    "03_kitti_nuscenes",
)

MODEL_BY_EXP = {
    "bevdetocc_lidar": Stage1SSCBEVDetOccLidarModel,
    "bevdetocc_lidar_dense_depth": Stage1SSCBEVDetOccLidarDenseDepthModel,
    "bevdetocc_lidar_pointmap": Stage1SSCBEVDetOccLidarPointmapModel,
    "bevdetocc_lidar_pointmap_dense_depth": Stage1SSCBEVDetOccLidarPointmapDenseDepthModel,
}

UNIFIED_COLORS = np.array(
    [
        [0, 0, 0],        # empty
        [100, 150, 245],  # car
        [100, 230, 245],  # bicycle
        [30, 60, 150],    # motorcycle
        [80, 30, 180],    # truck
        [100, 80, 250],   # other-vehicle
        [255, 30, 30],    # person
        [255, 40, 200],   # bicyclist
        [150, 30, 90],    # motorcyclist
        [255, 0, 255],    # road
        [255, 150, 255],  # parking
        [75, 0, 75],      # sidewalk
        [175, 0, 75],     # other-ground
        [255, 200, 0],    # building
        [255, 120, 50],   # fence
        [0, 175, 0],      # vegetation
        [135, 60, 0],     # trunk
        [150, 240, 80],   # terrain
        [255, 240, 150],  # pole
        [255, 0, 0],      # traffic-sign
        [180, 180, 180],  # other
        [255, 120, 50],   # barrier
        [80, 30, 180],    # bus
        [120, 70, 180],   # construction-vehicle
        [255, 210, 80],   # traffic-cone
        [100, 80, 250],   # trailer
        [255, 200, 0],    # manmade
    ],
    dtype=np.uint8,
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Visualize three unified KITTI/nuScenes Stage-1 experiment checkpoints"
    )
    parser.add_argument(
        "--dataset",
        choices=["both", "kitti", "nuscenes"],
        default="both",
        help="Which validation set to visualize. Default runs one KITTI frame and one nuScenes frame.",
    )
    parser.add_argument("--sample_idx", type=int, default=None,
                        help="Fallback validation index. If omitted, sample one with --seed.")
    parser.add_argument("--kitti_sample_idx", type=int, default=None,
                        help="KITTI validation index. Overrides --sample_idx for KITTI.")
    parser.add_argument("--nuscenes_sample_idx", type=int, default=None,
                        help="nuScenes validation index. Overrides --sample_idx for nuScenes.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpts", nargs="+", default=[str(p) for p in DEFAULT_CKPTS])
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_ROOT), type=str)
    parser.add_argument("--processed_root", default=None, type=str)
    parser.add_argument("--nuscenes_processed_root", default=None, type=str)
    parser.add_argument("--occany_ckpt", default=None, type=str)
    parser.add_argument("--width", default=None, type=int)
    parser.add_argument("--height", default=None, type=int)
    parser.add_argument("--num_frames", default=None, type=int)
    parser.add_argument("--frame_stride", default=None, type=int)
    parser.add_argument("--nuscenes_frame_stride", default=None, type=int)
    parser.add_argument("--max_points_per_sweep", default=None, type=int)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--max_voxels_plot", default=60000, type=int)
    parser.add_argument("--elev", default=22.0, type=float)
    parser.add_argument("--azim", default=-55.0, type=float)
    parser.add_argument("--save_grids", action="store_true",
                        help="Also save GT and prediction voxel grids as a compressed npz.")
    return parser.parse_args()


def ckpt_arg(ckpt_args: Dict, name: str, default):
    return ckpt_args.get(name, default) if isinstance(ckpt_args, dict) else default


def arg_or_ckpt(args: argparse.Namespace, ckpt_args: Dict, name: str, default):
    value = getattr(args, name)
    return value if value is not None else ckpt_arg(ckpt_args, name, default)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def amp_dtype_for(device: torch.device, amp: str) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if amp == "bf16":
        return torch.bfloat16
    if amp == "fp16":
        return torch.float16
    return None


def load_checkpoint(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def move_views_to_device(
    views: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    moved: List[Dict[str, torch.Tensor]] = []
    for view in views:
        item: Dict[str, torch.Tensor] = {}
        for key, value in view.items():
            item[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        moved.append(item)
    return moved


def move_points_to_device(
    points_per_frame: List[List[torch.Tensor]],
    device: torch.device,
) -> List[List[torch.Tensor]]:
    return [
        [points.to(device, non_blocking=True) for points in per_sample]
        for per_sample in points_per_frame
    ]


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def build_model_from_checkpoint(
    ckpt: Dict,
    ckpt_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module:
    ckpt_args = ckpt.get("args", {})
    ckpt_args = dict(ckpt_args) if isinstance(ckpt_args, dict) else {}
    exp = ckpt_arg(ckpt_args, "exp", "bevdetocc_lidar")
    if exp not in MODEL_BY_EXP:
        raise ValueError(f"Unsupported checkpoint exp={exp!r} in {ckpt_path}")

    amp = str(ckpt_arg(ckpt_args, "amp", "bf16"))
    if device.type != "cuda":
        backbone_dtype = torch.float32
    elif amp == "fp16":
        backbone_dtype = torch.float16
    elif amp == "none":
        backbone_dtype = torch.float32
    else:
        backbone_dtype = torch.bfloat16

    occany_ckpt = args.occany_ckpt or ckpt_arg(
        ckpt_args,
        "occany_ckpt",
        str(REPO_ROOT / "checkpoints" / "occany_recon.pth"),
    )
    if not Path(occany_ckpt).is_file():
        raise FileNotFoundError(f"OccAny reconstruction checkpoint not found: {occany_ckpt}")

    model_kwargs = dict(
        occany_ckpt=occany_ckpt,
        c_lift=int(ckpt_arg(ckpt_args, "c_lift", 64)),
        num_classes=(
            len(UNIFIED_SSC_CLASS_NAMES)
            if bool(ckpt_arg(ckpt_args, "multi_dataset", True))
            else 20
        ),
        patch_size=int(ckpt_arg(ckpt_args, "patch_size", 16)),
        token_dim=int(ckpt_arg(ckpt_args, "token_dim", 768)),
        backbone_img_size=(
            int(ckpt_arg(ckpt_args, "height", 160)),
            int(ckpt_arg(ckpt_args, "width", 512)),
        ),
        backbone_dtype=backbone_dtype,
        fusion_attn_type="cross",
        num_frames=int(ckpt_arg(ckpt_args, "num_frames", 5)),
        freeze_backbone=bool(ckpt_arg(ckpt_args, "freeze_backbone", True)),
    )
    if exp in ("bevdetocc_lidar_dense_depth", "bevdetocc_lidar_pointmap_dense_depth"):
        model_kwargs["dense_depth_features"] = int(ckpt_arg(ckpt_args, "dense_depth_features", 128))

    model = MODEL_BY_EXP[exp](**model_kwargs).to(device)
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint {ckpt_path} does not contain a 'model' state_dict")
    status = model.load_state_dict(strip_module_prefix(ckpt["model"]), strict=False)
    print(
        f"[{ckpt_path.parent.name}] loaded exp={exp}: "
        f"missing={len(status.missing_keys)} unexpected={len(status.unexpected_keys)}"
    )
    model.eval()
    return model


def build_dataset(args: argparse.Namespace, first_ckpt_args: Dict, dataset_name: str):
    width = int(arg_or_ckpt(args, first_ckpt_args, "width", 512))
    height = int(arg_or_ckpt(args, first_ckpt_args, "height", 160))
    num_frames = int(arg_or_ckpt(args, first_ckpt_args, "num_frames", 5))
    max_points = int(arg_or_ckpt(args, first_ckpt_args, "max_points_per_sweep", 0))

    if dataset_name == "kitti":
        processed_root = args.processed_root or ckpt_arg(
            first_ckpt_args,
            "processed_root",
            str(REPO_ROOT / "data" / "kitti_processed"),
        )
        dataset = Kitti5FrameStage1LidarDataset(
            processed_root=processed_root,
            split="val",
            num_frames=num_frames,
            frame_stride=int(arg_or_ckpt(args, first_ckpt_args, "frame_stride", 4)),
            output_resolution=(width, height),
            cam_idx=0,
            velodyne_root=None,
            max_points_per_sweep=max_points,
        )
        return dataset, collate_stage1_lidar, processed_root

    processed_root = args.nuscenes_processed_root or ckpt_arg(
        first_ckpt_args,
        "nuscenes_processed_root",
        str(REPO_ROOT / "data" / "nuscenes_processed"),
    )
    dataset = NuScenes5FrameStage1LidarDataset(
        processed_root=processed_root,
        split="val",
        num_frames=num_frames,
        frame_stride=int(arg_or_ckpt(args, first_ckpt_args, "nuscenes_frame_stride", 1)),
        output_resolution=(width, height),
        max_points_per_sweep=max_points,
    )
    return dataset, collate_stage1_nuscenes_lidar, processed_root


@torch.no_grad()
def run_model_on_batch(
    model: torch.nn.Module,
    batch: Dict,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> torch.Tensor:
    views = move_views_to_device(batch["views"], device)
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
    ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype is not None
        else nullcontext()
    )
    with ctx:
        out = model(
            views,
            batch["T_target_from_refcam"].to(device, non_blocking=True),
            move_points_to_device(batch["points_per_frame"], device),
            batch["T_cam_from_velo"].to(device, non_blocking=True),
            batch["K_per_frame"].to(device, non_blocking=True),
            batch["image_hw"].to(device, non_blocking=True),
            return_depth=False,
            grid_config=grid_config,
        )
    return out["ssc_logit"] if isinstance(out, dict) else out


def occupied_voxels(label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = (label != 0) & (label != 255) & (label >= 0)
    return np.argwhere(mask), label[mask].astype(np.int64)


def voxel_centers(coords: np.ndarray, origin: np.ndarray, voxel_size: np.ndarray) -> np.ndarray:
    return origin[None, :] + (coords.astype(np.float32) + 0.5) * voxel_size[None, :]


def downsample_indices(n: int, max_n: int, seed: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_n, replace=False)


def scatter_semantic_voxels(
    ax,
    label: np.ndarray,
    title: str,
    origin: np.ndarray,
    voxel_size: np.ndarray,
    grid_size: np.ndarray,
    max_voxels: int,
    seed: int,
) -> Dict[str, int]:
    coords, vals = occupied_voxels(label)
    n_total = int(coords.shape[0])
    if n_total == 0:
        ax.text2D(0.5, 0.5, "no occupied voxels", transform=ax.transAxes, ha="center")
        shown = 0
    else:
        sel = downsample_indices(n_total, max_voxels, seed)
        coords = coords[sel]
        vals = vals[sel]
        centers = voxel_centers(coords, origin, voxel_size)
        colors = UNIFIED_COLORS[np.clip(vals, 0, len(UNIFIED_COLORS) - 1)] / 255.0
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            c=colors,
            s=2.0,
            alpha=0.9,
            linewidths=0.0,
            depthshade=False,
        )
        shown = int(len(sel))

    xyz_min = origin
    xyz_max = origin + grid_size.astype(np.float32) * voxel_size
    ax.set_title(f"{title}\noccupied={n_total}, shown={shown}", fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(float(xyz_min[0]), float(xyz_max[0]))
    ax.set_ylim(float(xyz_min[1]), float(xyz_max[1]))
    ax.set_zlim(float(xyz_min[2]), float(xyz_max[2]))
    ax.set_box_aspect(
        (
            float(xyz_max[0] - xyz_min[0]),
            float(xyz_max[1] - xyz_min[1]),
            max(float(xyz_max[2] - xyz_min[2]), 1e-3),
        )
    )
    return {"occupied": n_total, "shown": shown}


def label_to_rgb(label: np.ndarray) -> np.ndarray:
    safe = np.clip(label.astype(np.int64), 0, len(UNIFIED_COLORS) - 1)
    rgb = UNIFIED_COLORS[safe].copy()
    rgb[(label == 0) | (label == 255)] = 0
    return rgb


def bev_projection(label: np.ndarray) -> np.ndarray:
    occ = (label != 0) & (label != 255)
    z_any = occ.any(axis=2)
    z_idx = (occ * np.arange(label.shape[2], dtype=np.int32)[None, None, :]).argmax(axis=2)
    top = np.take_along_axis(label, z_idx[..., None], axis=2)[..., 0]
    top[~z_any] = 0
    return label_to_rgb(top).transpose(1, 0, 2)


def diff_projection(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    valid = gt != 255
    gt_occ = (gt != 0) & valid
    pred_occ = (pred != 0) & valid
    same = (pred == gt) & gt_occ
    fp = pred_occ & ~gt_occ
    fn = gt_occ & ~pred_occ
    mismatch = pred_occ & gt_occ & (pred != gt)
    diff = np.zeros(gt.shape[:2] + (3,), dtype=np.uint8)
    diff[same.any(axis=2)] = [0, 180, 0]
    diff[fp.any(axis=2)] = [255, 0, 0]
    diff[fn.any(axis=2)] = [0, 80, 255]
    diff[mismatch.any(axis=2)] = [255, 180, 0]
    return diff.transpose(1, 0, 2)


def save_input_montage(sample: Dict, out_path: Path) -> None:
    views = sample["views"]
    imgs = torch.stack([view["img"].detach().cpu() for view in views], dim=0)
    imgs_u8 = convert_images_to_uint8_hwc(imgs)
    frame_ids = tuple(int(v) for v in sample["frame_ids"])
    fig, axes = plt.subplots(1, len(views), figsize=(4.0 * len(views), 2.2), dpi=160)
    if len(views) == 1:
        axes = [axes]
    for ax, img, frame_id in zip(axes, imgs_u8, frame_ids):
        ax.imshow(img)
        ax.set_title(f"frame {frame_id}", fontsize=9)
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path)
    plt.close(fig)


def save_multi_3d(
    gt: np.ndarray,
    preds: Sequence[np.ndarray],
    labels: Sequence[str],
    out_path: Path,
    title: str,
    origin: np.ndarray,
    voxel_size: np.ndarray,
    grid_size: np.ndarray,
    max_voxels: int,
    seed: int,
    elev: float,
    azim: float,
) -> None:
    ncols = 1 + len(preds)
    fig = plt.figure(figsize=(4.5 * ncols, 5.2), dpi=170)
    axes = [fig.add_subplot(1, ncols, idx + 1, projection="3d") for idx in range(ncols)]
    scatter_semantic_voxels(axes[0], gt, "GT", origin, voxel_size, grid_size, max_voxels, seed)
    for idx, (pred, label) in enumerate(zip(preds, labels), start=1):
        scatter_semantic_voxels(
            axes[idx],
            pred,
            label,
            origin,
            voxel_size,
            grid_size,
            max_voxels,
            seed + idx,
        )
    for ax in axes:
        ax.view_init(elev=elev, azim=azim)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_multi_bev(
    gt: np.ndarray,
    preds: Sequence[np.ndarray],
    labels: Sequence[str],
    out_path: Path,
    title: str,
) -> None:
    ncols = 1 + len(preds)
    fig, axes = plt.subplots(2, ncols, figsize=(4.0 * ncols, 7.2), dpi=170)
    axes[0, 0].imshow(bev_projection(gt), origin="lower")
    axes[0, 0].set_title("GT BEV")
    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.0,
        0.7,
        "Diff legend\n"
        "green: correct occupied\n"
        "red: false positive\n"
        "blue: false negative\n"
        "orange: semantic mismatch",
        fontsize=9,
        transform=axes[1, 0].transAxes,
    )
    for col, (pred, label) in enumerate(zip(preds, labels), start=1):
        axes[0, col].imshow(bev_projection(pred), origin="lower")
        axes[0, col].set_title(f"{label} BEV")
        axes[1, col].imshow(diff_projection(pred, gt), origin="lower")
        axes[1, col].set_title(f"{label} diff")
    for ax in axes.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def valid_classes_for(dataset_name: str) -> List[int]:
    if dataset_name == "kitti":
        return list(range(1, 20))
    return sorted({int(v) for v in NUSCENES_TO_UNIFIED.tolist() if int(v) != 0})


def compute_summary(pred: np.ndarray, gt: np.ndarray, dataset_name: str) -> Dict:
    valid = gt != 255
    ssc = SSCMetrics(
        n_classes=len(UNIFIED_SSC_CLASS_NAMES),
        class_names=list(UNIFIED_SSC_CLASS_NAMES),
        other_class=None,
        ignore_other_class_in_mIoU=False,
        empty_class=0,
        valid_classes=valid_classes_for(dataset_name),
    )
    ssc.add_batch(pred[None].astype(np.int64), gt[None].astype(np.int64))
    stats = ssc.get_stats()
    return {
        "voxel_acc": float(((pred == gt) & valid).sum() / max(int(valid.sum()), 1)),
        "ssc_precision": float(stats["precision"]),
        "ssc_recall": float(stats["recall"]),
        "ssc_iou": float(stats["iou"]),
        "ssc_mIoU": float(stats["mIoU"]),
        "pred_occupied": int(((pred != 0) & valid).sum()),
        "gt_occupied": int(((gt != 0) & valid).sum()),
        "class_names": [str(v) for v in stats["class_names"]],
        "iou_per_class": [float(v) for v in stats["iou_per_class"]],
    }


def sample_stem(dataset_name: str, sample: Dict, sample_idx: int) -> Tuple[str, str]:
    sequence = str(sample["sequence"])
    target = sample["target_frame_id"]
    if isinstance(target, (list, tuple)):
        target = target[0]
    return (
        f"{dataset_name}_idx{sample_idx:06d}_{sequence}_{int(target):06d}",
        f"{dataset_name} idx={sample_idx}, seq={sequence}, target={int(target):06d}, "
        f"frames={tuple(int(v) for v in sample['frame_ids'])}",
    )


@torch.no_grad()
def visualize_dataset(
    args: argparse.Namespace,
    dataset_name: str,
    ckpt_paths: Sequence[Path],
    first_ckpt: Dict,
    first_ckpt_args: Dict,
    device: torch.device,
) -> Dict:
    dataset, collate_fn, data_root = build_dataset(args, first_ckpt_args, dataset_name)
    if len(dataset) == 0:
        raise RuntimeError(f"{dataset_name} val dataset is empty under {data_root}")

    dataset_specific_idx = (
        args.kitti_sample_idx if dataset_name == "kitti" else args.nuscenes_sample_idx
    )
    requested_idx = dataset_specific_idx if dataset_specific_idx is not None else args.sample_idx
    sample_idx = random.Random(args.seed).randrange(len(dataset)) if requested_idx is None else int(requested_idx)
    if sample_idx < 0 or sample_idx >= len(dataset):
        raise IndexError(f"sample_idx={sample_idx} out of range [0, {len(dataset) - 1}]")

    sample = dataset[sample_idx]
    batch = collate_fn([sample])
    gt = batch["voxel_label"][0].cpu().numpy().astype(np.uint8)

    grid = GRID_CONFIGS[dataset_name]
    origin = np.array(grid.full_voxel_origin, dtype=np.float32)
    voxel_size = np.array(grid.full_voxel_size, dtype=np.float32)
    grid_size = np.array(grid.full_grid_size, dtype=np.int64)

    stem, title = sample_stem(dataset_name, sample, sample_idx)
    out_dir = Path(args.output_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    save_input_montage(sample, out_dir / "input_frames.png")

    preds: List[np.ndarray] = []
    summaries: Dict[str, Dict] = {}
    ckpt_meta: Dict[str, Dict] = {}
    for ckpt_path, label in zip(ckpt_paths, args.labels):
        ckpt = first_ckpt if ckpt_path == ckpt_paths[0] else load_checkpoint(ckpt_path)
        ckpt_args = ckpt.get("args", {})
        ckpt_args = dict(ckpt_args) if isinstance(ckpt_args, dict) else {}
        model = build_model_from_checkpoint(ckpt, ckpt_path, args, device)
        logits = run_model_on_batch(
            model,
            batch,
            device,
            amp_dtype_for(device, str(ckpt_arg(ckpt_args, "amp", "bf16"))),
        )
        pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        preds.append(pred)
        summaries[label] = compute_summary(pred, gt, dataset_name)
        ckpt_meta[label] = {
            "checkpoint": str(ckpt_path),
            "epoch": ckpt.get("epoch", None),
            "exp": ckpt_arg(ckpt_args, "exp", None),
            "dataset_ratio": ckpt_arg(ckpt_args, "dataset_ratio", None),
        }
        del model, logits, ckpt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_multi_3d(
        gt,
        preds,
        args.labels,
        out_dir / "compare_semantic_3d.png",
        title,
        origin,
        voxel_size,
        grid_size,
        args.max_voxels_plot,
        args.seed,
        args.elev,
        args.azim,
    )
    save_multi_bev(gt, preds, args.labels, out_dir / "compare_semantic_bev.png", title)

    meta = {
        "dataset": dataset_name,
        "data_root": data_root,
        "sample_idx": sample_idx,
        "sequence": str(sample["sequence"]),
        "target_frame_id": int(sample["target_frame_id"]),
        "frame_ids": [int(v) for v in sample["frame_ids"]],
        "checkpoints": ckpt_meta,
        "metrics": summaries,
        "class_names": list(UNIFIED_SSC_CLASS_NAMES),
        "valid_classes": valid_classes_for(dataset_name),
        "output_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(meta, f, indent=2)

    if args.save_grids:
        payload = {"gt": gt}
        for label, pred in zip(args.labels, preds):
            payload[f"pred_{label}"] = pred
        np.savez_compressed(out_dir / "voxel_grids.npz", **payload)

    print(f"[{dataset_name}] saved visualizations to {out_dir}")
    for label in args.labels:
        metric = summaries[label]
        print(f"  {label}: IoU={metric['ssc_iou']:.4f}, mIoU={metric['ssc_mIoU']:.4f}")
    return meta


@torch.no_grad()
def main() -> None:
    args = get_args()
    if len(args.labels) != len(args.ckpts):
        raise ValueError("--labels length must match --ckpts length")

    register_legacy_checkpoint_modules()
    toggle_memory_efficient_attention(enabled=False)

    ckpt_paths = [Path(path) for path in args.ckpts]
    first_ckpt = load_checkpoint(ckpt_paths[0])
    first_ckpt_args = first_ckpt.get("args", {})
    first_ckpt_args = dict(first_ckpt_args) if isinstance(first_ckpt_args, dict) else {}
    device = resolve_device(args.device)

    dataset_names = ("kitti", "nuscenes") if args.dataset == "both" else (args.dataset,)
    metas = [
        visualize_dataset(args, name, ckpt_paths, first_ckpt, first_ckpt_args, device)
        for name in dataset_names
    ]

    print("Saved outputs:")
    for meta in metas:
        print(f"  {meta['dataset']}: {meta['output_dir']}")


if __name__ == "__main__":
    main()
