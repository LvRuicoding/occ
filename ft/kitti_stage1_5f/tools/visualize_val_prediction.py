"""Visualize one Stage-1 validation prediction against SemanticKITTI GT.

Example:
  python -m ft.kitti_stage1_5f.tools.visualize_val_prediction

  python -m ft.kitti_stage1_5f.tools.visualize_val_prediction \
    --ckpt output/kitti_stage1_5f_4gpu/checkpoint-last.pth \
    --sample_idx 12
"""
from __future__ import annotations

from .. import _paths  # noqa: F401

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from occany.model.must3r_blocks.attention import toggle_memory_efficient_attention
from occany.metrics.ssc import SSCMetrics
from occany.utils.checkpoint_io import register_legacy_checkpoint_modules
from occany.utils.image_util import convert_images_to_uint8_hwc

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
from ..models import (
    Stage1SSCBEVDetOccLidarModel,
    Stage1SSCModel,
    Stage1SSCMonoLidarModel,
    Stage1SSCMonoModel,
)


LIDAR_EXPS = ("monoscene_lidar", "bevdetocc_lidar")


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE1_CKPT = REPO_ROOT / "output" / "kitti_stage1_5f_4gpu" / "checkpoint-last.pth"
DEFAULT_PROCESSED_ROOT = REPO_ROOT / "data" / "kitti_processed"
DEFAULT_KITTIODO_ROOT = REPO_ROOT / "raw_data" / "semantickitti_occany_root"
DEFAULT_FALLBACK_KITTIODO_ROOT = REPO_ROOT / "raw_data" / "OpenDataLab___KITTI_Odometry_2012"
DEFAULT_OCCANY_CKPT = REPO_ROOT / "checkpoints" / "occany_recon.pth"
DEFAULT_VELODYNE_ROOT = REPO_ROOT / "data" / "kitti"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visuals" / "kitti_stage1_val_prediction_compare"
MAJOR_CLASS_NAMES = ("road", "sidewalk", "building", "vegetation", "terrain", "car")
MAJOR_CLASS_IDS = tuple(KITTI_SSC_CLASS_NAMES.index(name) for name in MAJOR_CLASS_NAMES)


SEMANTIC_KITTI_COLORS = np.array(
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
    ],
    dtype=np.uint8,
)


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Visualize Stage-1 val prediction vs GT")
    p.add_argument("--ckpt", default=str(DEFAULT_STAGE1_CKPT), type=str,
                   help="Fine-tuned Stage-1 checkpoint from train.py.")
    p.add_argument("--sample_idx", default=None, type=int,
                   help="Validation dataset index. If omitted, sample one randomly.")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), type=str)
    p.add_argument("--processed_root", default=None, type=str)
    p.add_argument("--kittiodo_root", default=None, type=str)
    p.add_argument("--occany_ckpt", default=None, type=str,
                   help="Original OccAny reconstruction checkpoint. Defaults to checkpoint args.")
    p.add_argument("--model_type", "--exp", dest="exp",
                   choices=["light", "monoscene", "monoscene_lidar", "bevdetocc_lidar"], default=None,
                   help="Model variant. If omitted, read from checkpoint args.")
    p.add_argument("--velodyne_root", default=None, type=str,
                   help="Raw KITTI Odometry root for LiDAR checkpoints.")
    p.add_argument("--max_points_per_sweep", default=None, type=int)
    p.add_argument("--width", default=None, type=int)
    p.add_argument("--height", default=None, type=int)
    p.add_argument("--num_frames", default=None, type=int)
    p.add_argument("--frame_stride", default=None, type=int)
    p.add_argument("--fusion_attn_type", choices=["self", "cross"], default=None)
    p.add_argument("--fusion3d", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--fusion3d_seq_len", type=int, default=None)
    p.add_argument("--fusion3d_num_heads", type=int, default=None)
    p.add_argument("--fusion3d_ffn_ratio", type=float, default=None)
    p.add_argument("--fusion3d_alpha_init", type=float, default=None)
    p.add_argument("--post_lift_lidar", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--post_lift_lidar_channels", type=int, default=None)
    p.add_argument("--memory_voxel", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--memory_voxel_kernel", type=int, default=None)
    p.add_argument("--memory_voxel_num_heads", type=int, default=None)
    p.add_argument("--memory_voxel_num_layers", type=int, default=None)
    p.add_argument("--memory_voxel_ffn_ratio", type=float, default=None)
    p.add_argument("--memory_voxel_alpha_init", type=float, default=None)
    p.add_argument("--memory_voxel_d_voxel", type=int, default=None)
    p.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    p.add_argument("--max_voxels_plot", default=60000, type=int)
    p.add_argument("--elev", default=22.0, type=float)
    p.add_argument("--azim", default=-55.0, type=float)
    p.add_argument("--save_logits", action="store_true",
                   help="Also save full logits to npz; this is large.")
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def ckpt_arg(ckpt_args: Dict, name: str, default):
    return ckpt_args.get(name, default) if isinstance(ckpt_args, dict) else default


def override_or_ckpt(args: argparse.Namespace, ckpt_args: Dict, name: str, default):
    value = getattr(args, name)
    return value if value is not None else ckpt_arg(ckpt_args, name, default)


def infer_exp(args: argparse.Namespace, ckpt: Dict, ckpt_args: Dict) -> str:
    if args.exp is not None:
        return args.exp
    exp = ckpt_arg(ckpt_args, "exp", None)
    if exp in {"light", "monoscene", "monoscene_lidar", "bevdetocc_lidar"}:
        return str(exp)
    if "model" in ckpt:
        return "bevdetocc_lidar"
    if "fusion" in ckpt:
        return "monoscene_lidar"
    occ_keys = ckpt.get("occ_head", {}).keys()
    if any(str(k).startswith("unet3d.") for k in occ_keys):
        return "monoscene"
    return "light"


def move_points_to_device(
    points_per_frame: List[List[torch.Tensor]], device: torch.device
) -> List[List[torch.Tensor]]:
    return [
        [pts.to(device, non_blocking=True) for pts in per_sample]
        for per_sample in points_per_frame
    ]


def calib_root_has_tr(root: Path) -> bool:
    candidates = [
        root / "dataset" / "sequences" / "00" / "calib.txt",
        root / "sequences" / "00" / "calib.txt",
    ]
    for path in candidates:
        if path.exists() and "Tr:" in path.read_text():
            return True
    return False


def resolve_paths(args: argparse.Namespace, ckpt_args: Dict) -> Tuple[str, str, str]:
    processed_root = args.processed_root or ckpt_arg(
        ckpt_args, "processed_root", str(DEFAULT_PROCESSED_ROOT)
    )
    kittiodo_root = args.kittiodo_root or ckpt_arg(
        ckpt_args, "kittiodo_root", str(DEFAULT_KITTIODO_ROOT)
    )
    occany_ckpt = args.occany_ckpt or ckpt_arg(
        ckpt_args, "occany_ckpt", str(DEFAULT_OCCANY_CKPT)
    )

    if not Path(processed_root).exists():
        raise FileNotFoundError(f"processed_root does not exist: {processed_root}")
    if not calib_root_has_tr(Path(kittiodo_root)):
        if calib_root_has_tr(DEFAULT_FALLBACK_KITTIODO_ROOT):
            print(
                f"kittiodo_root={kittiodo_root} does not expose calib.txt with Tr; "
                f"falling back to {DEFAULT_FALLBACK_KITTIODO_ROOT}"
            )
            kittiodo_root = str(DEFAULT_FALLBACK_KITTIODO_ROOT)
        else:
            raise FileNotFoundError(
                f"kittiodo_root does not contain KITTI odometry calib with Tr: {kittiodo_root}"
            )
    if not Path(occany_ckpt).is_file():
        raise FileNotFoundError(f"OccAny reconstruction checkpoint not found: {occany_ckpt}")
    return processed_root, kittiodo_root, occany_ckpt


def move_views_to_device(
    views: List[Dict[str, torch.Tensor]], device: torch.device
) -> List[Dict[str, torch.Tensor]]:
    moved: List[Dict[str, torch.Tensor]] = []
    for view in views:
        out: Dict[str, torch.Tensor] = {}
        for key, val in view.items():
            out[key] = val.to(device, non_blocking=True) if isinstance(val, torch.Tensor) else val
        moved.append(out)
    return moved


def save_frame_montage(
    views: List[Dict[str, torch.Tensor]],
    frame_ids: Tuple[int, ...],
    out_path: Path,
) -> None:
    imgs = torch.stack([v["img"][0].detach().cpu() for v in views], dim=0)
    imgs_u8 = convert_images_to_uint8_hwc(imgs)
    fig, axes = plt.subplots(1, len(views), figsize=(4.0 * len(views), 2.2), dpi=160)
    if len(views) == 1:
        axes = [axes]
    for ax, img, fid in zip(axes, imgs_u8, frame_ids):
        ax.imshow(img)
        ax.set_title(f"frame {fid}", fontsize=9)
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path)
    plt.close(fig)


def occupied_voxels(label: np.ndarray, empty_class: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    mask = (label != empty_class) & (label != 255) & (label >= 0)
    coords = np.argwhere(mask)
    vals = label[mask].astype(np.int64)
    return coords, vals


def voxel_centers(coords: np.ndarray) -> np.ndarray:
    origin = np.array([0.0, -25.6, -2.0], dtype=np.float32)
    voxel_size = np.array([0.2, 0.2, 0.2], dtype=np.float32)
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
    max_voxels: int,
    seed: int,
) -> Tuple[int, int]:
    coords, vals = occupied_voxels(label)
    n_total = int(coords.shape[0])
    if n_total == 0:
        ax.text2D(0.5, 0.5, "no occupied voxels", transform=ax.transAxes, ha="center")
        return 0, 0

    sel = downsample_indices(n_total, max_voxels, seed)
    coords = coords[sel]
    vals = vals[sel]
    centers = voxel_centers(coords)
    colors = SEMANTIC_KITTI_COLORS[np.clip(vals, 0, len(SEMANTIC_KITTI_COLORS) - 1)] / 255.0
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
    ax.set_title(f"{title}\noccupied={n_total}, shown={len(sel)}")
    ax.set_xlabel("x / forward (m)")
    ax.set_ylabel("y / left (m)")
    ax.set_zlabel("z / up (m)")
    ax.set_xlim(0.0, 51.2)
    ax.set_ylim(-25.6, 25.6)
    ax.set_zlim(-2.0, 4.4)
    return n_total, int(len(sel))


def save_pred_gt_3d(
    pred: np.ndarray,
    gt: np.ndarray,
    out_path: Path,
    title: str,
    max_voxels: int,
    seed: int,
    elev: float,
    azim: float,
) -> None:
    fig = plt.figure(figsize=(15, 7), dpi=170)
    ax_gt = fig.add_subplot(121, projection="3d")
    ax_pred = fig.add_subplot(122, projection="3d")
    scatter_semantic_voxels(ax_gt, gt, "GT", max_voxels, seed)
    scatter_semantic_voxels(ax_pred, pred, "Prediction", max_voxels, seed + 1)
    for ax in (ax_gt, ax_pred):
        ax.view_init(elev=elev, azim=azim)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_binary_occupied_3d(
    pred: np.ndarray,
    gt: np.ndarray,
    out_path: Path,
    title: str,
    max_voxels: int,
    seed: int,
    elev: float,
    azim: float,
) -> None:
    pred_bin = ((pred != 0) & (pred != 255)).astype(np.uint8)
    gt_bin = ((gt != 0) & (gt != 255)).astype(np.uint8)
    fig = plt.figure(figsize=(15, 7), dpi=170)
    ax_gt = fig.add_subplot(121, projection="3d")
    ax_pred = fig.add_subplot(122, projection="3d")
    scatter_semantic_voxels(ax_gt, gt_bin, "GT occupied/free", max_voxels, seed)
    scatter_semantic_voxels(ax_pred, pred_bin, "Pred occupied/free", max_voxels, seed + 1)
    for ax in (ax_gt, ax_pred):
        ax.view_init(elev=elev, azim=azim)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def keep_major_classes(label: np.ndarray) -> np.ndarray:
    out = np.zeros_like(label)
    keep = np.isin(label, np.array(MAJOR_CLASS_IDS, dtype=label.dtype))
    out[keep] = label[keep]
    return out


def save_major_classes_3d(
    pred: np.ndarray,
    gt: np.ndarray,
    out_path: Path,
    title: str,
    max_voxels: int,
    seed: int,
    elev: float,
    azim: float,
) -> None:
    pred_major = keep_major_classes(pred)
    gt_major = keep_major_classes(gt)
    fig = plt.figure(figsize=(15, 7), dpi=170)
    ax_gt = fig.add_subplot(121, projection="3d")
    ax_pred = fig.add_subplot(122, projection="3d")
    scatter_semantic_voxels(ax_gt, gt_major, "GT major classes", max_voxels, seed)
    scatter_semantic_voxels(ax_pred, pred_major, "Pred major classes", max_voxels, seed + 1)
    for ax in (ax_gt, ax_pred):
        ax.view_init(elev=elev, azim=azim)
    fig.suptitle(title + "\nmajor classes: " + ", ".join(MAJOR_CLASS_NAMES))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def label_to_rgb(label: np.ndarray) -> np.ndarray:
    safe = np.clip(label.astype(np.int64), 0, len(SEMANTIC_KITTI_COLORS) - 1)
    rgb = SEMANTIC_KITTI_COLORS[safe]
    empty = (label == 0) | (label == 255)
    rgb[empty] = 0
    return rgb


def bev_projection(label: np.ndarray) -> np.ndarray:
    """Top-down projection: nearest occupied Z gets shown for each X/Y."""
    occ = (label != 0) & (label != 255)
    z_any = occ.any(axis=2)
    # Prefer higher voxels when several classes share an X/Y column.
    z_idx = (occ * np.arange(label.shape[2], dtype=np.int32)[None, None, :]).argmax(axis=2)
    top = np.take_along_axis(label, z_idx[..., None], axis=2)[..., 0]
    top[~z_any] = 0
    return label_to_rgb(top).transpose(1, 0, 2)


def save_bev_compare(pred: np.ndarray, gt: np.ndarray, out_path: Path, title: str) -> None:
    pred_bev = bev_projection(pred)
    gt_bev = bev_projection(gt)
    diff = np.zeros_like(pred_bev)
    gt_occ = (gt != 0) & (gt != 255)
    pred_occ = (pred != 0) & (pred != 255)
    same = (pred == gt) & gt_occ
    fp = pred_occ & ~gt_occ
    fn = gt_occ & ~pred_occ
    mismatch = pred_occ & gt_occ & (pred != gt)
    diff2d = np.zeros(gt.shape[:2] + (3,), dtype=np.uint8)
    diff2d[same.any(axis=2)] = [0, 180, 0]
    diff2d[fp.any(axis=2)] = [255, 0, 0]
    diff2d[fn.any(axis=2)] = [0, 80, 255]
    diff2d[mismatch.any(axis=2)] = [255, 180, 0]
    diff = diff2d.transpose(1, 0, 2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=170)
    for ax, img, name in zip(axes, [gt_bev, pred_bev, diff], ["GT BEV", "Pred BEV", "Diff"]):
        ax.imshow(img, origin="lower")
        ax.set_title(name)
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_binary_bev_compare(pred: np.ndarray, gt: np.ndarray, out_path: Path, title: str) -> None:
    pred_bin = ((pred != 0) & (pred != 255)).astype(np.uint8)
    gt_bin = ((gt != 0) & (gt != 255)).astype(np.uint8)
    save_bev_compare(pred_bin, gt_bin, out_path, title + " | occupied/free")


def save_major_bev_compare(pred: np.ndarray, gt: np.ndarray, out_path: Path, title: str) -> None:
    save_bev_compare(
        keep_major_classes(pred),
        keep_major_classes(gt),
        out_path,
        title + " | major classes: " + ", ".join(MAJOR_CLASS_NAMES),
    )


def compute_summary(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    valid = gt != 255
    occ_gt = (gt != 0) & valid
    occ_pred = (pred != 0) & valid
    correct = (pred == gt) & valid
    inter = occ_gt & occ_pred
    union = occ_gt | occ_pred

    ssc = SSCMetrics(
        n_classes=20,
        class_names=list(KITTI_SSC_CLASS_NAMES),
        other_class=None,
        ignore_other_class_in_mIoU=False,
        empty_class=0,
    )
    ssc.add_batch(pred[None].astype(np.int64), gt[None].astype(np.int64))
    stats = ssc.get_stats()
    iou_per_class = [float(v) for v in stats["iou_per_class"]]

    out = {
        "voxel_acc": float(correct.sum() / max(valid.sum(), 1)),
        "occ_iou": float(inter.sum() / max(union.sum(), 1)),
        "gt_occupied": int(occ_gt.sum()),
        "pred_occupied": int(occ_pred.sum()),
        "valid_voxels": int(valid.sum()),
        "ssc_precision": float(stats["precision"]),
        "ssc_recall": float(stats["recall"]),
        "ssc_iou": float(stats["iou"]),
        "ssc_mIoU": float(stats["mIoU"]),
        "ssc_iou_per_class": iou_per_class,
        "major_class_ids": list(MAJOR_CLASS_IDS),
        "major_class_names": list(MAJOR_CLASS_NAMES),
    }
    return out


def build_model(
    ckpt: Dict,
    ckpt_args: Dict,
    occany_ckpt: str,
    exp: str,
    device: torch.device,
) -> torch.nn.Module:
    amp = ckpt_arg(ckpt_args, "amp", "bf16")
    if device.type != "cuda":
        backbone_dtype = torch.float32
    elif amp == "fp16":
        backbone_dtype = torch.float16
    elif amp == "none":
        backbone_dtype = torch.float32
    else:
        backbone_dtype = torch.bfloat16

    if exp == "bevdetocc_lidar":
        model_cls = Stage1SSCBEVDetOccLidarModel
    elif exp == "monoscene_lidar":
        model_cls = Stage1SSCMonoLidarModel
    elif exp == "monoscene":
        model_cls = Stage1SSCMonoModel
    else:
        model_cls = Stage1SSCModel

    model_kwargs = dict(
        occany_ckpt=occany_ckpt,
        c_lift=int(ckpt_arg(ckpt_args, "c_lift", 64)),
        num_classes=20,
        patch_size=int(ckpt_arg(ckpt_args, "patch_size", 16)),
        token_dim=int(ckpt_arg(ckpt_args, "token_dim", 768)),
        backbone_img_size=(
            int(ckpt_arg(ckpt_args, "height", 160)),
            int(ckpt_arg(ckpt_args, "width", 512)),
        ),
        backbone_dtype=backbone_dtype,
    )
    if exp == "bevdetocc_lidar":
        model_kwargs["fusion_attn_type"] = "cross"
        model_kwargs["num_frames"] = int(ckpt_arg(ckpt_args, "num_frames", 5))
    if exp == "monoscene_lidar":
        model_kwargs["fusion_attn_type"] = ckpt_arg(ckpt_args, "fusion_attn_type", "self")
        model_kwargs["fusion3d_enabled"] = bool(ckpt_arg(ckpt_args, "fusion3d", False))
        model_kwargs["fusion3d_seq_len"] = int(ckpt_arg(ckpt_args, "fusion3d_seq_len", 80))
        fusion3d_num_heads = ckpt_arg(ckpt_args, "fusion3d_num_heads", None)
        model_kwargs["fusion3d_num_heads"] = (
            int(fusion3d_num_heads) if fusion3d_num_heads is not None else None
        )
        model_kwargs["fusion3d_ffn_ratio"] = float(
            ckpt_arg(ckpt_args, "fusion3d_ffn_ratio", 2.0)
        )
        model_kwargs["fusion3d_alpha_init"] = float(
            ckpt_arg(ckpt_args, "fusion3d_alpha_init", 0.0)
        )
        model_kwargs["post_lift_lidar_enabled"] = bool(
            ckpt_arg(ckpt_args, "post_lift_lidar", False)
        )
        model_kwargs["post_lift_lidar_channels"] = int(
            ckpt_arg(ckpt_args, "post_lift_lidar_channels", 32)
        )
        model_kwargs["memory_voxel_enabled"] = bool(
            ckpt_arg(ckpt_args, "memory_voxel", False)
        )
        model_kwargs["memory_voxel_kernel"] = int(
            ckpt_arg(ckpt_args, "memory_voxel_kernel", 7)
        )
        model_kwargs["memory_voxel_num_heads"] = int(
            ckpt_arg(ckpt_args, "memory_voxel_num_heads", 4)
        )
        model_kwargs["memory_voxel_num_layers"] = int(
            ckpt_arg(ckpt_args, "memory_voxel_num_layers", 2)
        )
        model_kwargs["memory_voxel_ffn_ratio"] = float(
            ckpt_arg(ckpt_args, "memory_voxel_ffn_ratio", 2.0)
        )
        model_kwargs["memory_voxel_alpha_init"] = float(
            ckpt_arg(ckpt_args, "memory_voxel_alpha_init", 0.0)
        )
        model_kwargs["memory_voxel_d_voxel"] = int(
            ckpt_arg(ckpt_args, "memory_voxel_d_voxel", 128)
        )
        model_kwargs["num_frames"] = int(ckpt_arg(ckpt_args, "num_frames", 5))

    model = model_cls(**model_kwargs).to(device)
    if exp == "bevdetocc_lidar":
        if "model" not in ckpt:
            raise KeyError("bevdetocc_lidar checkpoint must contain a 'model' state_dict.")
        status = model.load_state_dict(ckpt["model"], strict=False)
        print(
            "[visualize:bevdetocc_lidar] loaded non-backbone model state: "
            f"missing={len(status.missing_keys)} unexpected={len(status.unexpected_keys)}"
        )
        model.eval()
        return model

    model.lifting.load_state_dict(ckpt["lifting"], strict=True)
    model.occ_head.load_state_dict(ckpt["occ_head"], strict=True)
    if exp == "monoscene_lidar":
        if "fusion" not in ckpt:
            raise KeyError("monoscene_lidar checkpoint must contain a 'fusion' state_dict.")
        model.fusion.load_state_dict(ckpt["fusion"], strict=True)
        if model.post_lift_lidar is not None:
            model.post_lift_lidar.load_state_dict(ckpt["post_lift_lidar"], strict=True)
            model.post_lift_fuse.load_state_dict(ckpt["post_lift_fuse"], strict=True)
        if model.memory_fusion is not None:
            if "memory_fusion" not in ckpt:
                raise KeyError(
                    "memory-voxel checkpoint must contain a 'memory_fusion' state_dict."
                )
            model.memory_fusion.load_state_dict(ckpt["memory_fusion"], strict=True)
    model.eval()
    return model


def build_dataset(
    exp: str,
    processed_root: str,
    kittiodo_root: str,
    velodyne_root: str,
    max_points_per_sweep: int,
    width: int,
    height: int,
    num_frames: int,
    frame_stride: int,
):
    common = dict(
        processed_root=processed_root,
        kittiodo_root=kittiodo_root,
        split="val",
        num_frames=num_frames,
        frame_stride=frame_stride,
        output_resolution=(width, height),
        cam_idx=0,
    )
    if exp == "bevdetocc_lidar":
        return Kitti5FrameStage1LidarDataset(
            velodyne_root=velodyne_root,
            max_points_per_sweep=max_points_per_sweep,
            **common,
        )
    if exp == "monoscene_lidar":
        return Kitti5FrameStage1MonoLidarDataset(
            velodyne_root=velodyne_root,
            max_points_per_sweep=max_points_per_sweep,
            **common,
        )
    if exp == "monoscene":
        return Kitti5FrameStage1MonoDataset(**common)
    return Kitti5FrameStage1Dataset(**common)


def collate_for_exp(exp: str, samples):
    if exp == "bevdetocc_lidar":
        return collate_stage1_lidar(samples)
    if exp == "monoscene_lidar":
        return collate_stage1_mono_lidar(samples)
    if exp == "monoscene":
        return collate_stage1_mono(samples)
    return collate_stage1(samples)


def run_model(model: torch.nn.Module, batch: Dict, exp: str, device: torch.device) -> torch.Tensor:
    views = move_views_to_device(batch["views"], device)
    T_target_from_refcam = batch["T_target_from_refcam"].to(device, non_blocking=True)
    if exp in LIDAR_EXPS:
        out = model(
            views,
            T_target_from_refcam,
            move_points_to_device(batch["points_per_frame"], device),
            batch["T_cam_from_velo"].to(device, non_blocking=True),
            batch["K_per_frame"].to(device, non_blocking=True),
            batch["image_hw"].to(device, non_blocking=True),
        )
    else:
        out = model(views, T_target_from_refcam)
    return out["ssc_logit"] if isinstance(out, dict) else out


@torch.no_grad()
def main() -> None:
    args = get_args()
    register_legacy_checkpoint_modules()
    toggle_memory_efficient_attention(enabled=False)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    ckpt_args = dict(ckpt_args) if isinstance(ckpt_args, dict) else {}
    exp = infer_exp(args, ckpt, ckpt_args)
    ckpt_args["exp"] = exp
    for name in (
        "fusion_attn_type",
        "fusion3d",
        "fusion3d_seq_len",
        "fusion3d_num_heads",
        "fusion3d_ffn_ratio",
        "fusion3d_alpha_init",
        "post_lift_lidar",
        "post_lift_lidar_channels",
        "memory_voxel",
        "memory_voxel_kernel",
        "memory_voxel_num_heads",
        "memory_voxel_num_layers",
        "memory_voxel_ffn_ratio",
        "memory_voxel_alpha_init",
        "memory_voxel_d_voxel",
    ):
        value = getattr(args, name)
        if value is not None:
            ckpt_args[name] = value
    processed_root, kittiodo_root, occany_ckpt = resolve_paths(args, ckpt_args)

    width = int(override_or_ckpt(args, ckpt_args, "width", 512))
    height = int(override_or_ckpt(args, ckpt_args, "height", 160))
    num_frames = int(override_or_ckpt(args, ckpt_args, "num_frames", 5))
    frame_stride = int(override_or_ckpt(args, ckpt_args, "frame_stride", 1))
    velodyne_root = args.velodyne_root or ckpt_arg(
        ckpt_args, "velodyne_root", str(DEFAULT_VELODYNE_ROOT)
    )
    max_points_per_sweep = int(
        override_or_ckpt(args, ckpt_args, "max_points_per_sweep", 0)
    )
    if exp in LIDAR_EXPS and not Path(velodyne_root).exists():
        raise FileNotFoundError(f"velodyne_root does not exist: {velodyne_root}")

    dataset = build_dataset(
        exp=exp,
        processed_root=processed_root,
        kittiodo_root=kittiodo_root,
        velodyne_root=velodyne_root,
        max_points_per_sweep=max_points_per_sweep,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_stride=frame_stride,
    )
    if len(dataset) == 0:
        raise RuntimeError("Validation dataset is empty.")

    if args.sample_idx is None:
        sample_idx = random.Random(args.seed).randrange(len(dataset))
    else:
        sample_idx = int(args.sample_idx)
    if sample_idx < 0 or sample_idx >= len(dataset):
        raise IndexError(f"sample_idx={sample_idx} out of range [0, {len(dataset) - 1}]")

    device = resolve_device(args.device)
    model = build_model(ckpt, ckpt_args, occany_ckpt, exp, device)

    sample = dataset[sample_idx]
    batch = collate_for_exp(exp, [sample])
    views = move_views_to_device(batch["views"], device)
    target = batch["voxel_label"].to(device)

    logits = run_model(model, batch, exp, device)
    pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    gt = target[0].cpu().numpy().astype(np.uint8)
    summary = compute_summary(pred, gt)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seq = str(sample["sequence"])
    target_frame = int(sample["target_frame_id"])
    frame_ids = tuple(int(v) for v in sample["frame_ids"])
    stem = f"val_idx{sample_idx:06d}_{seq}_{target_frame:06d}"
    title = (
        f"val idx={sample_idx}, seq={seq}, target={target_frame:06d}, "
        f"frames={frame_ids}, epoch={ckpt.get('epoch', 'unknown')}"
    )

    save_frame_montage(views, frame_ids, out_dir / f"{stem}_frames.png")
    save_pred_gt_3d(
        pred,
        gt,
        out_dir / f"{stem}_pred_gt_3d.png",
        title,
        max_voxels=args.max_voxels_plot,
        seed=args.seed,
        elev=args.elev,
        azim=args.azim,
    )
    save_bev_compare(pred, gt, out_dir / f"{stem}_pred_gt_bev.png", title)
    save_binary_occupied_3d(
        pred,
        gt,
        out_dir / f"{stem}_occupied_free_3d.png",
        title,
        max_voxels=args.max_voxels_plot,
        seed=args.seed,
        elev=args.elev,
        azim=args.azim,
    )
    save_binary_bev_compare(pred, gt, out_dir / f"{stem}_occupied_free_bev.png", title)
    save_major_classes_3d(
        pred,
        gt,
        out_dir / f"{stem}_major_classes_3d.png",
        title,
        max_voxels=args.max_voxels_plot,
        seed=args.seed,
        elev=args.elev,
        azim=args.azim,
    )
    save_major_bev_compare(pred, gt, out_dir / f"{stem}_major_classes_bev.png", title)

    payload = {
        "pred": pred,
        "gt": gt,
        "pred_occupied": ((pred != 0) & (pred != 255)).astype(np.uint8),
        "gt_occupied": ((gt != 0) & (gt != 255)).astype(np.uint8),
        "pred_major": keep_major_classes(pred),
        "gt_major": keep_major_classes(gt),
        "frame_ids": np.array(frame_ids, dtype=np.int32),
        "sample_idx": np.array(sample_idx, dtype=np.int32),
        "target_frame_id": np.array(target_frame, dtype=np.int32),
        "sequence": seq,
    }
    if args.save_logits:
        payload["logits"] = logits[0].cpu().numpy().astype(np.float16)
    np.savez_compressed(out_dir / f"{stem}_pred_gt.npz", **payload)

    meta = {
        "stage1_ckpt": str(ckpt_path),
        "occany_ckpt": occany_ckpt,
        "model_type": exp,
        "processed_root": processed_root,
        "kittiodo_root": kittiodo_root,
        "velodyne_root": velodyne_root if exp in LIDAR_EXPS else None,
        "max_points_per_sweep": max_points_per_sweep if exp in LIDAR_EXPS else None,
        "sample_idx": sample_idx,
        "sequence": seq,
        "target_frame_id": target_frame,
        "frame_ids": frame_ids,
        "checkpoint_epoch": ckpt.get("epoch", None),
        "class_names": list(KITTI_SSC_CLASS_NAMES),
        **summary,
    }
    with open(out_dir / f"{stem}_summary.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved visualizations to {out_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
