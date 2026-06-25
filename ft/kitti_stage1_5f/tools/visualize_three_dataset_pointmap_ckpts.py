"""Visualize pointmap predictions from selected KITTI/nuScenes checkpoints.

Examples:
  python -m ft.kitti_stage1_5f.tools.visualize_three_dataset_pointmap_ckpts \
    --dataset kitti --sample_idx -1 --seed 0 --ckpts 01 03 --frame_idx 0

  python -m ft.kitti_stage1_5f.tools.visualize_three_dataset_pointmap_ckpts \
    --dataset nuscenes --sample_idx -1 --seed 1 --ckpts 02 03 --frame_idx 0

``--ckpts`` accepts checkpoint files, run directories, or aliases 01/02/03
under output/three_dataset_pointmap_postfusion_only_experiments.
"""
from __future__ import annotations

try:
    from .. import _paths  # noqa: F401  (must run before project imports)
except ImportError:  # Allows direct `python ft/.../visualize_three_dataset_pointmap_ckpts.py`.
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from ft.kitti_stage1_5f import _paths  # noqa: F401

import argparse
import json
import random
from copy import copy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from occany.model.must3r_blocks.attention import toggle_memory_efficient_attention
from occany.utils.checkpoint_io import register_legacy_checkpoint_modules
from occany.utils.image_util import convert_images_to_uint8_hwc

from ft.kitti_stage1_5f.datasets import (
    collate_stage1_dense_depth,
    collate_stage1_lidar_dense_depth,
    collate_stage1_nuscenes_lidar,
)
from ft.kitti_stage1_5f.datasets.unified_occ import GRID_CONFIGS
from ft.kitti_stage1_5f.pointmap_metrics import valid_pointmap_mask
from ft.kitti_stage1_5f.tools.eval_pointmap_quality import (
    POINTMAP_EVAL_EXPS,
    POINTMAP_LIDAR_EXPS,
    _amp_context,
    _build_pointmap_model,
    _ckpt_arg,
    _load_pointmap_weights,
    _resolve_ckpt_path,
)
from ft.kitti_stage1_5f.tools.train import (
    POINTMAP_DENSE_DEPTH_DATA_EXPS,
    _build_kitti_dataset,
    _build_nuscenes_dataset,
    _model_forward,
    _stack_cam2world_from_views,
)
from ft.kitti_stage1_5f.tools.visualize_pointmap import (
    collect_pixel_projection_points,
    plot_frame,
    save_pixel_projection_3d_plot,
    selected_frames,
    to_numpy,
    write_metrics,
    write_rgb_ply,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CKPT_ROOT = (
    REPO_ROOT / "output" / "three_dataset_pointmap_postfusion_only_experiments"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "visuals" / "three_dataset_pointmap_postfusion_only"


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Visualize one KITTI/nuScenes val sample for one or more pointmap checkpoints"
    )
    p.add_argument("--dataset", choices=["kitti", "nuscenes"], required=True)
    p.add_argument(
        "--sample_idx",
        type=int,
        default=-1,
        help="Validation index. Use -1 to choose a random sample with --seed.",
    )
    p.add_argument("--seed", default=0, type=int)
    p.add_argument(
        "--ckpts",
        nargs="+",
        required=True,
        help="Checkpoint files, run directories, or aliases such as 01 03.",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels matching --ckpts. Defaults to checkpoint run names.",
    )
    p.add_argument("--ckpt_root", default=str(DEFAULT_CKPT_ROOT), type=str)
    p.add_argument("--checkpoint_name", default="checkpoint-last.pth", type=str)
    p.add_argument("--output_dir", default=None, type=str)
    p.add_argument("--frame_idx", default=0, type=int, help="Frame index, or -1 for all frames.")
    p.add_argument("--device", default="auto", type=str, help="auto, cuda, cuda:0, or cpu.")
    p.add_argument("--amp", choices=["bf16", "fp16", "none"], default=None)
    p.add_argument("--max_points_plot", default=20000, type=int)
    p.add_argument("--point_size", default=1.0, type=float)
    p.add_argument("--depth_vmax", default=80.0, type=float)
    p.add_argument("--error_vmax", default=5.0, type=float)
    p.add_argument("--save_npz", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_ply", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--filter_scene_bounds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep projected points inside the selected dataset voxel bounds.",
    )
    p.add_argument("--proj_elev", default=22.0, type=float)
    p.add_argument("--proj_azim", default=-55.0, type=float)
    p.add_argument("--scene_xlim", nargs=2, type=float, default=None)
    p.add_argument("--scene_ylim", nargs=2, type=float, default=None)
    p.add_argument("--scene_zlim", nargs=2, type=float, default=None)

    # Optional path/config overrides for checkpoints that do not contain args.
    p.add_argument("--processed_root", default=None, type=str)
    p.add_argument("--nuscenes_processed_root", default=None, type=str)
    p.add_argument("--velodyne_root", default=None, type=str)
    p.add_argument("--occany_ckpt", default=None, type=str)
    p.add_argument("--width", default=None, type=int)
    p.add_argument("--height", default=None, type=int)
    p.add_argument("--num_frames", default=None, type=int)
    p.add_argument("--frame_stride", default=None, type=int)
    p.add_argument("--nuscenes_frame_stride", default=None, type=int)
    p.add_argument("--c_lift", default=None, type=int)
    p.add_argument("--token_dim", default=None, type=int)
    p.add_argument("--patch_size", default=None, type=int)
    p.add_argument("--max_points_per_sweep", default=None, type=int)
    p.add_argument("--dense_depth_features", default=None, type=int)
    p.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--num_workers", default=0, type=int)
    return p.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_ckpt_ref(ref: str, ckpt_root: Path, checkpoint_name: str) -> Path:
    path = Path(ref).expanduser()
    if path.exists():
        return _resolve_ckpt_path(str(path))

    root_candidate = ckpt_root / ref
    if root_candidate.exists():
        return _resolve_ckpt_path(str(root_candidate))

    if ref.isdigit():
        prefix = f"{int(ref):02d}_"
        matches = sorted(p for p in ckpt_root.glob(f"{prefix}*") if p.is_dir())
        if len(matches) == 1:
            return _resolve_ckpt_path(str(matches[0] / checkpoint_name))
        if len(matches) > 1:
            joined = ", ".join(str(p) for p in matches)
            raise RuntimeError(f"Ambiguous checkpoint alias {ref!r}: {joined}")

    raise FileNotFoundError(
        f"Could not resolve checkpoint {ref!r}. Tried as path, under {ckpt_root}, "
        "and as a numeric experiment alias."
    )


def ckpt_label(path: Path) -> str:
    return path.parent.name if path.name.startswith("checkpoint") else path.stem


def override_or_ckpt(args: argparse.Namespace, ckpt_args, name: str, default):
    value = getattr(args, name)
    return value if value is not None else _ckpt_arg(ckpt_args, name, default)


def build_run_args(base_args: argparse.Namespace, ckpt_args) -> argparse.Namespace:
    args = copy(base_args)
    args.exp = _ckpt_arg(ckpt_args, "exp", "bevdetocc_lidar_pointmap")
    if args.exp not in POINTMAP_EVAL_EXPS:
        raise ValueError(
            f"Expected a pointmap checkpoint trained with one of {POINTMAP_EVAL_EXPS}; "
            f"checkpoint exp={args.exp!r}."
        )

    args.processed_root = override_or_ckpt(args, ckpt_args, "processed_root", None)
    args.nuscenes_processed_root = override_or_ckpt(
        args, ckpt_args, "nuscenes_processed_root", None
    )
    args.velodyne_root = override_or_ckpt(args, ckpt_args, "velodyne_root", None)
    args.occany_ckpt = override_or_ckpt(args, ckpt_args, "occany_ckpt", None)
    args.width = int(override_or_ckpt(args, ckpt_args, "width", 512))
    args.height = int(override_or_ckpt(args, ckpt_args, "height", 160))
    args.num_frames = int(override_or_ckpt(args, ckpt_args, "num_frames", 5))
    args.frame_stride = int(override_or_ckpt(args, ckpt_args, "frame_stride", 4))
    args.nuscenes_frame_stride = int(
        override_or_ckpt(args, ckpt_args, "nuscenes_frame_stride", 1)
    )
    args.c_lift = int(override_or_ckpt(args, ckpt_args, "c_lift", 64))
    args.token_dim = int(override_or_ckpt(args, ckpt_args, "token_dim", 768))
    args.patch_size = int(override_or_ckpt(args, ckpt_args, "patch_size", 16))
    args.max_points_per_sweep = int(
        override_or_ckpt(args, ckpt_args, "max_points_per_sweep", 0)
    )
    args.dense_depth_features = int(
        override_or_ckpt(args, ckpt_args, "dense_depth_features", 128)
    )
    args.freeze_backbone = bool(
        override_or_ckpt(args, ckpt_args, "freeze_backbone", False)
    )
    args.amp = args.amp or _ckpt_arg(ckpt_args, "amp", "bf16")
    args.batch_size = 1
    args.num_workers = 0
    args.distributed = False
    args.multi_dataset = False
    args.depth_supervision = False
    args.dense_depth_supervision = False
    args.pointmap_supervision = False

    if not args.processed_root:
        raise ValueError("--processed_root is required when checkpoint args do not contain it.")
    if args.dataset == "nuscenes" and not args.nuscenes_processed_root:
        raise ValueError(
            "--nuscenes_processed_root is required for --dataset nuscenes when "
            "checkpoint args do not contain it."
        )
    if args.exp in POINTMAP_LIDAR_EXPS and args.dataset == "kitti":
        # Current KITTI datasets read LiDAR from processed_root/<split>_<seq>/lidar.
        # velodyne_root is kept only for old checkpoint compatibility.
        args.velodyne_root = args.velodyne_root
    if not args.occany_ckpt:
        raise ValueError("--occany_ckpt is required when checkpoint args do not contain it.")
    return args


def apply_default_scene_bounds(args: argparse.Namespace) -> None:
    grid = GRID_CONFIGS[args.dataset]
    origin = np.asarray(grid.full_voxel_origin, dtype=np.float32)
    size = np.asarray(grid.full_voxel_size, dtype=np.float32)
    dims = np.asarray(grid.full_grid_size, dtype=np.float32)
    upper = origin + size * dims
    if args.scene_xlim is None:
        args.scene_xlim = (float(origin[0]), float(upper[0]))
    if args.scene_ylim is None:
        args.scene_ylim = (float(origin[1]), float(upper[1]))
    if args.scene_zlim is None:
        args.scene_zlim = (float(origin[2]), float(upper[2]))


def build_dataset(args: argparse.Namespace):
    if args.dataset == "kitti":
        return _build_kitti_dataset(args, "val")
    return _build_nuscenes_dataset(args, "val")


def collate_one(args: argparse.Namespace, sample: Dict) -> Dict:
    if args.dataset == "nuscenes":
        return collate_stage1_nuscenes_lidar([sample])
    if args.exp in POINTMAP_DENSE_DEPTH_DATA_EXPS:
        return collate_stage1_dense_depth([sample])
    return collate_stage1_lidar_dense_depth([sample])


def select_sample_idx(dataset_len: int, sample_idx: int, seed: int) -> int:
    if dataset_len <= 0:
        raise RuntimeError("Validation dataset is empty.")
    if sample_idx >= 0:
        if sample_idx >= dataset_len:
            raise IndexError(
                f"--sample_idx={sample_idx} out of range for val dataset of size {dataset_len}."
            )
        return int(sample_idx)
    rng = random.Random(int(seed))
    return int(rng.randrange(dataset_len))


def safe_scalar(value) -> str | int:
    if isinstance(value, (list, tuple)):
        return value[0] if len(value) == 1 else str(value)
    return value


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def run_one_checkpoint(
    *,
    base_args: argparse.Namespace,
    ckpt_path: Path,
    label: str,
    sample: Dict,
    batch: Dict,
    out_dir: Path,
    device: torch.device,
) -> Dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = build_run_args(base_args, ckpt.get("args", {}))
    apply_default_scene_bounds(args)

    model = _build_pointmap_model(args, device)
    missing, unexpected = _load_pointmap_weights(model, ckpt)
    model.eval()

    with torch.no_grad(), _amp_context(device, args.amp):
        out = _model_forward(model, batch, device, args)

    pred_ref = out["pointmap_pts3d"].float()
    pred_local = out["pointmap_pts3d_local"].float()
    pred_conf = out.get("pointmap_conf")
    if pred_conf is not None:
        pred_conf = pred_conf.float()

    dense_depth = batch["dense_depth"].to(device=device, dtype=torch.float32)
    if "K_per_frame" in batch:
        K_per_frame = batch["K_per_frame"].to(device=device, dtype=torch.float32)
    else:
        K_per_frame = torch.stack(
            [
                view["camera_intrinsics"].to(device=device, dtype=torch.float32)
                for view in batch["views"]
            ],
            dim=1,
        )
    frame_mask = batch.get("dense_depth_frame_mask")
    if frame_mask is not None:
        frame_mask = frame_mask.to(device=device)
    cam2world = _stack_cam2world_from_views(batch, device=device)
    gt_ref, gt_local, valid = valid_pointmap_mask(
        pred_ref,
        pred_local,
        dense_depth,
        K_per_frame,
        cam2world,
        frame_mask,
    )

    imgs = torch.cat([view["img"] for view in batch["views"]], dim=0)
    rgbs = convert_images_to_uint8_hwc(imgs)

    ckpt_out_dir = out_dir / label
    ckpt_out_dir.mkdir(parents=True, exist_ok=True)

    pred_ref_np = to_numpy(pred_ref[0])
    pred_local_np = to_numpy(pred_local[0])
    gt_ref_np = to_numpy(gt_ref[0])
    gt_local_np = to_numpy(gt_local[0])
    valid_np = to_numpy(valid[0]).astype(bool)
    depth_np = to_numpy(dense_depth[0])
    conf_np = to_numpy(pred_conf[0]) if pred_conf is not None else None
    T_target_from_refcam_np = tensor_to_numpy(batch["T_target_from_refcam"][0].float())

    frame_ids = list(sample.get("frame_ids", range(pred_ref_np.shape[0])))
    rows: List[Tuple[int, Dict[str, float]]] = []
    all_pred_points: List[np.ndarray] = []
    all_pred_colors: List[np.ndarray] = []
    all_gt_points: List[np.ndarray] = []
    all_gt_colors: List[np.ndarray] = []
    for frame_idx in selected_frames(pred_ref_np.shape[0], int(args.frame_idx)):
        stats = plot_frame(
            out_path=ckpt_out_dir / f"frame_{frame_idx}_pointmap.png",
            rgb=rgbs[frame_idx],
            pred_ref=pred_ref_np[frame_idx],
            pred_local=pred_local_np[frame_idx],
            gt_ref=gt_ref_np[frame_idx],
            dense_depth=depth_np[frame_idx],
            valid=valid_np[frame_idx],
            conf=conf_np[frame_idx] if conf_np is not None else None,
            frame_idx=frame_idx,
            frame_id=int(frame_ids[frame_idx]),
            args=args,
        )
        rows.append((frame_idx, stats))
        projection = collect_pixel_projection_points(
            pred_ref=pred_ref_np[frame_idx],
            gt_ref=gt_ref_np[frame_idx],
            rgb=rgbs[frame_idx],
            valid_gt=valid_np[frame_idx],
            T_target_from_refcam=T_target_from_refcam_np,
            args=args,
        )
        all_pred_points.append(projection["pred_points"])
        all_pred_colors.append(projection["pred_colors"])
        all_gt_points.append(projection["gt_points"])
        all_gt_colors.append(projection["gt_colors"])
        save_pixel_projection_3d_plot(
            out_path=ckpt_out_dir / f"frame_{frame_idx}_pixel_projection_3d.png",
            rgb=rgbs[frame_idx],
            projection=projection,
            title=(
                f"{label}: 2D pixels projected to {args.dataset} target space "
                f"(frame_idx={frame_idx}, frame_id={int(frame_ids[frame_idx])})"
            ),
            args=args,
            seed=int(args.seed) + int(frame_idx),
        )
        if args.save_ply:
            write_rgb_ply(
                ckpt_out_dir / f"frame_{frame_idx}_pred_pixels_3d.ply",
                projection["pred_points"],
                projection["pred_colors"],
            )
            write_rgb_ply(
                ckpt_out_dir / f"frame_{frame_idx}_gt_pixels_3d.ply",
                projection["gt_points"],
                projection["gt_colors"],
            )

    write_metrics(ckpt_out_dir / "metrics.csv", rows)
    if all_pred_points:
        merged_projection = {
            "pred_points": np.concatenate(all_pred_points, axis=0),
            "pred_colors": np.concatenate(all_pred_colors, axis=0),
            "gt_points": np.concatenate(all_gt_points, axis=0),
            "gt_colors": np.concatenate(all_gt_colors, axis=0),
        }
        save_pixel_projection_3d_plot(
            out_path=ckpt_out_dir / "all_selected_frames_pixel_projection_3d.png",
            rgb=None,
            projection=merged_projection,
            title=f"{label}: all selected pixels projected to {args.dataset} target space",
            args=args,
            seed=int(args.seed) + 1009,
        )
        if args.save_ply:
            write_rgb_ply(
                ckpt_out_dir / "all_selected_frames_pred_pixels_3d.ply",
                merged_projection["pred_points"],
                merged_projection["pred_colors"],
            )
            write_rgb_ply(
                ckpt_out_dir / "all_selected_frames_gt_pixels_3d.ply",
                merged_projection["gt_points"],
                merged_projection["gt_colors"],
            )

    if args.save_npz:
        payload = dict(
            pred_ref=pred_ref_np,
            pred_local=pred_local_np,
            gt_ref=gt_ref_np,
            gt_local=gt_local_np,
            valid=valid_np,
            dense_depth=depth_np,
            T_target_from_refcam=T_target_from_refcam_np,
            frame_ids=np.asarray(frame_ids, dtype=np.int64),
            sequence=np.asarray([safe_scalar(sample.get("sequence", ""))]),
            target_frame_id=np.asarray(
                [safe_scalar(sample.get("target_frame_id", -1))], dtype=np.int64
            ),
            dataset=np.asarray([args.dataset]),
            ckpt=np.asarray([str(ckpt_path)]),
        )
        if conf_np is not None:
            payload["conf"] = conf_np
        np.savez_compressed(ckpt_out_dir / "pointmap_arrays.npz", **payload)

    return {
        "label": label,
        "ckpt": str(ckpt_path),
        "exp": args.exp,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "output_dir": str(ckpt_out_dir),
        "metrics": {str(frame_idx): stats for frame_idx, stats in rows},
    }


def write_summary(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = get_args()
    if args.labels is not None and len(args.labels) != len(args.ckpts):
        raise ValueError("--labels must have the same length as --ckpts.")

    toggle_memory_efficient_attention(enabled=False)
    register_legacy_checkpoint_modules()

    ckpt_root = Path(args.ckpt_root)
    ckpt_paths = [
        resolve_ckpt_ref(ref, ckpt_root, args.checkpoint_name)
        for ref in args.ckpts
    ]
    labels = args.labels or [ckpt_label(path) for path in ckpt_paths]

    first_ckpt = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)
    dataset_args = build_run_args(args, first_ckpt.get("args", {}))
    apply_default_scene_bounds(dataset_args)
    dataset = build_dataset(dataset_args)
    sample_idx = select_sample_idx(len(dataset), int(args.sample_idx), int(args.seed))
    sample = dataset[sample_idx]
    batch = collate_one(dataset_args, sample)

    sequence = safe_scalar(sample.get("sequence", ""))
    target_frame_id = safe_scalar(sample.get("target_frame_id", -1))
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / args.dataset / f"sample_{sample_idx:06d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    print(
        f"[pointmap-vis] dataset={args.dataset} sample_idx={sample_idx} "
        f"sequence={sequence} target_frame_id={target_frame_id}"
    )
    print(f"[pointmap-vis] output_dir={out_dir}")

    runs = []
    for ckpt_path, label in zip(ckpt_paths, labels):
        print(f"[pointmap-vis] running label={label} ckpt={ckpt_path}")
        runs.append(
            run_one_checkpoint(
                base_args=args,
                ckpt_path=ckpt_path,
                label=label,
                sample=sample,
                batch=batch,
                out_dir=out_dir,
                device=device,
            )
        )

    summary = {
        "dataset": args.dataset,
        "sample_idx": sample_idx,
        "sequence": str(sequence),
        "target_frame_id": int(target_frame_id),
        "frame_ids": [int(v) for v in sample.get("frame_ids", [])],
        "frame_idx": int(args.frame_idx),
        "runs": runs,
    }
    write_summary(out_dir / "summary.json", summary)
    print(f"[pointmap-vis] wrote {out_dir}")


if __name__ == "__main__":
    main()
