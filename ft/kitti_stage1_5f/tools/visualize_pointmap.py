"""Visualize Stage-1 pointmap predictions for one validation sample.

Example:
  python -m ft.kitti_stage1_5f.tools.visualize_pointmap \
    --ckpt output/kitti_stage1_5f_4gpu_pointmap_postfusion_only/checkpoint-last.pth \
    --sample_idx 12

The checkpoint's saved args are used to recover the model type, data roots,
OccAny checkpoint, image size, and frame settings. Override those paths only
when they are absent from the checkpoint.
"""
from __future__ import annotations

try:
    from .. import _paths  # noqa: F401  (must run before project imports)
except ImportError:  # Allows direct `python ft/.../visualize_pointmap.py`.
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from ft.kitti_stage1_5f import _paths  # noqa: F401

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from occany.model.must3r_blocks.attention import toggle_memory_efficient_attention
from occany.utils.checkpoint_io import register_legacy_checkpoint_modules
from occany.utils.image_util import convert_images_to_uint8_hwc

from ft.kitti_stage1_5f.pointmap_metrics import valid_pointmap_mask
from ft.kitti_stage1_5f.tools.eval_pointmap_quality import (
    _amp_context,
    _build_pointmap_model,
    _fill_args_from_checkpoint,
    _load_pointmap_weights,
    _resolve_ckpt_path,
)
from ft.kitti_stage1_5f.tools.train import (
    _build_dataset,
    _collate_fn,
    _model_forward,
    _stack_cam2world_from_views,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "visuals" / "kitti_stage1_pointmap"


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Visualize Stage-1 pointmap prediction")
    p.add_argument("--ckpt", required=True, type=str, help="Checkpoint file or directory.")
    p.add_argument("--sample_idx", required=True, type=int, help="Validation dataset index.")
    p.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Output directory. Defaults to visuals/kitti_stage1_pointmap/<run>/sample_<idx>.",
    )
    p.add_argument(
        "--frame_idx",
        default=-1,
        type=int,
        help="Frame to visualize. Use -1 to render all frames.",
    )
    p.add_argument("--device", default="auto", type=str, help="auto, cuda, cuda:0, or cpu.")
    p.add_argument("--amp", choices=["bf16", "fp16", "none"], default=None)
    p.add_argument("--seed", default=0, type=int)
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
        help="Keep projected points inside the KITTI target voxel bounds.",
    )
    p.add_argument("--proj_elev", default=22.0, type=float)
    p.add_argument("--proj_azim", default=-55.0, type=float)
    p.add_argument("--scene_xlim", nargs=2, type=float, default=(0.0, 51.2))
    p.add_argument("--scene_ylim", nargs=2, type=float, default=(-25.6, 25.6))
    p.add_argument("--scene_zlim", nargs=2, type=float, default=(-2.0, 4.4))

    # Optional path/config overrides for checkpoints that do not contain args.
    p.add_argument("--processed_root", default=None, type=str)
    p.add_argument("--velodyne_root", default=None, type=str)
    p.add_argument("--occany_ckpt", default=None, type=str)
    p.add_argument("--width", default=None, type=int)
    p.add_argument("--height", default=None, type=int)
    p.add_argument("--num_frames", default=None, type=int)
    p.add_argument("--frame_stride", default=None, type=int)
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


def default_output_dir(ckpt_path: Path, sample_idx: int) -> Path:
    run_name = ckpt_path.parent.name if ckpt_path.name.startswith("checkpoint") else ckpt_path.stem
    return DEFAULT_OUTPUT_ROOT / run_name / f"sample_{sample_idx:06d}"


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def masked_image(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    out[~valid] = np.nan
    return out


def robust_limits(points: np.ndarray, dims: Tuple[int, int]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if points.size == 0:
        return (-1.0, 1.0), (-1.0, 1.0)
    a = points[:, dims[0]]
    b = points[:, dims[1]]
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        return (-1.0, 1.0), (-1.0, 1.0)
    a = a[finite]
    b = b[finite]
    lo_a, hi_a = np.percentile(a, [1, 99])
    lo_b, hi_b = np.percentile(b, [1, 99])
    pad_a = max((hi_a - lo_a) * 0.08, 1e-3)
    pad_b = max((hi_b - lo_b) * 0.08, 1e-3)
    return (float(lo_a - pad_a), float(hi_a + pad_a)), (float(lo_b - pad_b), float(hi_b + pad_b))


def sample_points(
    pred_ref: np.ndarray,
    gt_ref: np.ndarray,
    valid: np.ndarray,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    pred = pred_ref[valid]
    gt = gt_ref[valid]
    finite = np.isfinite(pred).all(axis=1) & np.isfinite(gt).all(axis=1)
    pred = pred[finite]
    gt = gt[finite]
    if max_points > 0 and pred.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(pred.shape[0], size=max_points, replace=False)
        pred = pred[idx]
        gt = gt[idx]
    return pred, gt


def transform_ref_to_target(points_ref: np.ndarray, T_target_from_refcam: np.ndarray) -> np.ndarray:
    flat = points_ref.reshape(-1, 3).astype(np.float32)
    T = T_target_from_refcam.astype(np.float32)
    out = flat @ T[:3, :3].T + T[:3, 3]
    return out.reshape(points_ref.shape)


def scene_bounds_mask(points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    finite = np.isfinite(points).all(axis=-1)
    if not bool(args.filter_scene_bounds):
        return finite
    return (
        finite
        & (points[..., 0] >= float(args.scene_xlim[0]))
        & (points[..., 0] <= float(args.scene_xlim[1]))
        & (points[..., 1] >= float(args.scene_ylim[0]))
        & (points[..., 1] <= float(args.scene_ylim[1]))
        & (points[..., 2] >= float(args.scene_zlim[0]))
        & (points[..., 2] <= float(args.scene_zlim[1]))
    )


def collect_pixel_projection_points(
    *,
    pred_ref: np.ndarray,
    gt_ref: np.ndarray,
    rgb: np.ndarray,
    valid_gt: np.ndarray,
    T_target_from_refcam: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    pred_target = transform_ref_to_target(pred_ref, T_target_from_refcam)
    gt_target = transform_ref_to_target(gt_ref, T_target_from_refcam)

    pred_mask = scene_bounds_mask(pred_target, args)
    gt_mask = valid_gt & scene_bounds_mask(gt_target, args)

    rgb_flat = rgb.reshape(-1, 3).astype(np.uint8)
    pred_flat = pred_target.reshape(-1, 3)
    gt_flat = gt_target.reshape(-1, 3)
    pred_keep = pred_mask.reshape(-1)
    gt_keep = gt_mask.reshape(-1)
    return {
        "pred_points": pred_flat[pred_keep],
        "pred_colors": rgb_flat[pred_keep],
        "gt_points": gt_flat[gt_keep],
        "gt_colors": rgb_flat[gt_keep],
    }


def subsample_for_plot(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx], colors[idx]


def write_rgb_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    finite = np.isfinite(points).all(axis=1)
    points = points[finite].astype(np.float32)
    colors = colors[finite].astype(np.uint8)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("w") as f:
        f.write(header)
        if points.shape[0] == 0:
            return
        data = np.column_stack([points, colors])
        np.savetxt(f, data, fmt="%.6f %.6f %.6f %d %d %d")


def setup_3d_axis(ax, title: str, args: argparse.Namespace) -> None:
    ax.set_title(title)
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.set_zlabel("z up (m)")
    ax.set_xlim(float(args.scene_xlim[0]), float(args.scene_xlim[1]))
    ax.set_ylim(float(args.scene_ylim[0]), float(args.scene_ylim[1]))
    ax.set_zlim(float(args.scene_zlim[0]), float(args.scene_zlim[1]))
    ax.view_init(elev=float(args.proj_elev), azim=float(args.proj_azim))
    ax.grid(True, linewidth=0.3, alpha=0.35)


def scatter_rgb_points(ax, points: np.ndarray, colors: np.ndarray, args: argparse.Namespace) -> None:
    if points.shape[0] == 0:
        ax.text2D(0.5, 0.5, "No points in scene bounds", transform=ax.transAxes, ha="center")
        return
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors.astype(np.float32) / 255.0,
        s=float(args.point_size),
        alpha=0.75,
        linewidths=0.0,
        depthshade=False,
    )


def save_pixel_projection_3d_plot(
    *,
    out_path: Path,
    rgb: Optional[np.ndarray],
    projection: Dict[str, np.ndarray],
    title: str,
    args: argparse.Namespace,
    seed: int,
) -> None:
    pred_pts, pred_rgb = subsample_for_plot(
        projection["pred_points"],
        projection["pred_colors"],
        int(args.max_points_plot),
        seed,
    )
    gt_pts, gt_rgb = subsample_for_plot(
        projection["gt_points"],
        projection["gt_colors"],
        int(args.max_points_plot),
        seed + 7919,
    )

    ncols = 3 if rgb is not None else 2
    fig = plt.figure(figsize=(7.0 * ncols, 6.5), dpi=160)
    fig.suptitle(title, fontsize=13)
    col = 1
    if rgb is not None:
        ax_img = fig.add_subplot(1, ncols, 1)
        ax_img.imshow(rgb)
        ax_img.set_title("2D image pixels")
        ax_img.axis("off")
        col = 2

    ax_pred = fig.add_subplot(1, ncols, col, projection="3d")
    scatter_rgb_points(ax_pred, pred_pts, pred_rgb, args)
    setup_3d_axis(
        ax_pred,
        f"Pred pointmap pixels\nshown={pred_pts.shape[0]} total={projection['pred_points'].shape[0]}",
        args,
    )

    ax_gt = fig.add_subplot(1, ncols, col + 1, projection="3d")
    scatter_rgb_points(ax_gt, gt_pts, gt_rgb, args)
    setup_3d_axis(
        ax_gt,
        f"GT depth pixels\nshown={gt_pts.shape[0]} total={projection['gt_points'].shape[0]}",
        args,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def add_imshow(
    ax,
    image: np.ndarray,
    title: str,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    if cmap is not None:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def set_scatter_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.3, alpha=0.35)


def plot_frame(
    *,
    out_path: Path,
    rgb: np.ndarray,
    pred_ref: np.ndarray,
    pred_local: np.ndarray,
    gt_ref: np.ndarray,
    dense_depth: np.ndarray,
    valid: np.ndarray,
    conf: Optional[np.ndarray],
    frame_idx: int,
    frame_id: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    pred_depth = pred_local[..., 2]
    gt_depth = dense_depth
    depth_err = np.abs(pred_depth - gt_depth)
    l2_err = np.linalg.norm(pred_ref - gt_ref, axis=-1)

    metric_mask = valid & np.isfinite(l2_err) & np.isfinite(pred_depth) & np.isfinite(gt_depth)
    if np.any(metric_mask):
        pts_l2 = float(np.nanmean(l2_err[metric_mask]))
        pts_l2_med = float(np.nanmedian(l2_err[metric_mask]))
        depth_absrel = float(
            np.nanmean(depth_err[metric_mask] / np.clip(gt_depth[metric_mask], 1e-6, None))
        )
    else:
        pts_l2 = float("nan")
        pts_l2_med = float("nan")
        depth_absrel = float("nan")

    pred_pts, gt_pts = sample_points(
        pred_ref,
        gt_ref,
        metric_mask,
        int(args.max_points_plot),
        int(args.seed) + int(frame_idx),
    )
    all_pts = np.concatenate([pred_pts, gt_pts], axis=0) if pred_pts.size else gt_pts
    xlim_xz, ylim_xz = robust_limits(all_pts, (0, 2))
    xlim_zy, ylim_zy = robust_limits(all_pts, (2, 1))

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    title = (
        f"frame_idx={frame_idx} frame_id={frame_id} "
        f"L2={pts_l2:.3f} median={pts_l2_med:.3f} AbsRel={depth_absrel:.4f}"
    )
    fig.suptitle(title, fontsize=13)

    add_imshow(axes[0, 0], rgb, "RGB")
    add_imshow(
        axes[0, 1],
        masked_image(gt_depth, valid),
        "GT depth",
        cmap="magma",
        vmin=0.0,
        vmax=float(args.depth_vmax),
    )
    add_imshow(
        axes[0, 2],
        masked_image(pred_depth, valid),
        "Pred depth",
        cmap="magma",
        vmin=0.0,
        vmax=float(args.depth_vmax),
    )
    add_imshow(
        axes[0, 3],
        masked_image(depth_err, valid),
        "Abs depth error",
        cmap="viridis",
        vmin=0.0,
        vmax=float(args.error_vmax),
    )

    add_imshow(
        axes[1, 0],
        masked_image(l2_err, valid),
        "3D L2 error",
        cmap="viridis",
        vmin=0.0,
        vmax=float(args.error_vmax),
    )
    if conf is None:
        axes[1, 1].text(0.5, 0.5, "No confidence", ha="center", va="center")
        axes[1, 1].set_title("Confidence")
        axes[1, 1].axis("off")
    else:
        add_imshow(
            axes[1, 1],
            masked_image(np.log(np.clip(conf, 1e-6, None)), valid),
            "log confidence",
            cmap="plasma",
        )

    axes[1, 2].scatter(gt_pts[:, 0], gt_pts[:, 2], s=args.point_size, c="0.65", alpha=0.45, label="GT")
    axes[1, 2].scatter(
        pred_pts[:, 0],
        pred_pts[:, 2],
        s=args.point_size,
        c="#d62728",
        alpha=0.45,
        label="Pred",
    )
    axes[1, 2].set_xlim(*xlim_xz)
    axes[1, 2].set_ylim(*ylim_xz)
    axes[1, 2].legend(loc="upper right", markerscale=4)
    set_scatter_axes(axes[1, 2], "Reference cam X-Z", "x (m)", "z (m)")

    axes[1, 3].scatter(gt_pts[:, 2], gt_pts[:, 1], s=args.point_size, c="0.65", alpha=0.45, label="GT")
    axes[1, 3].scatter(
        pred_pts[:, 2],
        pred_pts[:, 1],
        s=args.point_size,
        c="#1f77b4",
        alpha=0.45,
        label="Pred",
    )
    axes[1, 3].set_xlim(*xlim_zy)
    axes[1, 3].set_ylim(*ylim_zy)
    axes[1, 3].invert_yaxis()
    axes[1, 3].legend(loc="upper right", markerscale=4)
    set_scatter_axes(axes[1, 3], "Reference cam Z-Y", "z (m)", "y (m)")

    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return {
        "pts3d_l2_mean": pts_l2,
        "pts3d_l2_median": pts_l2_med,
        "depth_absrel": depth_absrel,
        "valid_pixels": float(np.count_nonzero(metric_mask)),
    }


def selected_frames(n_frames: int, frame_idx: int) -> List[int]:
    if frame_idx < 0:
        return list(range(n_frames))
    if frame_idx >= n_frames:
        raise IndexError(f"--frame_idx={frame_idx} out of range for {n_frames} frames.")
    return [int(frame_idx)]


def write_metrics(path: Path, rows: Iterable[Tuple[int, Dict[str, float]]]) -> None:
    lines = ["frame_idx,pts3d_l2_mean,pts3d_l2_median,depth_absrel,valid_pixels"]
    for frame_idx, stats in rows:
        lines.append(
            f"{frame_idx},"
            f"{stats['pts3d_l2_mean']:.6f},"
            f"{stats['pts3d_l2_median']:.6f},"
            f"{stats['depth_absrel']:.6f},"
            f"{stats['valid_pixels']:.0f}"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = get_args()
    toggle_memory_efficient_attention(enabled=False)
    register_legacy_checkpoint_modules()

    ckpt_path = _resolve_ckpt_path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _fill_args_from_checkpoint(args, ckpt.get("args", {}))
    args.distributed = False
    args.batch_size = 1
    args.num_workers = 0

    device = resolve_device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    model = _build_pointmap_model(args, device)
    missing, unexpected = _load_pointmap_weights(model, ckpt)
    model.eval()

    dataset = _build_dataset(args, "val")
    if args.sample_idx < 0 or args.sample_idx >= len(dataset):
        raise IndexError(
            f"--sample_idx={args.sample_idx} out of range for val dataset of size {len(dataset)}."
        )
    sample = dataset[int(args.sample_idx)]
    batch = _collate_fn(args)([sample])

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
                v["camera_intrinsics"].to(device=device, dtype=torch.float32)
                for v in batch["views"]
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

    imgs = torch.cat([v["img"] for v in batch["views"]], dim=0)
    rgbs = convert_images_to_uint8_hwc(imgs)

    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir(ckpt_path, args.sample_idx)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_ref_np = to_numpy(pred_ref[0])
    pred_local_np = to_numpy(pred_local[0])
    gt_ref_np = to_numpy(gt_ref[0])
    gt_local_np = to_numpy(gt_local[0])
    valid_np = to_numpy(valid[0]).astype(bool)
    depth_np = to_numpy(dense_depth[0])
    conf_np = to_numpy(pred_conf[0]) if pred_conf is not None else None
    T_target_from_refcam_np = to_numpy(batch["T_target_from_refcam"][0].float())

    frame_ids = list(sample.get("frame_ids", range(pred_ref_np.shape[0])))
    rows: List[Tuple[int, Dict[str, float]]] = []
    all_pred_points: List[np.ndarray] = []
    all_pred_colors: List[np.ndarray] = []
    all_gt_points: List[np.ndarray] = []
    all_gt_colors: List[np.ndarray] = []
    for frame_idx in selected_frames(pred_ref_np.shape[0], int(args.frame_idx)):
        stats = plot_frame(
            out_path=out_dir / f"frame_{frame_idx}_pointmap.png",
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
            out_path=out_dir / f"frame_{frame_idx}_pixel_projection_3d.png",
            rgb=rgbs[frame_idx],
            projection=projection,
            title=(
                f"2D pixels projected to 3D target space "
                f"(frame_idx={frame_idx}, frame_id={int(frame_ids[frame_idx])})"
            ),
            args=args,
            seed=int(args.seed) + int(frame_idx),
        )
        if args.save_ply:
            write_rgb_ply(
                out_dir / f"frame_{frame_idx}_pred_pixels_3d.ply",
                projection["pred_points"],
                projection["pred_colors"],
            )
            write_rgb_ply(
                out_dir / f"frame_{frame_idx}_gt_pixels_3d.ply",
                projection["gt_points"],
                projection["gt_colors"],
            )

    write_metrics(out_dir / "metrics.csv", rows)
    if all_pred_points:
        merged_projection = {
            "pred_points": np.concatenate(all_pred_points, axis=0),
            "pred_colors": np.concatenate(all_pred_colors, axis=0),
            "gt_points": np.concatenate(all_gt_points, axis=0),
            "gt_colors": np.concatenate(all_gt_colors, axis=0),
        }
        save_pixel_projection_3d_plot(
            out_path=out_dir / "all_frames_pixel_projection_3d.png",
            rgb=None,
            projection=merged_projection,
            title="All selected 2D pixels projected to 3D target space",
            args=args,
            seed=int(args.seed) + 1009,
        )
        if args.save_ply:
            write_rgb_ply(
                out_dir / "all_frames_pred_pixels_3d.ply",
                merged_projection["pred_points"],
                merged_projection["pred_colors"],
            )
            write_rgb_ply(
                out_dir / "all_frames_gt_pixels_3d.ply",
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
            sequence=np.asarray([sample.get("sequence", "")]),
            target_frame_id=np.asarray([sample.get("target_frame_id", -1)], dtype=np.int64),
        )
        if conf_np is not None:
            payload["conf"] = conf_np
        np.savez_compressed(out_dir / "pointmap_arrays.npz", **payload)

    print(f"[pointmap-vis] ckpt={ckpt_path}")
    print(f"[pointmap-vis] exp={args.exp} sample_idx={args.sample_idx} sequence={sample.get('sequence')}")
    print(f"[pointmap-vis] load_state missing={len(missing)} unexpected={len(unexpected)}")
    print(f"[pointmap-vis] wrote {out_dir}")


if __name__ == "__main__":
    main()
