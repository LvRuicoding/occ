"""Visualize OccAny patch-token projections into SemanticKITTI voxels.

Example:
  python -m ft.kitti_stage1_5f.tools.visualize_patch_projection --num_samples 3

The script samples train-set 5-frame examples, runs the frozen pretrained
OccAny reconstruction backbone, maps each decoder patch token to one 3D point
by confidence-weighted averaging of its patch's pointmap pixels, and counts how
many patch tokens land in each SemanticKITTI voxel. It does not load or depend
on any Stage-1 SSC fine-tuned checkpoint.
"""
from __future__ import annotations

from .. import _paths  # noqa: F401

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from occany.utils.image_util import convert_images_to_uint8_hwc

from ..datasets import Kitti5FrameStage1Dataset, collate_stage1
from ..models import OccAnyRecon5FrameBackbone


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OCCANY_CKPTS = (
    REPO_ROOT / "checkpoints" / "occany_recon.pth",
    REPO_ROOT / "checkpoints" / "occany.pth",
)
DEFAULT_PROCESSED_ROOT = REPO_ROOT / "data" / "kitti_processed"
DEFAULT_KITTIODO_ROOT = REPO_ROOT / "raw_data" / "OpenDataLab___KITTI_Odometry_2012"


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Visualize 5-frame OccAny patch projections into SemanticKITTI voxels"
    )
    p.add_argument("--processed_root", default=str(DEFAULT_PROCESSED_ROOT), type=str)
    p.add_argument("--kittiodo_root", default=str(DEFAULT_KITTIODO_ROOT), type=str)
    p.add_argument(
        "--occany_ckpt",
        default=None,
        type=str,
        help=(
            "Pretrained OccAny reconstruction checkpoint. Defaults to "
            "checkpoints/occany_recon.pth, then checkpoints/occany.pth."
        ),
    )
    p.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "visuals" / "kitti_stage1_patch_pixel_projection"),
        type=str,
    )
    p.add_argument("--split", default="train", choices=["train", "val", "trainval"])
    p.add_argument("--num_samples", default=3, type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--width", default=512, type=int)
    p.add_argument("--height", default=160, type=int)
    p.add_argument("--num_frames", default=5, type=int)
    p.add_argument("--frame_stride", default=1, type=int)
    p.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    p.add_argument(
        "--max_text_labels",
        default=250,
        type=int,
        help="Maximum number of numeric labels drawn in the 3D plot.",
    )
    p.add_argument(
        "--max_points",
        default=5000,
        type=int,
        help="Maximum voxel points drawn; all exact counts are still saved to npz.",
    )
    p.add_argument(
        "--max_pixel_points",
        default=50000,
        type=int,
        help="Maximum projected image pixels drawn in the 3D RGB point plot.",
    )
    p.add_argument("--elev", default=22.0, type=float)
    p.add_argument("--azim", default=-55.0, type=float)
    return p.parse_args()


def resolve_occany_ckpt(path_arg: str | None) -> Path:
    if path_arg:
        path = Path(path_arg).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"--occany_ckpt does not exist: {path}")
        return path
    for path in DEFAULT_OCCANY_CKPTS:
        if path.is_file():
            return path
    tried = ", ".join(str(p) for p in DEFAULT_OCCANY_CKPTS)
    raise FileNotFoundError(
        "Could not find a pretrained OccAny checkpoint automatically. "
        f"Tried: {tried}. Pass --occany_ckpt explicitly."
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def validate_default_paths(processed_root: str, kittiodo_root: str) -> None:
    processed = Path(processed_root)
    kitti = Path(kittiodo_root)
    expected = [
        processed / "train_00",
        processed / "train_00" / "voxels",
        processed / "train_00" / "000005_0.npz",
        processed / "train_00" / "voxels" / "000005.npz",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "processed_root does not match the expected kitti_processed layout. "
            f"Missing: {missing}"
        )

    calib_candidates = [
        kitti / "dataset" / "sequences" / "00" / "calib.txt",
        kitti / "sequences" / "00" / "calib.txt",
    ]
    calib_path = next((p for p in calib_candidates if p.exists()), None)
    if calib_path is None:
        raise FileNotFoundError(
            "kittiodo_root does not contain KITTI odometry calib.txt. "
            f"Tried: {[str(p) for p in calib_candidates]}"
        )
    if "Tr:" not in calib_path.read_text():
        raise RuntimeError(
            f"{calib_path} exists but does not contain the required 'Tr:' line. "
            "Use the KITTI Odometry calibration root, not the SemanticKITTI label-only "
            "calib directory."
        )


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


def sample_indices(n_items: int, n_samples: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    n = min(max(n_samples, 0), n_items)
    return rng.sample(range(n_items), n)


def patch_points_from_pointmap(
    p_rec_global: torch.Tensor,
    c_rec: torch.Tensor,
    patch_size: int,
    conf_clamp_max: float = 50.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return one reference-frame 3D point per decoder patch token.

    Args:
      p_rec_global: (N, H, W, 3), pointmaps in reference camera coords.
      c_rec: (N, H, W), OccAny confidence.

    Returns:
      patch_points: (N, Ht, Wt, 3)
      valid: (N, Ht, Wt)
    """
    n, h, w, _ = p_rec_global.shape
    if h % patch_size or w % patch_size:
        raise RuntimeError(f"pointmap shape {(h, w)} is not divisible by patch_size={patch_size}")
    ht, wt = h // patch_size, w // patch_size

    pts = p_rec_global.view(n, ht, patch_size, wt, patch_size, 3)
    conf = c_rec.view(n, ht, patch_size, wt, patch_size)
    finite = torch.isfinite(pts).all(dim=-1) & torch.isfinite(conf) & (conf > 0)
    weights = conf.clamp(max=conf_clamp_max) * finite.to(conf.dtype)
    sum_w = weights.sum(dim=(2, 4))
    sum_wp = (pts * weights.unsqueeze(-1)).sum(dim=(2, 4))
    patch_points = sum_wp / sum_w.clamp_min(1e-6).unsqueeze(-1)
    valid = sum_w > 0
    return patch_points, valid


def voxel_counts_from_patch_points(
    patch_points_refcam: torch.Tensor,
    valid: torch.Tensor,
    T_target_from_refcam: torch.Tensor,
    voxel_origin: torch.Tensor,
    voxel_size: torch.Tensor,
    grid_size: Iterable[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project patch points into voxels and count patch tokens per voxel."""
    device = patch_points_refcam.device
    dtype = patch_points_refcam.dtype
    grid = tuple(int(v) for v in grid_size)
    x_size, y_size, z_size = grid

    points_flat = patch_points_refcam.reshape(-1, 3)
    valid_flat = valid.reshape(-1)

    T = T_target_from_refcam.to(device=device, dtype=dtype)
    points_target = points_flat @ T[:3, :3].T + T[:3, 3]

    origin = voxel_origin.to(device=device, dtype=dtype)
    vsize = voxel_size.to(device=device, dtype=dtype)
    idx = torch.floor((points_target - origin) / vsize).long()
    in_bounds = (
        (idx[:, 0] >= 0)
        & (idx[:, 0] < x_size)
        & (idx[:, 1] >= 0)
        & (idx[:, 1] < y_size)
        & (idx[:, 2] >= 0)
        & (idx[:, 2] < z_size)
    )
    keep = valid_flat & torch.isfinite(points_target).all(dim=-1) & in_bounds
    idx_kept = idx[keep]

    counts = torch.zeros(grid, device=device, dtype=torch.int32)
    if idx_kept.numel() > 0:
        lin = (idx_kept[:, 0] * y_size + idx_kept[:, 1]) * z_size + idx_kept[:, 2]
        binc = torch.bincount(lin, minlength=x_size * y_size * z_size)
        counts = binc.view(grid).to(torch.int32)

    return counts, idx_kept, points_target[keep], keep


def pixel_projection_from_pointmap(
    p_rec_global: torch.Tensor,
    c_rec: torch.Tensor,
    views: List[Dict[str, torch.Tensor]],
    T_target_from_refcam: torch.Tensor,
    voxel_origin: torch.Tensor,
    voxel_size: torch.Tensor,
    grid_size: Iterable[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project every valid image pixel point into target 3D/voxel coordinates."""
    device = p_rec_global.device
    dtype = p_rec_global.dtype
    n, h, w, _ = p_rec_global.shape
    grid = tuple(int(v) for v in grid_size)
    x_size, y_size, z_size = grid

    pts = p_rec_global.reshape(n * h * w, 3)
    conf = c_rec.reshape(n * h * w)
    valid = torch.isfinite(pts).all(dim=-1) & torch.isfinite(conf) & (conf > 0)

    T = T_target_from_refcam.to(device=device, dtype=dtype)
    points_target = pts @ T[:3, :3].T + T[:3, 3]

    origin = voxel_origin.to(device=device, dtype=dtype)
    vsize = voxel_size.to(device=device, dtype=dtype)
    idx = torch.floor((points_target - origin) / vsize).long()
    in_bounds = (
        (idx[:, 0] >= 0)
        & (idx[:, 0] < x_size)
        & (idx[:, 1] >= 0)
        & (idx[:, 1] < y_size)
        & (idx[:, 2] >= 0)
        & (idx[:, 2] < z_size)
    )
    keep = valid & torch.isfinite(points_target).all(dim=-1) & in_bounds

    imgs = torch.stack([v["img"][0].detach().cpu() for v in views], dim=0)
    colors_u8 = convert_images_to_uint8_hwc(imgs).reshape(n * h * w, 3)
    colors = torch.from_numpy(colors_u8).to(device=device, dtype=torch.uint8)

    return points_target[keep], colors[keep], conf[keep], idx[keep]


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


def save_pixel_projection_plot(
    points_target: np.ndarray,
    colors_u8: np.ndarray,
    conf: np.ndarray,
    frame_ids: Tuple[int, ...],
    out_path: Path,
    max_points: int,
    seed: int,
    elev: float,
    azim: float,
) -> None:
    if points_target.size == 0:
        fig = plt.figure(figsize=(8, 6), dpi=160)
        fig.text(0.5, 0.5, "No image pixels landed inside the voxel grid.", ha="center")
        fig.savefig(out_path)
        plt.close(fig)
        return

    n = points_target.shape[0]
    if n > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(n, size=max_points, replace=False)
    else:
        keep = np.arange(n)

    pts = points_target[keep]
    rgb = colors_u8[keep].astype(np.float32) / 255.0
    conf_sel = conf[keep]
    sizes = 1.0 + 3.0 * np.sqrt(np.clip(conf_sel, 0.0, 50.0) / 50.0)

    fig = plt.figure(figsize=(10, 8), dpi=170)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        c=rgb,
        s=sizes,
        alpha=0.75,
        linewidths=0.0,
        depthshade=False,
    )
    ax.set_title(
        "Image pixel pointmap projection\n"
        f"frames={frame_ids}, valid_pixels={n}, shown={pts.shape[0]}"
    )
    ax.set_xlabel("x / forward (m)")
    ax.set_ylabel("y / left (m)")
    ax.set_zlabel("z / up (m)")
    ax.set_xlim(0.0, 51.2)
    ax.set_ylim(-25.6, 25.6)
    ax.set_zlim(-2.0, 4.4)
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def select_points_for_plot(
    coords: np.ndarray,
    counts: np.ndarray,
    max_points: int,
    seed: int,
) -> np.ndarray:
    if coords.shape[0] <= max_points:
        return np.arange(coords.shape[0])
    rng = np.random.default_rng(seed)
    top_n = min(max_points // 2, coords.shape[0])
    top = np.argsort(-counts)[:top_n]
    remaining = np.setdiff1d(np.arange(coords.shape[0]), top, assume_unique=False)
    extra_n = max_points - top.shape[0]
    extra = rng.choice(remaining, size=min(extra_n, remaining.shape[0]), replace=False)
    return np.concatenate([top, extra])


def select_text_labels(counts: np.ndarray, max_labels: int, seed: int) -> np.ndarray:
    if max_labels <= 0 or counts.size == 0:
        return np.zeros(counts.shape, dtype=bool)

    label = np.zeros(counts.shape, dtype=bool)
    collision = np.where(counts > 1)[0]
    collision = collision[np.argsort(-counts[collision])]
    chosen = collision[:max_labels]
    label[chosen] = True

    remaining_slots = max_labels - chosen.size
    if remaining_slots > 0:
        ones = np.where((counts == 1) & (~label))[0]
        if ones.size > 0:
            rng = np.random.default_rng(seed)
            extra = rng.choice(ones, size=min(remaining_slots, ones.size), replace=False)
            label[extra] = True
    return label


def save_3d_count_plot(
    counts_np: np.ndarray,
    voxel_origin: np.ndarray,
    voxel_size: np.ndarray,
    frame_ids: Tuple[int, ...],
    out_path: Path,
    max_points: int,
    max_text_labels: int,
    seed: int,
    elev: float,
    azim: float,
) -> None:
    coords = np.argwhere(counts_np > 0)
    vals = counts_np[counts_np > 0]
    if coords.size == 0:
        fig = plt.figure(figsize=(8, 6), dpi=160)
        fig.text(0.5, 0.5, "No patch tokens landed inside the voxel grid.", ha="center")
        fig.savefig(out_path)
        plt.close(fig)
        return

    plot_sel = select_points_for_plot(coords, vals, max_points=max_points, seed=seed)
    coords_plot = coords[plot_sel]
    vals_plot = vals[plot_sel]
    centers = voxel_origin[None, :] + (coords_plot.astype(np.float32) + 0.5) * voxel_size[None, :]

    fig = plt.figure(figsize=(10, 8), dpi=170)
    ax = fig.add_subplot(111, projection="3d")
    sizes = 10.0 + 10.0 * np.sqrt(vals_plot.astype(np.float32))
    sc = ax.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        c=vals_plot,
        s=sizes,
        cmap="viridis",
        alpha=0.85,
        linewidths=0.0,
    )
    fig.colorbar(sc, ax=ax, shrink=0.62, pad=0.08, label="patch tokens per voxel")

    label_mask = select_text_labels(vals_plot, max_text_labels, seed=seed)
    for xyz, count in zip(centers[label_mask], vals_plot[label_mask]):
        ax.text(xyz[0], xyz[1], xyz[2], str(int(count)), fontsize=6, color="black")

    ax.set_title(
        "Patch-token voxel projection counts\n"
        f"frames={frame_ids}, hit_voxels={coords.shape[0]}, shown={coords_plot.shape[0]}"
    )
    ax.set_xlabel("x / forward (m)")
    ax.set_ylabel("y / left (m)")
    ax.set_zlabel("z / up (m)")
    ax.set_xlim(0.0, 51.2)
    ax.set_ylim(-25.6, 25.6)
    ax.set_zlim(-2.0, 4.4)
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_projection_maps(counts_np: np.ndarray, out_path: Path) -> None:
    """Save max-count projections for quick inspection."""
    projections = [
        ("XY max over Z", counts_np.max(axis=2).T, "x index", "y index"),
        ("XZ max over Y", counts_np.max(axis=1).T, "x index", "z index"),
        ("YZ max over X", counts_np.max(axis=0).T, "y index", "z index"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=160)
    vmax = max(int(counts_np.max()), 1)
    for ax, (title, img, xlabel, ylabel) in zip(axes, projections):
        im = ax.imshow(img, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="max patch count")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    args = get_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    validate_default_paths(args.processed_root, args.kittiodo_root)

    device = resolve_device(args.device)
    occany_ckpt = resolve_occany_ckpt(args.occany_ckpt)
    backbone_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    dataset = Kitti5FrameStage1Dataset(
        processed_root=args.processed_root,
        kittiodo_root=args.kittiodo_root,
        split=args.split,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        output_resolution=(args.width, args.height),
        cam_idx=0,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found for split={args.split!r}")

    backbone = OccAnyRecon5FrameBackbone(
        img_size=(args.height, args.width),
        backbone_dtype=backbone_dtype,
    ).to(device)
    print(f"Loading pretrained OccAny reconstruction checkpoint: {occany_ckpt}")
    backbone.load_checkpoint(str(occany_ckpt))
    backbone.eval()

    indices = sample_indices(len(dataset), args.num_samples, args.seed)
    print(f"Visualizing dataset indices: {indices}")

    for out_i, ds_i in enumerate(indices):
        sample = dataset[ds_i]
        batch = collate_stage1([sample])
        views = move_views_to_device(batch["views"], device)
        frame_ids = tuple(int(v) for v in sample["frame_ids"])
        seq = str(sample["sequence"])
        target_frame = int(sample["target_frame_id"])
        stem = f"{out_i:02d}_{seq}_{target_frame:06d}"

        backbone_out = backbone(views)
        patch_points, valid = patch_points_from_pointmap(
            backbone_out["p_rec_global"][0],
            backbone_out["c_rec"][0],
            patch_size=backbone.patch_size,
        )

        counts, idx_kept, points_kept, keep = voxel_counts_from_patch_points(
            patch_points_refcam=patch_points,
            valid=valid,
            T_target_from_refcam=batch["T_target_from_refcam"][0].to(device),
            voxel_origin=batch["voxel_origin"][0].to(device),
            voxel_size=batch["voxel_size"][0].to(device),
            grid_size=tuple(int(v) for v in batch["grid_size"][0].tolist()),
        )

        counts_np = counts.cpu().numpy()
        voxel_origin_np = batch["voxel_origin"][0].cpu().numpy().astype(np.float32)
        voxel_size_np = batch["voxel_size"][0].cpu().numpy().astype(np.float32)

        pixel_points, pixel_colors, pixel_conf, pixel_voxel_idx = pixel_projection_from_pointmap(
            p_rec_global=backbone_out["p_rec_global"][0],
            c_rec=backbone_out["c_rec"][0],
            views=views,
            T_target_from_refcam=batch["T_target_from_refcam"][0].to(device),
            voxel_origin=batch["voxel_origin"][0].to(device),
            voxel_size=batch["voxel_size"][0].to(device),
            grid_size=tuple(int(v) for v in batch["grid_size"][0].tolist()),
        )

        npz_path = out_dir / f"{stem}_patch_voxel_counts.npz"
        np.savez_compressed(
            npz_path,
            counts=counts_np,
            hit_voxel_indices=idx_kept.cpu().numpy().astype(np.int32),
            hit_points_target=points_kept.cpu().numpy().astype(np.float32),
            patch_valid_in_grid=keep.cpu().numpy(),
            voxel_origin=voxel_origin_np,
            voxel_size=voxel_size_np,
            grid_size=np.array(counts_np.shape, dtype=np.int32),
            frame_ids=np.array(frame_ids, dtype=np.int32),
            sequence=seq,
            target_frame_id=np.array(target_frame, dtype=np.int32),
        )
        np.savez_compressed(
            out_dir / f"{stem}_pixel_projection.npz",
            points_target=pixel_points.cpu().numpy().astype(np.float32),
            colors_rgb=pixel_colors.cpu().numpy().astype(np.uint8),
            confidence=pixel_conf.cpu().numpy().astype(np.float32),
            voxel_indices=pixel_voxel_idx.cpu().numpy().astype(np.int32),
            frame_ids=np.array(frame_ids, dtype=np.int32),
            sequence=seq,
            target_frame_id=np.array(target_frame, dtype=np.int32),
        )

        save_frame_montage(views, frame_ids, out_dir / f"{stem}_frames.png")
        save_pixel_projection_plot(
            points_target=pixel_points.cpu().numpy().astype(np.float32),
            colors_u8=pixel_colors.cpu().numpy().astype(np.uint8),
            conf=pixel_conf.cpu().numpy().astype(np.float32),
            frame_ids=frame_ids,
            out_path=out_dir / f"{stem}_pixel_projection_3d.png",
            max_points=args.max_pixel_points,
            seed=args.seed + out_i,
            elev=args.elev,
            azim=args.azim,
        )
        save_3d_count_plot(
            counts_np=counts_np,
            voxel_origin=voxel_origin_np,
            voxel_size=voxel_size_np,
            frame_ids=frame_ids,
            out_path=out_dir / f"{stem}_patch_voxel_counts_3d.png",
            max_points=args.max_points,
            max_text_labels=args.max_text_labels,
            seed=args.seed + out_i,
            elev=args.elev,
            azim=args.azim,
        )
        save_projection_maps(counts_np, out_dir / f"{stem}_patch_voxel_counts_proj.png")

        print(
            f"[{out_i + 1}/{len(indices)}] {stem}: "
            f"patch_tokens={valid.numel()}, valid_patch_tokens={int(valid.sum())}, "
            f"in_grid_patch_tokens={int(keep.sum())}, hit_voxels={int((counts > 0).sum())}, "
            f"max_count={int(counts.max())}, in_grid_pixels={pixel_points.shape[0]}; "
            f"saved under {out_dir}"
        )


if __name__ == "__main__":
    main()
