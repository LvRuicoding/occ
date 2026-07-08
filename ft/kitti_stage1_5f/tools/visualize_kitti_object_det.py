"""Visualize KITTI Object detections from a Stage-1 detection checkpoint."""
from __future__ import annotations

try:
    from .. import _paths  # noqa: F401
except ImportError:
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    from ft.kitti_stage1_5f import _paths  # noqa: F401

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw, ImageFont

from occany.model.must3r_blocks.attention import toggle_memory_efficient_attention
from occany.utils.checkpoint_io import register_legacy_checkpoint_modules

from ft.kitti_stage1_5f.datasets import (
    KITTI_OBJECT_CLASS_NAMES,
    KittiObject5FrameDetDataset,
    collate_kitti_object_det,
)
from ft.kitti_stage1_5f.datasets.kitti_object_det import _parse_object_calib
from ft.kitti_stage1_5f.tools.eval_kitti_object_det import (
    _build_model,
    _fill_args_from_checkpoint,
    _resolve_ckpt_path,
)
from ft.kitti_stage1_5f.tools.train import _model_forward, _strip_module_prefix


GT_COLOR = (25, 220, 80)
PRED_COLOR = (255, 70, 70)
TEXT_BG_GT = (0, 95, 30)
TEXT_BG_PRED = (130, 0, 0)
BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def get_args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Visualize KITTI Object detections")
    p.add_argument("--ckpt", required=True, type=str, help="Checkpoint file or directory with checkpoint-last.pth.")
    p.add_argument("--sample-index", default=0, type=int, help="Index in the selected split after reliable-history filtering.")
    p.add_argument("--sample-id", default=None, type=str, help="Raw KITTI Object sample id, e.g. 000123. Overrides --sample-index.")
    p.add_argument("--split", choices=["train", "val", "trainval"], default="val")
    p.add_argument("--kitti_det_root", default=None, type=str, help="Override KITTI Object root.")
    p.add_argument("--occany_ckpt", default=None, type=str, help="Override OccAny backbone checkpoint.")
    p.add_argument("--device", default="auto", type=str)
    p.add_argument("--amp", choices=["bf16", "fp16", "none"], default=None)
    p.add_argument("--score_threshold", default=None, type=float, help="Override decode score threshold.")
    p.add_argument("--max_points_per_sweep", default=None, type=int)
    p.add_argument("--max-preds", default=50, type=int, help="Maximum predictions to draw after score sorting; <=0 draws all.")
    p.add_argument("--point-stride", default=3, type=int, help="Stride used when plotting LiDAR points in BEV.")
    p.add_argument("--output_dir", default=None, type=str)
    p.add_argument("--output", default=None, type=str, help="Optional exact PNG output path.")
    p.add_argument("--dpi", default=160, type=int)
    p.add_argument("--seed", default=0, type=int)

    # Present for compatibility with _fill_args_from_checkpoint.
    p.add_argument("--batch_size", default=1, type=int)
    p.add_argument("--num_workers", default=0, type=int)
    return p


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _build_dataset(args: argparse.Namespace) -> KittiObject5FrameDetDataset:
    return KittiObject5FrameDetDataset(
        root=args.kitti_det_root,
        split=args.split,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        output_resolution=(args.width, args.height),
        max_points_per_sweep=args.max_points_per_sweep,
    )


def _select_sample_index(dataset: KittiObject5FrameDetDataset, args: argparse.Namespace) -> int:
    if args.sample_id is not None:
        sample_id = int(str(args.sample_id), 10)
        for idx, (target_id, _frame_ids) in enumerate(dataset.samples):
            if int(target_id) == sample_id:
                return idx
        raise ValueError(
            f"sample_id={sample_id:06d} is not present in split={args.split!r} "
            f"after reliable-history filtering."
        )
    index = int(args.sample_index)
    if index < 0 or index >= len(dataset):
        raise IndexError(f"sample-index={index} out of range for split={args.split!r}; len={len(dataset)}.")
    return index


def _box_corners_lidar(box: Iterable[float]) -> np.ndarray:
    x, y, z, length, width, height, yaw = [float(v) for v in box]
    dx = length * 0.5
    dy = width * 0.5
    dz = height * 0.5
    corners = np.array(
        [
            [dx, dy, dz],
            [dx, -dy, dz],
            [-dx, -dy, dz],
            [-dx, dy, dz],
            [dx, dy, -dz],
            [dx, -dy, -dz],
            [-dx, -dy, -dz],
            [-dx, dy, -dz],
        ],
        dtype=np.float64,
    )
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return (rot @ corners.T).T + np.array([x, y, z], dtype=np.float64).reshape(1, 3)


def _project_lidar_corners(
    corners_lidar: np.ndarray,
    T_cam_from_velo: np.ndarray,
    P2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    homo = np.concatenate([corners_lidar, np.ones((corners_lidar.shape[0], 1), dtype=np.float64)], axis=1)
    cam = (T_cam_from_velo.astype(np.float64) @ homo.T).T[:, :3]
    valid = cam[:, 2] > 1e-3
    proj_homo = np.concatenate([cam, np.ones((cam.shape[0], 1), dtype=np.float64)], axis=1)
    proj = (P2.astype(np.float64) @ proj_homo.T).T
    uv = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-6)
    return uv, valid


def _draw_text(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], text: str, color: Tuple[int, int, int]) -> None:
    font = ImageFont.load_default()
    x, y = float(xy[0]), float(xy[1])
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 2
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=color)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def _draw_box_on_image(
    draw: ImageDraw.ImageDraw,
    box: Iterable[float],
    label: int,
    score: float | None,
    calib: dict,
    color: Tuple[int, int, int],
    text_bg: Tuple[int, int, int],
    width: int,
) -> None:
    box = [float(v) for v in box]
    if box[3] <= 0.0 or box[4] <= 0.0 or box[5] <= 0.0:
        return
    corners = _box_corners_lidar(box)
    uv, valid = _project_lidar_corners(corners, calib["T_cam_from_velo"], calib["P2"])
    if int(valid.sum()) < 2:
        return
    for a, b in BOX_EDGES:
        if not (valid[a] and valid[b]):
            continue
        draw.line((tuple(uv[a]), tuple(uv[b])), fill=color, width=width)
    valid_uv = uv[valid]
    text = KITTI_OBJECT_CLASS_NAMES[int(label)]
    if score is not None:
        text = f"{text} {float(score):.2f}"
    _draw_text(draw, (float(valid_uv[:, 0].min()), max(0.0, float(valid_uv[:, 1].min()) - 12.0)), text, text_bg)


def _sort_and_limit_predictions(pred: dict, max_preds: int) -> dict:
    scores = pred["scores_3d"].detach().cpu()
    order = torch.argsort(scores, descending=True)
    if int(max_preds) > 0:
        order = order[: int(max_preds)]
    return {
        "boxes_3d": pred["boxes_3d"].detach().cpu()[order],
        "scores_3d": pred["scores_3d"].detach().cpu()[order],
        "labels_3d": pred["labels_3d"].detach().cpu()[order],
    }


def _make_image_overlay(
    dataset: KittiObject5FrameDetDataset,
    sample_id: int,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    pred: dict,
) -> Image.Image:
    image = Image.open(dataset._image_path(sample_id)).convert("RGB")
    calib = _parse_object_calib(dataset._calib_path(sample_id))
    draw = ImageDraw.Draw(image)
    for box, label in zip(gt_boxes.cpu().numpy(), gt_labels.cpu().numpy()):
        _draw_box_on_image(draw, box, int(label), None, calib, GT_COLOR, TEXT_BG_GT, width=3)
    for box, score, label in zip(
        pred["boxes_3d"].cpu().numpy(),
        pred["scores_3d"].cpu().numpy(),
        pred["labels_3d"].cpu().numpy(),
    ):
        _draw_box_on_image(draw, box, int(label), float(score), calib, PRED_COLOR, TEXT_BG_PRED, width=2)
    return image


def _plot_bev(
    ax: plt.Axes,
    points: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    pred: dict,
    point_stride: int,
) -> None:
    pc_range = (0.0, -25.6, -2.0, 51.2, 25.6, 4.4)
    pts = points.detach().cpu().numpy()
    keep = (
        (pts[:, 0] >= pc_range[0])
        & (pts[:, 0] <= pc_range[3])
        & (pts[:, 1] >= pc_range[1])
        & (pts[:, 1] <= pc_range[4])
        & (pts[:, 2] >= pc_range[2])
        & (pts[:, 2] <= pc_range[5])
    )
    pts = pts[keep]
    stride = max(1, int(point_stride))
    pts = pts[::stride]
    if pts.size:
        ax.scatter(pts[:, 1], pts[:, 0], s=0.15, c="#666666", alpha=0.35, linewidths=0)

    def draw_bev_box(box: np.ndarray, color: str, label_text: str, linewidth: float) -> None:
        corners = _box_corners_lidar(box)[:4]
        loop = np.concatenate([corners, corners[:1]], axis=0)
        ax.plot(loop[:, 1], loop[:, 0], color=color, linewidth=linewidth)
        front = corners[[0, 1]]
        ax.plot(front[:, 1], front[:, 0], color=color, linewidth=linewidth + 0.8)
        ax.text(
            float(corners[:, 1].mean()),
            float(corners[:, 0].mean()),
            label_text,
            color=color,
            fontsize=6,
            ha="center",
            va="center",
        )

    for box, label in zip(gt_boxes.cpu().numpy(), gt_labels.cpu().numpy()):
        draw_bev_box(box, "#19d957", f"GT {KITTI_OBJECT_CLASS_NAMES[int(label)]}", linewidth=1.4)
    for box, score, label in zip(
        pred["boxes_3d"].cpu().numpy(),
        pred["scores_3d"].cpu().numpy(),
        pred["labels_3d"].cpu().numpy(),
    ):
        draw_bev_box(box, "#ff4d4d", f"{KITTI_OBJECT_CLASS_NAMES[int(label)]} {float(score):.2f}", linewidth=1.0)

    ax.set_xlim(pc_range[1], pc_range[4])
    ax.set_ylim(pc_range[0], pc_range[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#dddddd", linewidth=0.4)
    ax.set_xlabel("y left/right (m)")
    ax.set_ylabel("x forward (m)")
    ax.set_title("LiDAR BEV")


def _prediction_rows(pred: dict) -> List[dict]:
    rows = []
    for box, score, label in zip(pred["boxes_3d"], pred["scores_3d"], pred["labels_3d"]):
        rows.append(
            {
                "class": KITTI_OBJECT_CLASS_NAMES[int(label)],
                "score": round(float(score), 5),
                "box_lidar_xyzwlh_yaw": [round(float(v), 5) for v in box.tolist()],
            }
        )
    return rows


@torch.no_grad()
def main() -> None:
    args = get_args_parser().parse_args()
    ckpt_path = _resolve_ckpt_path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _fill_args_from_checkpoint(args, ckpt.get("args", {}))
    args.batch_size = 1
    args.num_workers = 0

    if args.output_dir is None:
        args.output_dir = str(ckpt_path.parent / "vis_kitti_object_det")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    toggle_memory_efficient_attention(enabled=False)
    register_legacy_checkpoint_modules()
    cudnn.benchmark = True
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = _resolve_device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    model = _build_model(args, device)
    state = ckpt.get("model", None)
    if state is None:
        raise KeyError(f"Checkpoint {ckpt_path} does not contain key 'model'.")
    status = model.load_state_dict(_strip_module_prefix(state), strict=False)
    model.eval()

    dataset = _build_dataset(args)
    sample_index = _select_sample_index(dataset, args)
    sample = dataset[sample_index]
    batch = collate_kitti_object_det([sample])
    sample_id = int(sample["sample_id"])

    amp_dtype = torch.bfloat16 if args.amp == "bf16" else (torch.float16 if args.amp == "fp16" else None)
    ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None and device.type == "cuda"
        else torch.autocast(device_type=device.type, enabled=False)
    )
    with ctx:
        out = _model_forward(model, batch, device, args)
        pred = model.det_decode(out["det_preds"])[0]
    pred = _sort_and_limit_predictions(pred, args.max_preds)

    overlay = _make_image_overlay(
        dataset,
        sample_id,
        sample["gt_bboxes_3d"],
        sample["gt_labels_3d"],
        pred,
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    axes[0].imshow(overlay)
    axes[0].axis("off")
    axes[0].set_title(
        f"KITTI Object {sample_id:06d} | green=GT red=pred | exp={args.exp}"
    )
    _plot_bev(
        axes[1],
        sample["points_per_frame"][0],
        sample["gt_bboxes_3d"],
        sample["gt_labels_3d"],
        pred,
        point_stride=args.point_stride,
    )

    if args.output is None:
        out_png = output_dir / f"sample_{sample_id:06d}_idx_{sample_index:04d}.png"
    else:
        out_png = Path(args.output)
        out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=int(args.dpi))
    plt.close(fig)

    summary = {
        "checkpoint": str(ckpt_path),
        "exp": args.exp,
        "split": args.split,
        "sample_index": sample_index,
        "sample_id": sample_id,
        "frame_ids": [int(v) for v in sample["frame_ids"]],
        "raw_mapping": list(sample["raw_mapping"]),
        "gt_count": int(sample["gt_bboxes_3d"].shape[0]),
        "pred_count": int(pred["boxes_3d"].shape[0]),
        "score_threshold": float(args.det_score_threshold),
        "load_missing_keys": len(status.missing_keys),
        "load_unexpected_keys": len(status.unexpected_keys),
        "predictions": _prediction_rows(pred),
    }
    summary_path = out_png.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(
        f"[vis] exp={args.exp} split={args.split} sample_index={sample_index} "
        f"sample_id={sample_id:06d} gt={summary['gt_count']} pred={summary['pred_count']} "
        f"device={device} amp={args.amp}"
    )
    print(f"[load] missing={len(status.missing_keys)} unexpected={len(status.unexpected_keys)}")
    print(f"[output] png={out_png}")
    print(f"[output] json={summary_path}")


if __name__ == "__main__":
    main()
