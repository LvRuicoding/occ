"""KITTI Object val evaluation for Stage-1 detection checkpoints.

Example:
  /home/dataset-local/envs/occany/bin/python -m ft.kitti_stage1_5f.tools.eval_kitti_object_det \
    --ckpt output/kitti_stage1_5f_4gpu_det_postfusion_only/checkpoint-last.pth
"""
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
import pickle
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Subset

import croco.utils.misc as misc
import dust3r.utils.path_to_croco  # noqa: F401
from occany.model.must3r_blocks.attention import toggle_memory_efficient_attention
from occany.utils.checkpoint_io import register_legacy_checkpoint_modules

from ft.kitti_stage1_5f.datasets import (
    KITTI_OBJECT_CLASS_NAMES,
    KittiObject5FrameDetDataset,
    collate_kitti_object_det,
)
from ft.kitti_stage1_5f.datasets.kitti_object_det import (
    KITTI_OBJECT_LEGACY_DET_DEPTH_BOUND,
    KITTI_OBJECT_LEGACY_DET_PC_RANGE,
    _normalize_angle,
    _parse_object_calib,
    make_kitti_object_det_grid_config,
)
from ft.kitti_stage1_5f.kitti_object_official_eval import kitti_object_eval
from ft.kitti_stage1_5f.models import (
    Stage1DetOriginalModel,
    Stage1DetPostFusionOnlyModel,
)
from ft.kitti_stage1_5f.tools.train import (
    DET_EXPS,
    _model_forward,
    _strip_module_prefix,
)


def get_args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("KITTI Object val evaluation for Stage-1 detection checkpoints")
    p.add_argument("--ckpt", required=True, type=str, help="Checkpoint file or directory with checkpoint-last.pth.")
    p.add_argument(
        "--exp",
        choices=DET_EXPS,
        default=None,
        help="Detection experiment type. Overrides checkpoint args when provided.",
    )
    p.add_argument("--kitti_det_root", default=None, type=str, help="Override KITTI Object root.")
    p.add_argument("--occany_ckpt", default=None, type=str, help="Override OccAny backbone checkpoint.")
    p.add_argument("--device", default="auto", type=str)
    p.add_argument("--batch_size", default=None, type=int)
    p.add_argument("--num_workers", default=None, type=int)
    p.add_argument("--amp", choices=["bf16", "fp16", "none"], default=None)
    p.add_argument("--score_threshold", default=None, type=float, help="Override decode score threshold.")
    p.add_argument("--det_pc_range", nargs=6, type=float, default=None,
                   metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
                   help="Override checkpoint KITTI Object DET pc_range.")
    p.add_argument("--det_depth_bound", nargs=3, type=float, default=None,
                   metavar=("START", "END", "STEP"),
                   help="Override checkpoint KITTI Object DET LSS depth bound.")
    p.add_argument("--max_points_per_sweep", default=None, type=int)
    p.add_argument("--max_batches", default=0, type=int, help="Debug only; 0 evaluates full val set.")
    p.add_argument("--print_freq", default=20, type=int)
    p.add_argument("--output_dir", default=None, type=str)
    p.add_argument("--output_json", default=None, type=str)
    p.add_argument("--save_annos", action="store_true", help="Save gt/dt annos as pickle.")
    p.add_argument(
        "--eval_backend",
        choices=("official_cpp", "python"),
        default="official_cpp",
        help="Use KITTI devkit C++ evaluator by default; 'python' keeps the local AP40 reimplementation.",
    )
    p.add_argument(
        "--official_eval_cpp",
        default=None,
        type=str,
        help="Path to KITTI devkit cpp/evaluate_object.cpp or a compiled evaluate_object binary. "
        "Defaults to <kitti_det_root>/cpp/evaluate_object.cpp.",
    )
    p.add_argument(
        "--official_eval_num_images",
        default=7518,
        type=int,
        help="Number of files expected by the official C++ evaluator source/binary.",
    )
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--world_size", default=1, type=int)
    p.add_argument("--local_rank", default=-1, type=int)
    p.add_argument("--dist_url", default="env://", type=str)
    p.add_argument("--nodist", action="store_true", help="Disable distributed mode under torchrun.")
    return p


def _resolve_ckpt_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_dir():
        path = path / "checkpoint-last.pth"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _ckpt_arg(ckpt_args, name: str, default):
    if isinstance(ckpt_args, dict):
        return ckpt_args.get(name, default)
    return getattr(ckpt_args, name, default)


def _override_or_ckpt(args: argparse.Namespace, ckpt_args, name: str, default):
    value = getattr(args, name)
    return value if value is not None else _ckpt_arg(ckpt_args, name, default)


def _fill_args_from_checkpoint(args: argparse.Namespace, ckpt_args) -> None:
    args.exp = _override_or_ckpt(args, ckpt_args, "exp", None)
    if args.exp not in DET_EXPS:
        raise ValueError(f"Expected a detection checkpoint exp in {DET_EXPS}, got {args.exp!r}.")
    args.kitti_det_root = _override_or_ckpt(args, ckpt_args, "kitti_det_root", None)
    args.occany_ckpt = _override_or_ckpt(args, ckpt_args, "occany_ckpt", None)
    if not args.kitti_det_root:
        raise ValueError("Checkpoint args do not contain kitti_det_root; pass --kitti_det_root.")
    if not args.occany_ckpt:
        raise ValueError("Checkpoint args do not contain occany_ckpt; pass --occany_ckpt.")

    args.width = int(_ckpt_arg(ckpt_args, "width", 512))
    args.height = int(_ckpt_arg(ckpt_args, "height", 160))
    args.num_frames = int(_ckpt_arg(ckpt_args, "num_frames", 5))
    args.frame_stride = int(_ckpt_arg(ckpt_args, "frame_stride", 4))
    args.c_lift = int(_ckpt_arg(ckpt_args, "c_lift", 64))
    args.token_dim = int(_ckpt_arg(ckpt_args, "token_dim", 768))
    args.patch_size = int(_ckpt_arg(ckpt_args, "patch_size", 16))
    args.backbone = _ckpt_arg(ckpt_args, "backbone", "must3r")
    args.freeze_backbone = bool(_ckpt_arg(ckpt_args, "freeze_backbone", True))
    args.det_score_threshold = (
        float(args.score_threshold)
        if args.score_threshold is not None
        else float(_ckpt_arg(ckpt_args, "det_score_threshold", 0.05))
    )
    args.det_pc_range = tuple(
        float(v)
        for v in _override_or_ckpt(
            args,
            ckpt_args,
            "det_pc_range",
            KITTI_OBJECT_LEGACY_DET_PC_RANGE,
        )
    )
    args.det_depth_bound = tuple(
        float(v)
        for v in _override_or_ckpt(
            args,
            ckpt_args,
            "det_depth_bound",
            KITTI_OBJECT_LEGACY_DET_DEPTH_BOUND,
        )
    )
    args.max_points_per_sweep = int(
        args.max_points_per_sweep
        if args.max_points_per_sweep is not None
        else _ckpt_arg(ckpt_args, "max_points_per_sweep", 0)
    )
    args.batch_size = int(args.batch_size if args.batch_size is not None else _ckpt_arg(ckpt_args, "batch_size", 1))
    args.num_workers = int(args.num_workers if args.num_workers is not None else _ckpt_arg(ckpt_args, "num_workers", 4))
    args.amp = args.amp or _ckpt_arg(ckpt_args, "amp", "bf16")
    args.multi_dataset = False
    args.processed_root = None


def _build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if args.amp == "bf16" and device.type == "cuda":
        backbone_dtype = torch.bfloat16
    elif args.amp == "fp16" and device.type == "cuda":
        backbone_dtype = torch.float16
    else:
        backbone_dtype = torch.float32
    model_cls = Stage1DetOriginalModel if args.exp == "det_original" else Stage1DetPostFusionOnlyModel
    model = model_cls(
        occany_ckpt=args.occany_ckpt,
        c_lift=args.c_lift,
        patch_size=args.patch_size,
        token_dim=args.token_dim,
        backbone_img_size=(args.height, args.width),
        backbone_dtype=backbone_dtype,
        num_frames=args.num_frames,
        freeze_backbone=args.freeze_backbone,
        backbone=args.backbone,
        det_score_threshold=args.det_score_threshold,
        det_pc_range=args.det_pc_range,
        depth_bound=args.det_depth_bound,
    )
    return model.to(device)


def _empty_kitti_anno(with_score: bool = False) -> Dict[str, np.ndarray]:
    anno = {
        "name": np.array([], dtype=object),
        "truncated": np.zeros((0,), dtype=np.float64),
        "occluded": np.zeros((0,), dtype=np.int64),
        "alpha": np.zeros((0,), dtype=np.float64),
        "bbox": np.zeros((0, 4), dtype=np.float64),
        "dimensions": np.zeros((0, 3), dtype=np.float64),
        "location": np.zeros((0, 3), dtype=np.float64),
        "rotation_y": np.zeros((0,), dtype=np.float64),
    }
    if with_score:
        anno["score"] = np.zeros((0,), dtype=np.float64)
    return anno


def _kitti_anno_from_lists(values: Dict[str, List], with_score: bool = False) -> Dict[str, np.ndarray]:
    if not values["name"]:
        return _empty_kitti_anno(with_score=with_score)
    anno = {
        "name": np.asarray(values["name"], dtype=object),
        "truncated": np.asarray(values["truncated"], dtype=np.float64),
        "occluded": np.asarray(values["occluded"], dtype=np.int64),
        "alpha": np.asarray(values["alpha"], dtype=np.float64),
        "bbox": np.asarray(values["bbox"], dtype=np.float64).reshape(-1, 4),
        "dimensions": np.asarray(values["dimensions"], dtype=np.float64).reshape(-1, 3),
        "location": np.asarray(values["location"], dtype=np.float64).reshape(-1, 3),
        "rotation_y": np.asarray(values["rotation_y"], dtype=np.float64),
    }
    if with_score:
        anno["score"] = np.asarray(values["score"], dtype=np.float64)
    return anno


def _load_gt_anno(dataset: KittiObject5FrameDetDataset, sample_id: int) -> Dict[str, np.ndarray]:
    values = {k: [] for k in ("name", "truncated", "occluded", "alpha", "bbox", "dimensions", "location", "rotation_y")}
    with open(dataset._label_path(sample_id), "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 15:
                continue
            values["name"].append(parts[0])
            values["truncated"].append(float(parts[1]))
            values["occluded"].append(int(parts[2]))
            values["alpha"].append(float(parts[3]))
            values["bbox"].append([float(v) for v in parts[4:8]])
            values["dimensions"].append([float(parts[8]), float(parts[9]), float(parts[10])])
            values["location"].append([float(parts[11]), float(parts[12]), float(parts[13])])
            values["rotation_y"].append(float(parts[14]))
    return _kitti_anno_from_lists(values, with_score=False)


def _camera_box_corners(dim_hwl: Tuple[float, float, float], loc: np.ndarray, ry: float) -> np.ndarray:
    h, w, l = dim_hwl
    x_c = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], dtype=np.float64)
    y_c = np.array([0, 0, 0, 0, -h, -h, -h, -h], dtype=np.float64)
    z_c = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], dtype=np.float64)
    c, s = math.cos(float(ry)), math.sin(float(ry))
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    return (R @ np.stack([x_c, y_c, z_c], axis=0)).T + loc.reshape(1, 3)


def _project_bbox(P2: np.ndarray, corners: np.ndarray, image_hw: Tuple[int, int]) -> np.ndarray | None:
    valid = corners[:, 2] > 1e-3
    if not np.any(valid):
        return None
    corners = corners[valid]
    homo = np.concatenate([corners, np.ones((corners.shape[0], 1), dtype=np.float64)], axis=1)
    proj = (P2 @ homo.T).T
    uv = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-6)
    h, w = image_hw
    x1, y1 = uv.min(axis=0)
    x2, y2 = uv.max(axis=0)
    x1 = float(np.clip(x1, 0, max(w - 1, 1)))
    x2 = float(np.clip(x2, 0, max(w - 1, 1)))
    y1 = float(np.clip(y1, 0, max(h - 1, 1)))
    y2 = float(np.clip(y2, 0, max(h - 1, 1)))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def _pred_to_kitti_anno(
    dataset: KittiObject5FrameDetDataset,
    sample_id: int,
    boxes_lidar: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, np.ndarray]:
    values = {k: [] for k in ("name", "truncated", "occluded", "alpha", "bbox", "dimensions", "location", "rotation_y", "score")}
    calib = _parse_object_calib(dataset._calib_path(sample_id))
    P2 = calib["P2"]
    T_cam_from_velo = calib["T_cam_from_velo"].astype(np.float64)
    image = Image.open(dataset._image_path(sample_id))
    image_hw = (int(image.height), int(image.width))
    boxes_np = boxes_lidar.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    for box, score, label in zip(boxes_np, scores_np, labels_np):
        label = int(label)
        if label < 0 or label >= len(KITTI_OBJECT_CLASS_NAMES):
            continue
        x, y, z, length, width, height, yaw = [float(v) for v in box]
        if length <= 0.0 or width <= 0.0 or height <= 0.0:
            continue
        center_cam = T_cam_from_velo @ np.array([x, y, z, 1.0], dtype=np.float64)
        loc = center_cam[:3].copy()
        loc[1] += height * 0.5
        if loc[2] <= 0.1:
            continue
        ry = _normalize_angle(-yaw - math.pi * 0.5)
        dims = (height, width, length)
        bbox = _project_bbox(P2, _camera_box_corners(dims, loc, ry), image_hw)
        if bbox is None:
            continue
        alpha = _normalize_angle(ry - math.atan2(loc[0], loc[2]))
        values["name"].append(KITTI_OBJECT_CLASS_NAMES[label])
        values["truncated"].append(0.0)
        values["occluded"].append(0)
        values["alpha"].append(alpha)
        values["bbox"].append(bbox.tolist())
        values["dimensions"].append([height, width, length])
        values["location"].append(loc.tolist())
        values["rotation_y"].append(ry)
        values["score"].append(float(score))
    return _kitti_anno_from_lists(values, with_score=True)


def _jsonable(value):
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, np.generic):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _format_kitti_float(value: float) -> str:
    return f"{float(value):.6f}"


def _kitti_anno_lines(anno: Dict[str, np.ndarray], with_score: bool) -> List[str]:
    lines: List[str] = []
    for i in range(len(anno["name"])):
        bbox = np.asarray(anno["bbox"][i], dtype=np.float64).reshape(4)
        dims = np.asarray(anno["dimensions"][i], dtype=np.float64).reshape(3)
        loc = np.asarray(anno["location"][i], dtype=np.float64).reshape(3)
        fields = [
            str(anno["name"][i]),
            _format_kitti_float(float(anno["truncated"][i])),
            str(int(anno["occluded"][i])),
            _format_kitti_float(float(anno["alpha"][i])),
            *[_format_kitti_float(v) for v in bbox],
            *[_format_kitti_float(v) for v in dims],
            *[_format_kitti_float(v) for v in loc],
            _format_kitti_float(float(anno["rotation_y"][i])),
        ]
        if with_score:
            fields.append(_format_kitti_float(float(anno["score"][i])))
        lines.append(" ".join(fields))
    return lines


def _write_kitti_anno(path: Path, anno: Dict[str, np.ndarray], with_score: bool) -> None:
    lines = _kitti_anno_lines(anno, with_score=with_score)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _resolve_official_eval_input(path_arg: str | None, kitti_root: str) -> Path:
    if path_arg:
        path = Path(path_arg)
    else:
        path = Path(kitti_root) / "cpp" / "evaluate_object.cpp"
    if path.is_dir():
        cpp = path / "evaluate_object.cpp"
        binary = path / "evaluate_object"
        if cpp.is_file():
            return cpp
        if binary.is_file():
            return binary
    if not path.is_file():
        raise FileNotFoundError(
            f"KITTI official evaluator not found: {path}. "
            "Pass --official_eval_cpp pointing to evaluate_object.cpp or a compiled evaluate_object binary."
        )
    return path


def _get_official_eval_binary(args: argparse.Namespace, output_dir: Path) -> Tuple[Path, Dict[str, str]]:
    eval_input = _resolve_official_eval_input(args.official_eval_cpp, args.kitti_det_root)
    meta = {"input": str(eval_input)}
    if eval_input.suffix.lower() != ".cpp":
        meta["binary"] = str(eval_input)
        return eval_input, meta

    bin_dir = output_dir / "official_eval_bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    binary = bin_dir / "evaluate_object"
    log_path = bin_dir / "compile.log"
    needs_compile = (not binary.is_file()) or binary.stat().st_mtime < eval_input.stat().st_mtime
    if needs_compile:
        cmd = ["g++", "-O3", "-DNDEBUG", str(eval_input), "-o", str(binary)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        log_path.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to compile KITTI official evaluator from {eval_input}; "
                f"see {log_path}."
            )
    meta.update({"source": str(eval_input), "binary": str(binary), "compile_log": str(log_path)})
    return binary, meta


def _prepare_official_eval_tree(
    output_dir: Path,
    sample_ids: Sequence[int],
    gt_annos: Sequence[Dict[str, np.ndarray]],
    dt_annos: Sequence[Dict[str, np.ndarray]],
    num_images: int,
) -> Tuple[Path, Path, str]:
    if not (len(sample_ids) == len(gt_annos) == len(dt_annos)):
        raise ValueError("sample_ids, gt_annos and dt_annos must have the same length.")
    if len(set(int(s) for s in sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate sample ids cannot be written for official KITTI evaluation.")

    num_images = int(num_images)
    max_sample_id = max((int(s) for s in sample_ids), default=-1)
    if max_sample_id >= num_images:
        raise ValueError(
            f"sample_id {max_sample_id} exceeds --official_eval_num_images={num_images}; "
            "this value must match the official evaluator's compiled N_TESTIMAGES."
        )

    work_root = output_dir / "official_kitti_eval"
    result_sha = "occany_val"
    result_dir = work_root / "results" / result_sha
    label_dir = work_root / "data" / "object" / "label_2"
    det_dir = result_dir / "data"
    if work_root.exists():
        shutil.rmtree(work_root)
    label_dir.mkdir(parents=True, exist_ok=True)
    det_dir.mkdir(parents=True, exist_ok=True)

    # The official server evaluator hard-codes a sequential file loop.
    for idx in range(num_images):
        name = f"{idx:06d}.txt"
        (label_dir / name).touch()
        (det_dir / name).touch()

    for sample_id, gt_anno, dt_anno in zip(sample_ids, gt_annos, dt_annos):
        name = f"{int(sample_id):06d}.txt"
        _write_kitti_anno(label_dir / name, gt_anno, with_score=False)
        _write_kitti_anno(det_dir / name, dt_anno, with_score=True)

    (result_dir / "sample_ids.txt").write_text(
        "\n".join(f"{int(s):06d}" for s in sample_ids) + ("\n" if sample_ids else ""),
        encoding="utf-8",
    )
    return work_root, result_dir, result_sha


def _read_official_ap40(stats_path: Path) -> List[float]:
    if not stats_path.is_file():
        return [0.0, 0.0, 0.0]
    values: List[float] = []
    for line in stats_path.read_text(encoding="utf-8").splitlines()[:3]:
        nums = [float(x) for x in line.split()]
        curve = np.zeros((41,), dtype=np.float64)
        if nums:
            n = min(len(nums), 41)
            curve[:n] = np.asarray(nums[:n], dtype=np.float64)
        values.append(float(np.sum(curve[1:41]) / 40.0 * 100.0))
    while len(values) < 3:
        values.append(0.0)
    return values[:3]


def _collect_official_cpp_metrics(
    result_dir: Path,
    current_classes: Sequence[str],
    eval_types: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
    metric_files = {"bbox": "detection", "bev": "detection_ground", "3d": "detection_3d"}
    metric_names = {"bbox": "2D", "bev": "BEV", "3d": "3D"}
    overlaps = {"bbox": (0.7, 0.5, 0.5), "bev": (0.7, 0.5, 0.5), "3d": (0.7, 0.5, 0.5)}
    lines: List[str] = ["", "----------- Official KITTI C++ AP40 Results ------------", ""]
    ret: Dict[str, float] = {}

    for cls_name in current_classes:
        cls_idx = KITTI_OBJECT_CLASS_NAMES.index(cls_name)
        lines.append(
            f"{cls_name} AP40@"
            f"{overlaps['bbox'][cls_idx]:.2f}, {overlaps['bev'][cls_idx]:.2f}, "
            f"{overlaps['3d'][cls_idx]:.2f} (official_cpp):"
        )
        for metric in eval_types:
            vals = _read_official_ap40(result_dir / f"stats_{cls_name.lower()}_{metric_files[metric]}.txt")
            ret_name = metric_names[metric]
            for diff_name, value in zip(("easy", "moderate", "hard"), vals):
                ret[f"KITTI/{cls_name}_{ret_name}_AP40_{diff_name}_official_cpp"] = value
            lines.append(f"{metric:<4} AP40:{vals[0]:.4f}, {vals[1]:.4f}, {vals[2]:.4f}")
        lines.append("")

    if len(current_classes) > 1:
        lines.append("Overall AP40 (official_cpp):")
        for metric in eval_types:
            ret_name = metric_names[metric]
            vals = []
            for diff_name in ("easy", "moderate", "hard"):
                keys = [
                    f"KITTI/{cls_name}_{ret_name}_AP40_{diff_name}_official_cpp"
                    for cls_name in current_classes
                ]
                vals.append(float(np.mean([ret[k] for k in keys])))
                ret[f"KITTI/Overall_{ret_name}_AP40_{diff_name}_official_cpp"] = vals[-1]
            lines.append(f"{metric:<4} AP40:{vals[0]:.4f}, {vals[1]:.4f}, {vals[2]:.4f}")
        lines.append("")
    return "\n".join(lines), ret


def _run_official_cpp_eval(
    args: argparse.Namespace,
    output_dir: Path,
    sample_ids: Sequence[int],
    gt_annos: Sequence[Dict[str, np.ndarray]],
    dt_annos: Sequence[Dict[str, np.ndarray]],
    eval_types: Sequence[str],
) -> Tuple[str, Dict[str, float], Dict[str, str]]:
    binary, eval_meta = _get_official_eval_binary(args, output_dir)
    work_root, result_dir, result_sha = _prepare_official_eval_tree(
        output_dir=output_dir,
        sample_ids=sample_ids,
        gt_annos=gt_annos,
        dt_annos=dt_annos,
        num_images=args.official_eval_num_images,
    )
    proc = subprocess.run(
        [str(binary), result_sha, "offline", ""],
        cwd=str(work_root),
        capture_output=True,
        text=True,
    )
    log_path = work_root / "official_cpp_eval.log"
    log_path.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"KITTI official C++ evaluator failed; see {log_path}.")
    result, metrics = _collect_official_cpp_metrics(
        result_dir=result_dir,
        current_classes=KITTI_OBJECT_CLASS_NAMES,
        eval_types=eval_types,
    )
    info = {
        "backend": "official_cpp",
        "work_dir": str(work_root),
        "result_dir": str(result_dir),
        "eval_log": str(log_path),
        **eval_meta,
    }
    return result, metrics, info


def _build_eval_dataset(args: argparse.Namespace) -> KittiObject5FrameDetDataset:
    return KittiObject5FrameDetDataset(
        root=args.kitti_det_root,
        split="val",
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        output_resolution=(args.width, args.height),
        max_points_per_sweep=args.max_points_per_sweep,
        grid_config=make_kitti_object_det_grid_config(tuple(args.det_pc_range)),
    )


def _build_eval_loader(args: argparse.Namespace, dataset: KittiObject5FrameDetDataset, device: torch.device):
    if not bool(getattr(args, "distributed", False)):
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            collate_fn=collate_kitti_object_det,
        )
        return loader, len(dataset)

    rank = misc.get_rank()
    world_size = misc.get_world_size()
    indices = list(range(rank, len(dataset), world_size))
    shard = Subset(dataset, indices)
    loader = DataLoader(
        shard,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_kitti_object_det,
    )
    return loader, len(indices)


def _gather_eval_states(local_state: Dict, args: argparse.Namespace):
    if not bool(getattr(args, "distributed", False)):
        return [local_state]
    world_size = misc.get_world_size()
    if hasattr(dist, "gather_object"):
        gathered = [None for _ in range(world_size)] if misc.is_main_process() else None
        dist.gather_object(local_state, object_gather_list=gathered, dst=0)
        return gathered if misc.is_main_process() else None
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_state)
    return gathered if misc.is_main_process() else None


@torch.no_grad()
def main() -> None:
    args = get_args_parser().parse_args()
    misc.init_distributed_mode(args)
    ckpt_path = _resolve_ckpt_path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _fill_args_from_checkpoint(args, ckpt.get("args", {}))

    if args.output_dir is None:
        args.output_dir = str(ckpt_path.parent / "eval_kitti_object_val")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    toggle_memory_efficient_attention(enabled=False)
    register_legacy_checkpoint_modules()
    cudnn.benchmark = True
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if bool(getattr(args, "distributed", False)):
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    model = _build_model(args, device)
    state = ckpt.get("model", None)
    if state is None:
        raise KeyError(f"Checkpoint {ckpt_path} does not contain key 'model'.")
    status = model.load_state_dict(_strip_module_prefix(state), strict=False)
    if misc.is_main_process():
        print(
            f"[load] {ckpt_path} exp={args.exp} missing={len(status.missing_keys)} "
            f"unexpected={len(status.unexpected_keys)}"
        )
    model.eval()

    dataset = _build_eval_dataset(args)
    loader, rank_samples = _build_eval_loader(args, dataset, device)
    if misc.is_main_process():
        print(
            f"[data] val samples={len(dataset)} rank0_samples={rank_samples} "
            f"world_size={misc.get_world_size()} batch_size={args.batch_size} "
            f"device={device} amp={args.amp} score_thr={args.det_score_threshold}"
        )
        print(
            f"[det] pc_range={tuple(float(v) for v in args.det_pc_range)} "
            f"depth_bound={tuple(float(v) for v in args.det_depth_bound)}"
        )

    amp_dtype = torch.bfloat16 if args.amp == "bf16" else (torch.float16 if args.amp == "fp16" else None)
    gt_annos: List[Dict] = []
    dt_annos: List[Dict] = []
    seen = set()
    sample_ids: List[int] = []
    t0 = time.time()
    for step, batch in enumerate(loader):
        if int(args.max_batches) > 0 and step >= int(args.max_batches):
            break
        ctx = (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if amp_dtype is not None and device.type == "cuda"
            else torch.autocast(device_type=device.type, enabled=False)
        )
        with ctx:
            out = _model_forward(model, batch, device, args)
            decoded = model.det_decode(out["det_preds"])
        for i, pred in enumerate(decoded):
            sample_id = int(batch["sample_id"][i])
            if sample_id in seen:
                continue
            seen.add(sample_id)
            sample_ids.append(sample_id)
            gt_annos.append(_load_gt_anno(dataset, sample_id))
            dt_annos.append(
                _pred_to_kitti_anno(
                    dataset,
                    sample_id,
                    pred["boxes_3d"],
                    pred["scores_3d"],
                    pred["labels_3d"],
                )
            )
        if (step + 1) % int(args.print_freq) == 0:
            print(f"[eval][rank={misc.get_rank()}] batches={step + 1}/{len(loader)} samples={len(gt_annos)}")

    local_state = {
        "gt_annos": gt_annos,
        "dt_annos": dt_annos,
        "sample_ids": sample_ids,
        "num_batches": int(step + 1 if "step" in locals() else 0),
    }
    gathered_states = _gather_eval_states(local_state, args)
    if bool(getattr(args, "distributed", False)):
        dist.barrier()
    if not misc.is_main_process():
        return

    assert gathered_states is not None
    records = []
    seen = set()
    n_batches = 0
    for state in gathered_states:
        n_batches += int(state.get("num_batches", 0))
        for sample_id, gt_anno, dt_anno in zip(
            state.get("sample_ids", []),
            state.get("gt_annos", []),
            state.get("dt_annos", []),
        ):
            sid = int(sample_id)
            if sid in seen:
                continue
            seen.add(sid)
            records.append((sid, gt_anno, dt_anno))
    records.sort(key=lambda item: item[0])
    sample_ids = [sid for sid, _gt, _dt in records]
    gt_annos = [gt_anno for _sid, gt_anno, _dt in records]
    dt_annos = [dt_anno for _sid, _gt, dt_anno in records]

    eval_types = ("bbox", "bev", "3d")
    if args.eval_backend == "official_cpp":
        result, metrics, eval_info = _run_official_cpp_eval(
            args=args,
            output_dir=output_dir,
            sample_ids=sample_ids,
            gt_annos=gt_annos,
            dt_annos=dt_annos,
            eval_types=eval_types,
        )
    else:
        result, metrics = kitti_object_eval(
            gt_annos,
            dt_annos,
            current_classes=KITTI_OBJECT_CLASS_NAMES,
            eval_types=eval_types,
        )
        eval_info = {"backend": "python"}
    print(result)
    metrics_payload = {
        "checkpoint": str(ckpt_path),
        "exp": args.exp,
        "eval_backend": args.eval_backend,
        "eval_info": eval_info,
        "kitti_det_root": args.kitti_det_root,
        "det_pc_range": [float(v) for v in args.det_pc_range],
        "det_depth_bound": [float(v) for v in args.det_depth_bound],
        "num_samples": len(gt_annos),
        "elapsed_sec": time.time() - t0,
        "world_size": int(misc.get_world_size()),
        "num_batches": int(n_batches),
        "metrics": metrics,
    }
    (output_dir / "kitti_eval.txt").write_text(result + "\n", encoding="utf-8")
    metrics_path = Path(args.output_json) if args.output_json else (output_dir / "metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(metrics_payload), f, indent=2, sort_keys=True)
        f.write("\n")
    if args.save_annos:
        with (output_dir / "kitti_annos.pkl").open("wb") as f:
            pickle.dump({"sample_ids": sample_ids, "gt_annos": gt_annos, "dt_annos": dt_annos}, f)
    print(f"[output] wrote {output_dir}")


if __name__ == "__main__":
    main()
