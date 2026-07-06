#!/usr/bin/env python3
"""Visualize DDAD processed camera frames with projected LIDAR points.

The script checks frame alignment after moving/renaming DDAD LIDAR files into
data/ddad_processed. It projects each processed frame's LIDAR point cloud into
the six processed camera images using the original DDAD scene JSON pose and the
processed camera intrinsics/cam2world matrices.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


CAMERA_NAMES = {
    0: "CAMERA_01",
    1: "CAMERA_05",
    2: "CAMERA_06",
    3: "CAMERA_07",
    4: "CAMERA_08",
    5: "CAMERA_09",
}

DEFAULT_SAMPLES = ("train_0:000000", "train_0:000010", "train_0:000020", "val_0:000000")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Project processed DDAD LIDAR frames onto processed camera images."
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=root / "data" / "ddad_processed",
        help="Path to data/ddad_processed.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=root / "raw_data" / "OpenDataLab___DDAD" / "raw" / "ddad_train_val",
        help="Path containing ddad.json and raw scene_*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "visuals" / "ddad_lidar_alignment",
        help="Directory for rendered PNGs and summary.csv.",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=list(DEFAULT_SAMPLES),
        help="Samples as scene:frame, e.g. train_0:000000 val_0:000010.",
    )
    parser.add_argument(
        "--max-display-points",
        type=int,
        default=25000,
        help="Maximum projected points drawn per camera. Stats still use all projected points.",
    )
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=120.0)
    parser.add_argument("--point-size", type=float, default=0.35)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument(
        "--only-depthmap-agree",
        action="store_true",
        help="Draw only projected points that agree with the processed depthmap.",
    )
    parser.add_argument(
        "--depth-hit-abs-tol",
        type=float,
        default=0.5,
        help="Absolute depth tolerance, in meters, for --only-depthmap-agree.",
    )
    parser.add_argument(
        "--depth-hit-rel-tol",
        type=float,
        default=0.05,
        help="Relative depth tolerance for --only-depthmap-agree.",
    )
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def parse_sample_token(token: str) -> tuple[str, int, str]:
    if ":" not in token:
        raise ValueError(f"Bad sample token {token!r}; expected scene:frame")
    scene, frame = token.split(":", 1)
    frame_idx = int(frame)
    return scene, frame_idx, f"{frame_idx:06d}"


def split_key_and_index(scene: str) -> tuple[str, int]:
    if scene.startswith("train_"):
        return "0", int(scene.split("_", 1)[1])
    if scene.startswith("val_"):
        return "1", int(scene.split("_", 1)[1])
    raise ValueError(f"Unsupported DDAD processed scene name: {scene}")


def load_scene_json(raw_root: Path, scene: str) -> tuple[dict, str, Path]:
    ddad_index_path = raw_root / "ddad.json"
    with ddad_index_path.open() as f:
        ddad_index = json.load(f)

    split_key, scene_idx = split_key_and_index(scene)
    filenames = ddad_index["scene_splits"][split_key]["filenames"]
    rel_scene_json = filenames[scene_idx]
    scene_json_path = raw_root / rel_scene_json
    with scene_json_path.open() as f:
        scene_data = json.load(f)
    raw_scene_dir = rel_scene_json.split("/", 1)[0]
    return scene_data, raw_scene_dir, scene_json_path


def quat_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Zero-length quaternion")
    qw, qx, qy, qz = q / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def pose_to_matrix(pose: dict) -> np.ndarray:
    rotation = pose["rotation"]
    translation = pose["translation"]
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = quat_to_rotation_matrix(
        rotation["qw"], rotation["qx"], rotation["qy"], rotation["qz"]
    )
    mat[:3, 3] = [translation["x"], translation["y"], translation["z"]]
    return mat


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    return (matrix @ points_h.T).T[:, :3]


def get_lidar_record(scene_data: dict, frame_idx: int) -> tuple[dict, str]:
    samples = scene_data["samples"]
    if frame_idx >= len(samples):
        raise IndexError(f"Frame {frame_idx:06d} out of range; scene has {len(samples)} samples")

    sample = samples[frame_idx]
    by_key = {entry["key"]: entry for entry in scene_data["data"]}
    point_cloud_records = []
    for key in sample["datum_keys"]:
        datum = by_key[key]["datum"]
        if "point_cloud" in datum:
            point_cloud_records.append(datum["point_cloud"])

    if len(point_cloud_records) != 1:
        raise RuntimeError(
            f"Expected exactly one point cloud for frame {frame_idx:06d}; "
            f"found {len(point_cloud_records)}"
        )
    timestamp = sample.get("id", {}).get("timestamp", "")
    return point_cloud_records[0], timestamp


def project_points(
    points_world: np.ndarray,
    camera_npz: dict,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = camera_npz["image"]
    intrinsics = camera_npz["intrinsics"].astype(np.float64)
    world_to_camera = np.linalg.inv(camera_npz["cam2world"].astype(np.float64))

    camera_points = transform_points(world_to_camera, points_world)
    z = camera_points[:, 2]
    depth_mask = (z > min_depth) & (z < max_depth)
    camera_points = camera_points[depth_mask]
    z = z[depth_mask]

    if camera_points.size == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    u = intrinsics[0, 0] * camera_points[:, 0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * camera_points[:, 1] / z + intrinsics[1, 2]
    height, width = image.shape[:2]
    image_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[image_mask], v[image_mask], z[image_mask]


def depthmap_stats(depthmap: np.ndarray, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> tuple[int, float, float]:
    if u.size == 0:
        return 0, float("nan"), float("nan")

    height, width = depthmap.shape
    ui = np.rint(u).astype(np.int64).clip(0, width - 1)
    vi = np.rint(v).astype(np.int64).clip(0, height - 1)
    depth_values = depthmap[vi, ui]
    hit_mask = depth_values > 0
    hits = int(hit_mask.sum())
    if hits == 0:
        return hits, float("nan"), float("nan")

    abs_diff = np.abs(depth_values[hit_mask] - z[hit_mask])
    rel_diff = abs_diff / np.maximum(depth_values[hit_mask], 1e-6)
    return hits, float(np.median(abs_diff)), float(np.median(rel_diff))


def depthmap_agreement_mask(
    depthmap: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    abs_tol: float,
    rel_tol: float,
) -> np.ndarray:
    if u.size == 0:
        return np.zeros(0, dtype=bool)

    height, width = depthmap.shape
    ui = np.rint(u).astype(np.int64).clip(0, width - 1)
    vi = np.rint(v).astype(np.int64).clip(0, height - 1)
    depth_values = depthmap[vi, ui]
    hit_mask = depth_values > 0
    abs_diff = np.abs(depth_values - z)
    rel_diff = abs_diff / np.maximum(depth_values, 1e-6)
    return hit_mask & (abs_diff <= abs_tol) & (rel_diff <= rel_tol)


def choose_display_indices(count: int, max_display: int, rng: np.random.Generator) -> np.ndarray:
    if max_display <= 0 or count <= max_display:
        return np.arange(count)
    return np.sort(rng.choice(count, size=max_display, replace=False))


def render_sample(
    processed_root: Path,
    raw_root: Path,
    output_dir: Path,
    sample_token: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[dict]:
    scene, frame_idx, frame_id = parse_sample_token(sample_token)
    scene_dir = processed_root / scene
    scene_data, raw_scene_dir, scene_json_path = load_scene_json(raw_root, scene)
    lidar_record, sample_timestamp = get_lidar_record(scene_data, frame_idx)

    lidar_path = scene_dir / "point_cloud" / "LIDAR" / f"{frame_id}.npz"
    if not lidar_path.is_file():
        raise FileNotFoundError(lidar_path)

    lidar_points = np.load(lidar_path)["data"][:, :3].astype(np.float64)
    finite_mask = np.isfinite(lidar_points).all(axis=1)
    lidar_points = lidar_points[finite_mask]
    lidar_to_world = pose_to_matrix(lidar_record["pose"])
    points_world = transform_points(lidar_to_world, lidar_points)

    fig, axes = plt.subplots(2, 3, figsize=(18, 8), dpi=args.dpi)
    axes = axes.ravel()
    norm = Normalize(vmin=args.min_depth, vmax=args.max_depth)
    scatter = None
    summary_rows = []

    for camera_idx, axis in enumerate(axes):
        camera_path = scene_dir / f"{frame_id}_{camera_idx}.npz"
        if not camera_path.is_file():
            raise FileNotFoundError(camera_path)

        camera_npz = np.load(camera_path)
        image = camera_npz["image"]
        depthmap = camera_npz["depthmap"]
        u, v, z = project_points(points_world, camera_npz, args.min_depth, args.max_depth)
        hits, median_abs_diff, median_rel_diff = depthmap_stats(depthmap, u, v, z)
        agree_mask = depthmap_agreement_mask(
            depthmap,
            u,
            v,
            z,
            args.depth_hit_abs_tol,
            args.depth_hit_rel_tol,
        )
        if args.only_depthmap_agree:
            draw_source_idx = np.flatnonzero(agree_mask)
        else:
            draw_source_idx = np.arange(u.size)
        display_idx = choose_display_indices(draw_source_idx.size, args.max_display_points, rng)
        display_idx = draw_source_idx[display_idx]

        axis.imshow(image)
        if display_idx.size:
            scatter = axis.scatter(
                u[display_idx],
                v[display_idx],
                c=z[display_idx],
                s=args.point_size,
                alpha=args.alpha,
                cmap="turbo",
                norm=norm,
                linewidths=0,
            )

        hit_ratio = hits / u.size if u.size else 0.0
        agree_ratio = int(agree_mask.sum()) / u.size if u.size else 0.0
        rel_text = "nan" if np.isnan(median_rel_diff) else f"{median_rel_diff:.3f}"
        axis.set_title(
            f"{CAMERA_NAMES[camera_idx]}  proj={u.size}  hit={hit_ratio:.1%}  "
            f"agree={agree_ratio:.1%}  rel={rel_text}",
            fontsize=8,
        )
        axis.axis("off")

        summary_rows.append(
            {
                "sample": sample_token,
                "scene": scene,
                "raw_scene_dir": raw_scene_dir,
                "scene_json": str(scene_json_path),
                "frame_idx": frame_id,
                "camera_idx": camera_idx,
                "camera_name": CAMERA_NAMES[camera_idx],
                "camera_file": str(camera_path),
                "lidar_file": str(lidar_path),
                "original_lidar_file": lidar_record["filename"],
                "sample_timestamp": sample_timestamp,
                "projected_points": int(u.size),
                "displayed_points": int(display_idx.size),
                "depthmap_hits": hits,
                "depthmap_hit_ratio": hit_ratio,
                "depthmap_agree": int(agree_mask.sum()),
                "depthmap_agree_ratio": agree_ratio,
                "median_abs_depth_error": median_abs_diff,
                "median_rel_depth_error": median_rel_diff,
            }
        )

    title = (
        f"{scene} frame {frame_id} | raw {raw_scene_dir} | "
        f"renamed LIDAR={lidar_path.name} | original={Path(lidar_record['filename']).name}"
    )
    fig.suptitle(title, fontsize=11)
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=axes.tolist(), shrink=0.72, pad=0.01)
        colorbar.set_label("camera-frame depth (m)")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ddad_lidar_alignment_{scene}_{frame_id}.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    for row in summary_rows:
        row["output_png"] = str(output_path)
    print(f"wrote {output_path}")
    return summary_rows


def write_summary(output_dir: Path, rows: Iterable[dict]) -> Path:
    rows = list(rows)
    summary_path = output_dir / "summary.csv"
    if not rows:
        return summary_path

    fieldnames = list(rows[0].keys())
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    all_rows = []
    for sample_token in args.samples:
        rows = render_sample(
            args.processed_root,
            args.raw_root,
            args.output_dir,
            sample_token,
            args,
            rng,
        )
        all_rows.extend(rows)

    summary_path = write_summary(args.output_dir, all_rows)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
