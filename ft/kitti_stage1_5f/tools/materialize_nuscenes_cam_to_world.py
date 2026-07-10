"""Materialize nuScenes camera-to-world matrices into processed frame npz files."""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import tempfile
import zipfile
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)


def _quat_wxyz_to_rot(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def _transform_from_record(record: Dict[str, Any]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_wxyz_to_rot(record["rotation"])
    T[:3, 3] = np.asarray(record["translation"], dtype=np.float64)
    return T


def _raw_scene_name(processed_scene: str) -> str:
    if processed_scene.startswith("train_"):
        return processed_scene[len("train_") :]
    if processed_scene.startswith("val_"):
        return processed_scene[len("val_") :]
    return processed_scene


def _build_scene_sample_tokens(scene_infos: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for scene_name, samples in scene_infos.items():
        token = None
        for sample_token, info in samples.items():
            if info.get("prev") == "EOF" or not info.get("prev"):
                token = sample_token
                break
        if token is None and samples:
            token = next(iter(samples.keys()))

        tokens: List[str] = []
        while token and token in samples:
            tokens.append(token)
            nxt = samples[token].get("next", "")
            if nxt in ("", "EOF"):
                break
            token = nxt
        out[scene_name] = tokens
    return out


def _iter_processed_scenes(
    processed_root: str,
    splits: Sequence[str],
    scenes: Sequence[str] | None,
) -> Iterable[str]:
    wanted = set(scenes or [])
    for name in sorted(os.listdir(processed_root)):
        path = osp.join(processed_root, name)
        if not osp.isdir(path):
            continue
        if wanted and name not in wanted and _raw_scene_name(name) not in wanted:
            continue
        if splits and not any(name.startswith(f"{split}_scene-") for split in splits):
            continue
        yield name


def _frame_ids_for_scene(scene_dir: str) -> List[int]:
    frame_ids = []
    for name in sorted(os.listdir(scene_dir)):
        if not name.endswith("_0.npz"):
            continue
        stem = name[:-len("_0.npz")]
        if stem.isdigit():
            frame_ids.append(int(stem))
    return frame_ids


def _cam_to_world(
    scene_infos: Dict[str, Dict[str, Any]],
    scene_tokens: Dict[str, List[str]],
    raw_scene: str,
    frame_id: int,
    camera_name: str,
) -> np.ndarray:
    tokens = scene_tokens.get(raw_scene, [])
    if frame_id < 0 or frame_id >= len(tokens):
        raise KeyError(f"{raw_scene}/{frame_id:06d} is not present in annotations.json")
    info = scene_infos[raw_scene][tokens[frame_id]]
    cam = info["camera_sensor"][camera_name]
    T_ego_from_cam = _transform_from_record(cam["extrinsic"])
    T_world_from_ego = _transform_from_record(info["ego_pose"])
    return T_world_from_ego @ T_ego_from_cam


def _npz_uses_compression(path: str) -> bool:
    with zipfile.ZipFile(path, "r") as zf:
        return any(info.compress_type != zipfile.ZIP_STORED for info in zf.infolist())


def _write_npz(path: str, data: Dict[str, np.ndarray], compressed: bool) -> None:
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{osp.basename(path)}.", suffix=".tmp.npz", dir=osp.dirname(path)
    )
    os.close(fd)
    try:
        if compressed:
            np.savez_compressed(tmp_path, **data)
        else:
            np.savez(tmp_path, **data)
        os.replace(tmp_path, path)
    finally:
        if osp.exists(tmp_path):
            os.unlink(tmp_path)


def _update_npz(
    path: str,
    T_world_from_cam: np.ndarray,
    compression: str,
    dry_run: bool,
    skip_existing: bool,
) -> str:
    with np.load(path) as z:
        data = {key: z[key] for key in z.files}

    existing_cam2world = data.get("cam2world")
    existing_cam_to_world = data.get("cam_to_world")
    if skip_existing and existing_cam2world is not None and existing_cam_to_world is not None:
        return "skipped_existing"

    same_cam2world = (
        existing_cam2world is not None
        and existing_cam2world.shape == T_world_from_cam.shape
        and np.allclose(existing_cam2world, T_world_from_cam, rtol=1e-6, atol=1e-6)
    )
    same_cam_to_world = (
        existing_cam_to_world is not None
        and existing_cam_to_world.shape == T_world_from_cam.shape
        and np.allclose(existing_cam_to_world, T_world_from_cam, rtol=1e-6, atol=1e-6)
    )
    if same_cam2world and same_cam_to_world:
        return "unchanged"

    data["cam2world"] = T_world_from_cam.astype(np.float64)
    data["cam_to_world"] = T_world_from_cam.astype(np.float64)
    if dry_run:
        return "would_update"

    if compression == "preserve":
        compressed = _npz_uses_compression(path)
    else:
        compressed = compression == "compressed"
    _write_npz(path, data, compressed)
    return "updated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write cam2world and cam_to_world from raw nuScenes annotations into "
            "data/nuscenes_processed/*.npz files."
        )
    )
    parser.add_argument("--processed-root", default="data/nuscenes_processed")
    parser.add_argument("--raw-root", default="raw_data/nuscenes")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Optional scene filter, e.g. train_scene-0001 or scene-0001.",
    )
    parser.add_argument(
        "--all-cameras",
        action="store_true",
        help="Update camera ids 0-5. By default only camera id 0 / CAM_FRONT is updated.",
    )
    parser.add_argument(
        "--camera-indices",
        nargs="+",
        type=int,
        default=None,
        help="Explicit camera ids to update. Overrides --all-cameras.",
    )
    parser.add_argument(
        "--compression",
        choices=("preserve", "compressed", "stored"),
        default="preserve",
        help="How to write npz files after updating.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--limit-frames", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations_path = osp.join(args.raw_root, "annotations.json")
    with open(annotations_path, "r") as f:
        annotations = json.load(f)
    scene_infos = annotations["scene_infos"]
    scene_tokens = _build_scene_sample_tokens(scene_infos)

    if args.camera_indices is not None:
        camera_indices = tuple(args.camera_indices)
    elif args.all_cameras:
        camera_indices = tuple(range(len(CAMERA_ORDER)))
    else:
        camera_indices = (0,)
    for cam_idx in camera_indices:
        if cam_idx < 0 or cam_idx >= len(CAMERA_ORDER):
            raise ValueError(f"camera index must be in [0, 5], got {cam_idx}")

    counters: Dict[str, int] = {
        "updated": 0,
        "would_update": 0,
        "unchanged": 0,
        "skipped_existing": 0,
        "missing_npz": 0,
        "missing_annotation": 0,
        "errors": 0,
    }
    scene_count = 0
    seen_files = 0
    for scene in _iter_processed_scenes(args.processed_root, args.splits, args.scenes):
        scene_count += 1
        if args.limit_scenes > 0 and scene_count > args.limit_scenes:
            break
        raw_scene = _raw_scene_name(scene)
        scene_dir = osp.join(args.processed_root, scene)
        frame_ids = _frame_ids_for_scene(scene_dir)
        if args.limit_frames > 0:
            frame_ids = frame_ids[: args.limit_frames]
        for frame_id in frame_ids:
            for cam_idx in camera_indices:
                npz_path = osp.join(scene_dir, f"{frame_id:06d}_{cam_idx}.npz")
                if not osp.isfile(npz_path):
                    counters["missing_npz"] += 1
                    continue
                camera_name = CAMERA_ORDER[cam_idx]
                try:
                    T_world_from_cam = _cam_to_world(
                        scene_infos, scene_tokens, raw_scene, frame_id, camera_name
                    )
                except Exception:
                    counters["missing_annotation"] += 1
                    continue
                try:
                    status = _update_npz(
                        npz_path,
                        T_world_from_cam,
                        compression=args.compression,
                        dry_run=args.dry_run,
                        skip_existing=args.skip_existing,
                    )
                    counters[status] += 1
                    seen_files += 1
                except Exception as exc:
                    counters["errors"] += 1
                    print(f"[error] {npz_path}: {exc}")

                if args.log_every > 0 and seen_files > 0 and seen_files % args.log_every == 0:
                    print(f"processed {seen_files} files: {counters}")

    action = "dry-run" if args.dry_run else "done"
    print(f"{action}: scenes={scene_count}, camera_indices={camera_indices}, counters={counters}")


if __name__ == "__main__":
    main()
