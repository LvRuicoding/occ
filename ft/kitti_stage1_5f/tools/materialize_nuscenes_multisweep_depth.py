"""Materialize multi-sweep projected LiDAR depth for processed nuScenes frames.

The script reads raw nuScenes tables plus ``annotations.json`` and writes
sidecar depth files under each processed scene, e.g.
``data/nuscenes_processed/train_scene-0001/dense_depth/000000_0.npy``.
The output depth has the same resolution and intrinsics as the already
processed image npz.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import os.path as osp
import tempfile
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is a convenience only.
    tqdm = None


CAMERA_TO_IDX = {
    "CAM_FRONT": 0,
    "CAM_FRONT_RIGHT": 1,
    "CAM_FRONT_LEFT": 2,
    "CAM_BACK": 3,
    "CAM_BACK_LEFT": 4,
    "CAM_BACK_RIGHT": 5,
}

_WORKER: Dict[str, Any] = {}


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

        ordered: List[str] = []
        seen = set()
        while token and token in samples and token not in seen:
            ordered.append(token)
            seen.add(token)
            nxt = samples[token].get("next", "")
            if nxt in ("", "EOF"):
                break
            token = nxt
        out[scene_name] = ordered
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


def _frame_ids_for_scene(scene_dir: str, camera_idx: int) -> List[int]:
    suffix = f"_{camera_idx}.npz"
    frame_ids: List[int] = []
    for name in sorted(os.listdir(scene_dir)):
        if not name.endswith(suffix):
            continue
        stem = name[: -len(suffix)]
        if stem.isdigit():
            frame_ids.append(int(stem))
    return frame_ids


def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _load_lidar_tables(raw_root: str, version: str) -> Dict[str, Any]:
    table_root = osp.join(raw_root, version)
    sensors = {row["token"]: row for row in _load_json(osp.join(table_root, "sensor.json"))}
    calibrated = {
        row["token"]: row for row in _load_json(osp.join(table_root, "calibrated_sensor.json"))
    }
    ego_poses = {row["token"]: row for row in _load_json(osp.join(table_root, "ego_pose.json"))}

    lidar_by_token: Dict[str, Dict[str, Any]] = {}
    lidar_keyframe_by_sample: Dict[str, Dict[str, Any]] = {}
    for row in _load_json(osp.join(table_root, "sample_data.json")):
        calib = calibrated.get(row.get("calibrated_sensor_token", ""))
        if calib is None:
            continue
        sensor = sensors.get(calib.get("sensor_token", ""))
        if sensor is None or sensor.get("channel") != "LIDAR_TOP":
            continue
        lidar_by_token[row["token"]] = row
        if bool(row.get("is_key_frame", False)):
            lidar_keyframe_by_sample[row["sample_token"]] = row

    return {
        "calibrated": calibrated,
        "ego_poses": ego_poses,
        "lidar_by_token": lidar_by_token,
        "lidar_keyframe_by_sample": lidar_keyframe_by_sample,
    }


def _collect_jobs(args, scene_tokens: Dict[str, List[str]]) -> List[Tuple[str, str, int, str, int, str]]:
    jobs: List[Tuple[str, str, int, str, int, str]] = []
    processed_scenes = list(_iter_processed_scenes(args.processed_root, args.splits, args.scenes))
    if args.limit_scenes > 0:
        processed_scenes = processed_scenes[: args.limit_scenes]

    for processed_scene in processed_scenes:
        raw_scene = _raw_scene_name(processed_scene)
        tokens = scene_tokens.get(raw_scene, [])
        scene_dir = osp.join(args.processed_root, processed_scene)
        for camera_name in args.cameras:
            camera_idx = CAMERA_TO_IDX[camera_name]
            frame_ids = _frame_ids_for_scene(scene_dir, camera_idx)
            if args.limit_frames > 0:
                frame_ids = frame_ids[: args.limit_frames]
            for frame_id in frame_ids:
                if frame_id < 0 or frame_id >= len(tokens):
                    continue
                jobs.append(
                    (
                        processed_scene,
                        raw_scene,
                        frame_id,
                        camera_name,
                        camera_idx,
                        tokens[frame_id],
                    )
                )
    return jobs


def _sweep_chain(sample_token: str, num_sweeps: int) -> List[Dict[str, Any]]:
    lidar_keyframe_by_sample = _WORKER["lidar_keyframe_by_sample"]
    lidar_by_token = _WORKER["lidar_by_token"]
    first = lidar_keyframe_by_sample.get(sample_token)
    if first is None:
        raise KeyError(f"LIDAR_TOP keyframe missing for sample_token={sample_token}")

    sweeps: List[Dict[str, Any]] = []
    row = first
    while row is not None and len(sweeps) < num_sweeps:
        sweeps.append(row)
        prev_token = row.get("prev", "")
        if not prev_token:
            break
        row = lidar_by_token.get(prev_token)
    return sweeps


def _load_lidar_xyz(raw_root: str, filename: str, strict: bool) -> np.ndarray:
    path = osp.join(raw_root, filename)
    if not osp.isfile(path):
        if strict:
            raise FileNotFoundError(path)
        return np.zeros((0, 3), dtype=np.float64)
    points = np.fromfile(path, dtype=np.float32)
    if points.size % 5 != 0:
        raise ValueError(f"Bad nuScenes lidar file {path}: float_count={points.size}")
    return points.reshape(-1, 5)[:, :3].astype(np.float64, copy=False)


def _filter_lidar_distance(points: np.ndarray, min_distance: float) -> np.ndarray:
    if min_distance <= 0.0 or points.size == 0:
        return points
    keep = (np.abs(points[:, 0]) >= min_distance) | (np.abs(points[:, 1]) >= min_distance)
    return points[keep]


def _project_multisweep_depth(
    sample_token: str,
    camera_info: Dict[str, Any],
    intrinsics: np.ndarray,
    image_hw: Tuple[int, int],
) -> Tuple[np.ndarray, int]:
    args = _WORKER["args"]
    raw_root = args.raw_root
    calibrated = _WORKER["calibrated"]
    ego_poses = _WORKER["ego_poses"]
    sweeps = _sweep_chain(sample_token, int(args.num_sweeps))

    height, width = image_hw
    depth_flat = np.full(height * width, np.inf, dtype=np.float32)
    T_global_from_ego_cam = _transform_from_record(camera_info["ego_pose"])
    T_ego_from_cam = _transform_from_record(camera_info["extrinsic"])
    T_cam_from_global = np.linalg.inv(T_global_from_ego_cam @ T_ego_from_cam)
    K = intrinsics.astype(np.float64, copy=False)

    used_sweeps = 0
    for sweep in sweeps:
        points = _load_lidar_xyz(raw_root, sweep["filename"], strict=bool(args.strict_lidar))
        points = _filter_lidar_distance(points, float(args.min_lidar_distance))
        if points.size == 0:
            continue

        lidar_calib = calibrated[sweep["calibrated_sensor_token"]]
        lidar_pose = ego_poses[sweep["ego_pose_token"]]
        T_ego_from_lidar = _transform_from_record(lidar_calib)
        T_global_from_ego_lidar = _transform_from_record(lidar_pose)
        T_cam_from_lidar = T_cam_from_global @ T_global_from_ego_lidar @ T_ego_from_lidar

        points_h = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        points_cam = (T_cam_from_lidar @ points_h.T).T[:, :3]
        valid = points_cam[:, 2] > float(args.min_depth)
        points_cam = points_cam[valid]
        if points_cam.size == 0:
            used_sweeps += 1
            continue

        uvw = (K @ points_cam.T).T
        uv = uvw[:, :2] / uvw[:, 2:3]
        xy = np.rint(uv).astype(np.int64)
        depths = points_cam[:, 2].astype(np.float32, copy=False)
        valid = (
            (xy[:, 0] >= 0)
            & (xy[:, 0] < width)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < height)
        )
        if bool(valid.any()):
            flat_idx = xy[valid, 1] * width + xy[valid, 0]
            np.minimum.at(depth_flat, flat_idx, depths[valid])
        used_sweeps += 1

    depth_flat[~np.isfinite(depth_flat)] = 0.0
    return depth_flat.reshape(height, width), used_sweeps


def _output_depth_path(args, processed_scene: str, frame_id: int, camera_idx: int) -> str:
    ext = "npy" if args.output_format == "npy" else "npz"
    return osp.join(
        args.processed_root,
        processed_scene,
        args.output_subdir,
        f"{frame_id:06d}_{camera_idx}.{ext}",
    )


def _write_depth(path: str, depth: np.ndarray, args) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{osp.basename(path)}.", suffix=f".tmp.{args.output_format}", dir=osp.dirname(path)
    )
    fd_open = True
    try:
        with os.fdopen(fd, "wb") as f:
            fd_open = False
            if args.output_format == "npy":
                np.save(f, depth.astype(np.float32, copy=False))
            elif bool(args.no_compress):
                np.savez(f, **{args.depth_key: depth.astype(np.float32, copy=False)})
            else:
                np.savez_compressed(f, **{args.depth_key: depth.astype(np.float32, copy=False)})
        os.replace(tmp_path, path)
    finally:
        if fd_open:
            os.close(fd)
        if osp.exists(tmp_path):
            os.unlink(tmp_path)


def _read_camera_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        if "image" not in z.files or "intrinsics" not in z.files:
            raise KeyError("missing image/intrinsics")
        image_shape = np.asarray(z["image"].shape[:2], dtype=np.int64)
        intrinsics = np.asarray(z["intrinsics"], dtype=np.float64)
    return image_shape, intrinsics


def _existing_output_matches(path: str, args) -> bool:
    if not osp.isfile(path):
        return False
    if args.output_format == "npy":
        return True
    try:
        with np.load(path) as z:
            return args.depth_key in z.files
    except Exception:
        return False


def _process_job(job: Tuple[str, str, int, str, int, str]) -> Tuple[str, str]:
    processed_scene, raw_scene, frame_id, camera_name, camera_idx, sample_token = job
    args = _WORKER["args"]
    scene_infos = _WORKER["scene_infos"]
    path = osp.join(args.processed_root, processed_scene, f"{frame_id:06d}_{camera_idx}.npz")
    out_path = _output_depth_path(args, processed_scene, frame_id, camera_idx)
    label = f"{processed_scene}/{frame_id:06d}_{camera_idx}"

    if not osp.isfile(path):
        return "missing_processed", label
    if _existing_output_matches(out_path, args) and not bool(args.overwrite):
        return "skipped_existing", label
    frame_info = scene_infos.get(raw_scene, {}).get(sample_token)
    if frame_info is None:
        return "missing_annotation", label
    camera_info = frame_info.get("camera_sensor", {}).get(camera_name)
    if camera_info is None:
        return "missing_annotation", label

    try:
        image_shape, intrinsics = _read_camera_npz(path)
        height, width = int(image_shape[0]), int(image_shape[1])
        depth, used_sweeps = _project_multisweep_depth(
            sample_token=sample_token,
            camera_info=camera_info,
            intrinsics=intrinsics,
            image_hw=(height, width),
        )
        if bool(args.dry_run):
            valid = int((depth > 0).sum())
            return "would_update", f"{label}: used_sweeps={used_sweeps} valid_pixels={valid}"

        _write_depth(out_path, depth, args)
        return "updated", label
    except Exception as exc:
        return "error", f"{label}: {exc}"


def _init_worker(args, scene_infos, lidar_tables) -> None:
    _WORKER["args"] = args
    _WORKER["scene_infos"] = scene_infos
    _WORKER.update(lidar_tables)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write multi-sweep projected LIDAR_TOP depth sidecars for processed "
            "nuScenes camera npz files."
        )
    )
    parser.add_argument(
        "--processed-root",
        default="/home/dataset-local/lr/code/OccAny/data/nuscenes_processed",
    )
    parser.add_argument(
        "--raw-root",
        default="/home/dataset-local/lr/code/OccAny/raw_data/nuscenes",
    )
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--cameras", nargs="+", default=["CAM_FRONT"], choices=sorted(CAMERA_TO_IDX))
    parser.add_argument("--num-sweeps", type=int, default=10)
    parser.add_argument("--output-subdir", default="dense_depth")
    parser.add_argument("--output-format", choices=("npy", "npz"), default="npy")
    parser.add_argument("--depth-key", default="dense_depthmap")
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--min-lidar-distance", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--limit-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict-lidar", action="store_true")
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument("--log-errors", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.num_sweeps) <= 0:
        raise ValueError("--num-sweeps must be positive")
    annotations_path = args.annotations or osp.join(args.raw_root, "annotations.json")
    annotations = _load_json(annotations_path)
    scene_infos = annotations["scene_infos"]
    scene_tokens = _build_scene_sample_tokens(scene_infos)
    lidar_tables = _load_lidar_tables(args.raw_root, args.version)
    jobs = _collect_jobs(args, scene_tokens)
    print(
        f"jobs={len(jobs)} cameras={args.cameras} num_sweeps={args.num_sweeps} "
        f"output_subdir={args.output_subdir} output_format={args.output_format}"
    )

    counters: Dict[str, int] = {
        "updated": 0,
        "would_update": 0,
        "skipped_existing": 0,
        "missing_processed": 0,
        "missing_annotation": 0,
        "error": 0,
    }
    error_messages: List[str] = []

    if int(args.workers) <= 1:
        _init_worker(args, scene_infos, lidar_tables)
        iterator = map(_process_job, jobs)
    else:
        pool = mp.Pool(
            processes=int(args.workers),
            initializer=_init_worker,
            initargs=(args, scene_infos, lidar_tables),
        )
        iterator = pool.imap_unordered(_process_job, jobs, chunksize=8)

    progress = tqdm(iterator, total=len(jobs), dynamic_ncols=True) if tqdm is not None else iterator
    try:
        for status, message in progress:
            counters[status] = counters.get(status, 0) + 1
            if status == "error" and len(error_messages) < int(args.log_errors):
                error_messages.append(message)
    finally:
        if int(args.workers) > 1:
            pool.close()
            pool.join()

    if error_messages:
        print("sample errors:")
        for message in error_messages:
            print(f"  {message}")
    action = "dry-run" if bool(args.dry_run) else "done"
    print(f"{action}: counters={counters}")


if __name__ == "__main__":
    main()
