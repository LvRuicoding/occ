"""Materialize per-frame nuScenes metadata under processed scene directories."""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import tempfile
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


def _json_brace_delta(line: str) -> int:
    delta = 0
    in_string = False
    escaped = False
    for ch in line:
        if escaped:
            escaped = False
            continue
        if in_string and ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            delta += 1
        elif ch == "}":
            delta -= 1
    return delta


def _iter_json_array_objects(path: str):
    obj_lines: List[str] = []
    depth = 0
    in_object = False
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not in_object:
                if not stripped.startswith("{"):
                    continue
                in_object = True
            obj_lines.append(line)
            depth += _json_brace_delta(line)
            if in_object and depth == 0:
                text = "".join(obj_lines).strip()
                if text.endswith(","):
                    text = text[:-1]
                yield json.loads(text)
                obj_lines = []
                in_object = False


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
    frame_ids: List[int] = []
    for name in sorted(os.listdir(scene_dir)):
        if not name.endswith("_0.npz"):
            continue
        stem = name[: -len("_0.npz")]
        if stem.isdigit():
            frame_ids.append(int(stem))
    return frame_ids


def _collect_work(
    processed_root: str,
    splits: Sequence[str],
    scenes: Sequence[str] | None,
    limit_scenes: int,
    limit_frames: int,
) -> List[Tuple[str, str, List[int]]]:
    work: List[Tuple[str, str, List[int]]] = []
    for scene in _iter_processed_scenes(processed_root, splits, scenes):
        if limit_scenes > 0 and len(work) >= limit_scenes:
            break
        scene_dir = osp.join(processed_root, scene)
        frame_ids = _frame_ids_for_scene(scene_dir)
        if limit_frames > 0:
            frame_ids = frame_ids[:limit_frames]
        if frame_ids:
            work.append((scene, _raw_scene_name(scene), frame_ids))
    return work


def _wanted_sample_tokens(
    scene_tokens: Dict[str, List[str]],
    work: Sequence[Tuple[str, str, Sequence[int]]],
) -> set[str]:
    wanted: set[str] = set()
    for _, raw_scene, frame_ids in work:
        tokens = scene_tokens.get(raw_scene, [])
        for frame_id in frame_ids:
            if 0 <= frame_id < len(tokens):
                wanted.add(tokens[frame_id])
    return wanted


def _load_lidar_to_ego_by_sample(raw_root: str, wanted_samples: set[str]) -> Dict[str, np.ndarray]:
    if not wanted_samples:
        return {}
    table_root = osp.join(raw_root, "v1.0-trainval")
    with open(osp.join(table_root, "sensor.json"), "r") as f:
        sensors = {row["token"]: row for row in json.load(f)}
    with open(osp.join(table_root, "calibrated_sensor.json"), "r") as f:
        calibrated_sensors = {row["token"]: row for row in json.load(f)}

    out: Dict[str, np.ndarray] = {}
    sample_data_path = osp.join(table_root, "sample_data.json")
    for row in _iter_json_array_objects(sample_data_path):
        sample_token = row.get("sample_token", "")
        if sample_token not in wanted_samples:
            continue
        if not row.get("is_key_frame", False):
            continue
        calib = calibrated_sensors.get(row.get("calibrated_sensor_token", ""))
        if calib is None:
            continue
        sensor = sensors.get(calib.get("sensor_token", ""))
        if sensor is None or sensor.get("channel") != "LIDAR_TOP":
            continue
        out[sample_token] = _transform_from_record(calib)
        if len(out) >= len(wanted_samples):
            break
    return out


def _frame_meta(
    scene_infos: Dict[str, Dict[str, Any]],
    scene_tokens: Dict[str, List[str]],
    lidar_to_ego_by_sample: Dict[str, np.ndarray],
    raw_scene: str,
    frame_id: int,
    dtype: np.dtype,
) -> Dict[str, np.ndarray]:
    tokens = scene_tokens.get(raw_scene, [])
    if frame_id < 0 or frame_id >= len(tokens):
        raise KeyError(f"{raw_scene}/{frame_id:06d} is not present in annotations.json")
    sample_token = tokens[frame_id]
    info = scene_infos[raw_scene][sample_token]

    world_from_ego = _transform_from_record(info["ego_pose"])
    ego_from_lidar = lidar_to_ego_by_sample.get(sample_token)
    if ego_from_lidar is None:
        raise KeyError(f"LIDAR_TOP calibrated sensor not found for sample_token={sample_token}")
    lidar_from_ego = np.linalg.inv(ego_from_lidar)

    ego_from_cam: List[np.ndarray] = []
    cam_from_ego: List[np.ndarray] = []
    cam_to_world: List[np.ndarray] = []
    camera_intrinsics: List[np.ndarray] = []
    for camera_name in CAMERA_ORDER:
        cam = info["camera_sensor"][camera_name]
        T_ego_from_cam = _transform_from_record(cam["extrinsic"])
        T_cam_from_ego = np.linalg.inv(T_ego_from_cam)
        ego_from_cam.append(T_ego_from_cam)
        cam_from_ego.append(T_cam_from_ego)
        cam_to_world.append(world_from_ego @ T_ego_from_cam)
        camera_intrinsics.append(np.asarray(cam["intrinsics"], dtype=np.float64))

    return {
        "sample_token": np.asarray(sample_token, dtype="S64"),
        "frame_id": np.asarray(frame_id, dtype=np.int32),
        "camera_names": np.asarray(CAMERA_ORDER, dtype="S32"),
        "world_from_ego": world_from_ego.astype(dtype, copy=False),
        "ego_pose": world_from_ego.astype(dtype, copy=False),
        "ego_from_lidar": ego_from_lidar.astype(dtype, copy=False),
        "lidar_to_ego": ego_from_lidar.astype(dtype, copy=False),
        "lidar_from_ego": lidar_from_ego.astype(dtype, copy=False),
        "ego_from_cam": np.stack(ego_from_cam, axis=0).astype(dtype, copy=False),
        "cam_from_ego": np.stack(cam_from_ego, axis=0).astype(dtype, copy=False),
        "cam_to_world": np.stack(cam_to_world, axis=0).astype(dtype, copy=False),
        "camera_intrinsics": np.stack(camera_intrinsics, axis=0).astype(dtype, copy=False),
    }


def _arrays_match(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "SUO" or b.dtype.kind in "SUO":
        return np.array_equal(a, b)
    if np.issubdtype(a.dtype, np.floating) or np.issubdtype(b.dtype, np.floating):
        return np.allclose(a, b, rtol=1e-6, atol=1e-6)
    return np.array_equal(a, b)


def _metadata_matches(path: str, data: Dict[str, np.ndarray]) -> bool:
    if not osp.isfile(path):
        return False
    with np.load(path) as z:
        if set(z.files) != set(data.keys()):
            return False
        for key, value in data.items():
            if not _arrays_match(z[key], value):
                return False
    return True


def _write_npz(path: str, data: Dict[str, np.ndarray], compressed: bool) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write per-frame sample_token, ego pose, camera extrinsics, and "
            "LIDAR_TOP extrinsics into processed nuScenes scene meta npz files."
        )
    )
    parser.add_argument("--processed-root", default="data/nuscenes_processed")
    parser.add_argument("--raw-root", default="raw_data/nuscenes")
    parser.add_argument("--output-subdir", default="meta")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Optional scene filter, e.g. train_scene-0001 or scene-0001.",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--limit-frames", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = np.dtype(args.dtype)
    annotations_path = osp.join(args.raw_root, "annotations.json")
    with open(annotations_path, "r") as f:
        annotations = json.load(f)
    scene_infos = annotations["scene_infos"]
    scene_tokens = _build_scene_sample_tokens(scene_infos)

    work = _collect_work(
        processed_root=args.processed_root,
        splits=args.splits,
        scenes=args.scenes,
        limit_scenes=args.limit_scenes,
        limit_frames=args.limit_frames,
    )
    wanted_samples = _wanted_sample_tokens(scene_tokens, work)
    print(f"scenes={len(work)}, frames={sum(len(x[2]) for x in work)}, wanted_samples={len(wanted_samples)}")
    lidar_to_ego_by_sample = _load_lidar_to_ego_by_sample(args.raw_root, wanted_samples)
    print(f"loaded_lidar_to_ego={len(lidar_to_ego_by_sample)}")

    counters = {
        "updated": 0,
        "would_update": 0,
        "unchanged": 0,
        "skipped_existing": 0,
        "missing_annotation": 0,
        "errors": 0,
    }
    processed = 0
    for scene, raw_scene, frame_ids in work:
        meta_dir = osp.join(args.processed_root, scene, args.output_subdir)
        for frame_id in frame_ids:
            out_path = osp.join(meta_dir, f"{frame_id:06d}.npz")
            if args.skip_existing and osp.isfile(out_path):
                counters["skipped_existing"] += 1
                continue
            try:
                data = _frame_meta(
                    scene_infos=scene_infos,
                    scene_tokens=scene_tokens,
                    lidar_to_ego_by_sample=lidar_to_ego_by_sample,
                    raw_scene=raw_scene,
                    frame_id=frame_id,
                    dtype=dtype,
                )
            except KeyError:
                counters["missing_annotation"] += 1
                continue
            except Exception as exc:
                counters["errors"] += 1
                print(f"[error] {scene}/{frame_id:06d}: {exc}")
                continue

            if _metadata_matches(out_path, data):
                counters["unchanged"] += 1
            elif args.dry_run:
                counters["would_update"] += 1
            else:
                try:
                    _write_npz(out_path, data, compressed=args.compressed)
                    counters["updated"] += 1
                except Exception as exc:
                    counters["errors"] += 1
                    print(f"[error] write {out_path}: {exc}")
                    continue

            processed += 1
            if args.log_every > 0 and processed % args.log_every == 0:
                print(f"processed={processed}, counters={counters}")

    action = "dry-run" if args.dry_run else "done"
    print(f"{action}: processed={processed}, counters={counters}")


if __name__ == "__main__":
    main()
