import argparse
import json
import multiprocessing as mp
import os
import os.path as osp
import pickle
import sys
import tempfile

import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm


CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


CAMERA_TO_IDX = {
    "CAM_FRONT": 0,
    "CAM_FRONT_RIGHT": 1,
    "CAM_FRONT_LEFT": 2,
    "CAM_BACK": 3,
    "CAM_BACK_LEFT": 4,
    "CAM_BACK_RIGHT": 5,
}

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset_setup.base_make_seq import SeqMaker  # noqa: E402


_WORKER_CONTEXT = {}


def make_transform(translation, rotation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Quaternion(rotation).rotation_matrix
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def resize_image_and_intrinsics(image, intrinsics, target_long_edge):
    width, height = image.size
    if target_long_edge <= 0:
        return image, intrinsics.astype(np.float64)

    long_edge = max(width, height)
    scale = float(target_long_edge) / float(long_edge)
    if abs(scale - 1.0) < 1e-8:
        return image, intrinsics.astype(np.float64)

    new_size = (int(round(width * scale)), int(round(height * scale)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)

    intrinsics = intrinsics.astype(np.float64).copy()
    intrinsics[0, :] *= scale
    intrinsics[1, :] *= scale
    return resized, intrinsics


def load_lidar_points(nuscenes_root, lidar_filename):
    lidar_path = osp.join(nuscenes_root, lidar_filename)
    if not osp.exists(lidar_path):
        return np.zeros((0, 3), dtype=np.float32)

    points = np.fromfile(lidar_path, dtype=np.float32)
    if points.size % 5 != 0:
        raise ValueError(f"Bad nuScenes lidar point count in {lidar_path}: {points.size}")
    return points.reshape(-1, 5)[:, :3].astype(np.float64)


def project_lidar_to_depth(
    lidar_points,
    lidar_info,
    calibrated_sensors,
    ego_poses,
    cam_info,
    intrinsics,
    width,
    height,
):
    if lidar_points.size == 0:
        return np.zeros((height, width), dtype=np.float32)

    lidar_calib = calibrated_sensors[lidar_info["calibrated_sensor_token"]]
    lidar_ego_pose = ego_poses[lidar_info["ego_pose_token"]]
    lidar_to_ego = make_transform(lidar_calib["translation"], lidar_calib["rotation"])
    ego_to_global = make_transform(lidar_ego_pose["translation"], lidar_ego_pose["rotation"])
    global_to_cam_ego = np.linalg.inv(make_transform(
        cam_info["ego_pose"]["translation"],
        cam_info["ego_pose"]["rotation"],
    ))
    ego_to_cam = np.linalg.inv(make_transform(
        cam_info["extrinsic"]["translation"],
        cam_info["extrinsic"]["rotation"],
    ))
    lidar_to_cam = ego_to_cam @ global_to_cam_ego @ ego_to_global @ lidar_to_ego

    points_h = np.concatenate(
        [lidar_points, np.ones((lidar_points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    points_cam = (lidar_to_cam @ points_h.T).T[:, :3]
    valid = points_cam[:, 2] > 0.1
    points_cam = points_cam[valid]
    if points_cam.size == 0:
        return np.zeros((height, width), dtype=np.float32)

    points_2d_h = (intrinsics @ points_cam.T).T
    uv = points_2d_h[:, :2] / points_2d_h[:, 2:3]
    xy = np.rint(uv).astype(np.int32)
    depths = points_cam[:, 2]

    valid = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )
    xy = xy[valid]
    depths = depths[valid]

    depth_flat = np.full(height * width, np.inf, dtype=np.float32)
    flat_indices = xy[:, 1] * width + xy[:, 0]
    np.minimum.at(depth_flat, flat_indices, depths.astype(np.float32))
    depth_flat[~np.isfinite(depth_flat)] = 0.0
    depthmap = depth_flat.reshape(height, width)
    return depthmap


def build_scene_frames(scene_info):
    first_token = None
    for token, frame_info in scene_info.items():
        if frame_info.get("prev") in (None, "", "EOF"):
            first_token = token
            break

    if first_token is None:
        return list(scene_info.keys())

    ordered = []
    seen = set()
    token = first_token
    while token and token != "EOF" and token not in seen and token in scene_info:
        ordered.append(token)
        seen.add(token)
        token = scene_info[token].get("next")

    if len(ordered) != len(scene_info):
        remaining = [token for token in scene_info.keys() if token not in seen]
        ordered.extend(remaining)
    return ordered


def load_nuscenes_tables(nuscenes_root, cache_path=None):
    if cache_path is not None and osp.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    table_root = osp.join(nuscenes_root, "v1.0-trainval")

    def load_table(name):
        path = osp.join(table_root, f"{name}.json")
        if not osp.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r") as f:
            return json.load(f)

    sensors = {row["token"]: row for row in load_table("sensor")}
    sample_data = load_table("sample_data")
    calibrated_sensors = {row["token"]: row for row in load_table("calibrated_sensor")}
    ego_poses = {row["token"]: row for row in load_table("ego_pose")}

    lidar_by_sample = {}
    for row in sample_data:
        sensor = sensors[calibrated_sensors[row["calibrated_sensor_token"]]["sensor_token"]]
        if sensor["channel"] == "LIDAR_TOP" and row["is_key_frame"]:
            lidar_by_sample[row["sample_token"]] = row

    tables = (lidar_by_sample, calibrated_sensors, ego_poses)
    if cache_path is not None:
        os.makedirs(osp.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(tables, f, protocol=pickle.HIGHEST_PROTOCOL)
    return tables


def preprocess_scene(
    scene_name,
    output_scene_name,
    scene_info,
    args,
    lidar_by_sample,
    calibrated_sensors,
    ego_poses,
):
    scene_dir = osp.join(args.output_root, output_scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    frame_tokens = build_scene_frames(scene_info)
    if args.max_frames is not None:
        frame_tokens = frame_tokens[: args.max_frames]

    save_npz = np.savez if args.no_compress else np.savez_compressed
    processed = 0
    for frame_idx, frame_token in enumerate(frame_tokens):
        frame_info = scene_info[frame_token]
        camera_jobs = []

        for camera_name in args.cameras:
            cam_info = frame_info["camera_sensor"].get(camera_name)
            if cam_info is None:
                continue

            frame_id = f"{frame_idx:06d}_{CAMERA_TO_IDX[camera_name]}"
            output_path = osp.join(scene_dir, f"{frame_id}.npz")
            if args.skip_existing and osp.exists(output_path):
                continue
            camera_jobs.append((camera_name, cam_info, output_path))

        if not camera_jobs:
            processed += 1
            continue

        lidar_info = lidar_by_sample.get(frame_token)
        if lidar_info is None:
            raise FileNotFoundError(
                f"No LIDAR_TOP sample_data for sample token {frame_token} in scene {scene_name}"
            )
        lidar_points = load_lidar_points(args.nuscenes_root, lidar_info["filename"])

        for _, cam_info, output_path in camera_jobs:

            image_path = osp.join(args.nuscenes_root, "imgs", cam_info["img_path"])
            if not osp.exists(image_path):
                raise FileNotFoundError(image_path)

            image = Image.open(image_path).convert("RGB")
            intrinsics = np.asarray(cam_info["intrinsics"], dtype=np.float64)
            image, intrinsics = resize_image_and_intrinsics(
                image, intrinsics, args.target_long_edge
            )
            width, height = image.size

            depthmap = project_lidar_to_depth(
                lidar_points,
                lidar_info,
                calibrated_sensors,
                ego_poses,
                cam_info,
                intrinsics,
                width,
                height,
            )
            cam_to_world = make_transform(
                cam_info["ego_pose"]["translation"],
                cam_info["ego_pose"]["rotation"],
            ) @ make_transform(
                cam_info["extrinsic"]["translation"],
                cam_info["extrinsic"]["rotation"],
            )

            save_npz(
                output_path,
                image=np.asarray(image, dtype=np.uint8),
                depthmap=depthmap.astype(np.float32),
                intrinsics=intrinsics,
                cam2world=cam_to_world,
            )
        processed += 1

    return processed


def _init_worker(args, lidar_by_sample, calibrated_sensors, ego_poses, scene_infos):
    _WORKER_CONTEXT["args"] = args
    _WORKER_CONTEXT["lidar_by_sample"] = lidar_by_sample
    _WORKER_CONTEXT["calibrated_sensors"] = calibrated_sensors
    _WORKER_CONTEXT["ego_poses"] = ego_poses
    _WORKER_CONTEXT["scene_infos"] = scene_infos


def _preprocess_scene_item(scene_item):
    split_name, scene_name = scene_item
    args = _WORKER_CONTEXT["args"]
    return preprocess_scene(
        scene_name,
        f"{split_name}_{scene_name}",
        _WORKER_CONTEXT["scene_infos"][scene_name],
        args,
        _WORKER_CONTEXT["lidar_by_sample"],
        _WORKER_CONTEXT["calibrated_sensors"],
        _WORKER_CONTEXT["ego_poses"],
    )


def make_seq_files(output_root, modes, subsampling_rate, max_stride):
    old_override = os.environ.get("OCCANY_SEQ_ROOT")
    os.environ["OCCANY_SEQ_ROOT"] = output_root
    try:
        if "surround" in modes:
            SeqMaker(
                preprocessed_dir="nuscenes_processed",
                cameras=list(range(6)),
                img_track_pattern="*_{camera_id}",
                frame_id_format=":06d",
                file_ext=".npz",
                suffix="_all",
                seq_mode="surround",
            ).run()

        if "temporal" in modes:
            SeqMaker(
                preprocessed_dir="nuscenes_processed",
                cameras=list(range(6)),
                img_track_pattern="*_{camera_id}",
                frame_id_format=":06d",
                file_ext=".npz",
                suffix="_all",
                subsampling_rate=subsampling_rate,
                max_stride=max_stride,
                seq_mode="temporal",
            ).run()
    finally:
        if old_override is None:
            os.environ.pop("OCCANY_SEQ_ROOT", None)
        else:
            os.environ["OCCANY_SEQ_ROOT"] = old_override


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess nuScenes/Occ3D files into OccAny multiview npz format."
    )
    parser.add_argument(
        "--nuscenes-root",
        default=osp.join(REPO_ROOT, "raw_data", "nuscenes"),
        help="Root containing annotations.json, imgs/, samples/, and v1.0-trainval/.",
    )
    parser.add_argument(
        "--output-root",
        default=osp.join(REPO_ROOT, "data", "nuscenes_processed"),
        help="Output root for processed scene directories.",
    )
    parser.add_argument(
        "--target-long-edge",
        type=int,
        default=1024,
        help="Resize images so their long edge is this many pixels. Use 0 to keep original size.",
    )
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--pid", type=int, default=0, help="Shard id for distributed preprocessing.")
    parser.add_argument("--nproc", type=int, default=1, help="Number of preprocessing shards.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of local worker processes. Parallelizes by scene.",
    )
    parser.add_argument("--max-scenes", type=int, default=None, help="Limit scenes for debugging.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames per scene for debugging.")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=CAMERA_NAMES,
        choices=CAMERA_NAMES,
        help="Camera names to export.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not rewrite existing per-camera npz files.",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Use uncompressed npz files. Faster, but uses more disk space.",
    )
    parser.add_argument(
        "--cache-tables",
        action="store_true",
        help="Cache filtered nuScenes metadata tables outside the processed data tree.",
    )
    parser.add_argument(
        "--cache-path",
        default=None,
        help="Optional explicit path for --cache-tables metadata cache.",
    )
    parser.add_argument(
        "--make-seqs",
        action="store_true",
        help="Also generate seq pkl files after writing trajectory folders.",
    )
    parser.add_argument(
        "--seq-modes",
        nargs="+",
        choices=["surround", "temporal"],
        default=["surround", "temporal"],
        help="Sequence pkl files to generate after preprocessing.",
    )
    parser.add_argument(
        "--seq-subsampling-rate",
        type=int,
        default=5,
        help="Temporal seq subsampling rate.",
    )
    parser.add_argument(
        "--seq-max-stride",
        type=int,
        default=9,
        help="Temporal seq maximum stride.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    anno_path = osp.join(args.nuscenes_root, "annotations.json")
    if not osp.exists(anno_path):
        raise FileNotFoundError(anno_path)

    with open(anno_path, "r") as f:
        annotations = json.load(f)

    scene_items = []
    if args.split in ("train", "all"):
        scene_items.extend(("train", scene) for scene in annotations.get("train_split", []))
    if args.split in ("val", "all"):
        scene_items.extend(("val", scene) for scene in annotations.get("val_split", []))
    if not scene_items:
        scene_items = [("unknown", scene) for scene in sorted(annotations["scene_infos"].keys())]
    scene_items = [
        (split_name, scene)
        for split_name, scene in scene_items
        if scene in annotations["scene_infos"]
    ]
    scene_items = scene_items[args.pid :: args.nproc]
    if args.max_scenes is not None:
        scene_items = scene_items[: args.max_scenes]

    cache_path = None
    if args.cache_tables:
        cache_path = args.cache_path or osp.join(
            tempfile.gettempdir(),
            "occany_nuscenes_lidar_tables.pkl",
        )
    lidar_by_sample, calibrated_sensors, ego_poses = load_nuscenes_tables(
        args.nuscenes_root,
        cache_path=cache_path,
    )

    if args.workers == 1:
        total_frames = 0
        for split_name, scene_name in tqdm(scene_items, desc="Preprocessing nuScenes scenes"):
            output_scene_name = f"{split_name}_{scene_name}"
            total_frames += preprocess_scene(
                scene_name,
                output_scene_name,
                annotations["scene_infos"][scene_name],
                args,
                lidar_by_sample,
                calibrated_sensors,
                ego_poses,
            )
    else:
        with mp.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(
                args,
                lidar_by_sample,
                calibrated_sensors,
                ego_poses,
                annotations["scene_infos"],
            ),
        ) as pool:
            total_frames = 0
            results = pool.imap_unordered(_preprocess_scene_item, scene_items)
            for processed_frames in tqdm(
                results,
                total=len(scene_items),
                desc="Preprocessing nuScenes scenes",
            ):
                total_frames += processed_frames

    print(
        f"Processed {len(scene_items)} scenes / {total_frames} frames into {args.output_root}"
    )

    if args.make_seqs:
        make_seq_files(
            output_root=args.output_root,
            modes=args.seq_modes,
            subsampling_rate=args.seq_subsampling_rate,
            max_stride=args.seq_max_stride,
        )


if __name__ == "__main__":
    main()
