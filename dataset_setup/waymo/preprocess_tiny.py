#!/usr/bin/env python3
import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import os.path as osp
import random

import numpy as np
from PIL import Image
from tqdm import tqdm


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))


def check_waymo_deps_available():
    missing = [
        module
        for module in ("tensorflow", "waymo_open_dataset")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise ImportError(
            "Waymo tiny preprocessing requires tensorflow and waymo_open_dataset. "
            f"Missing: {', '.join(missing)}."
        )


def import_waymo_deps():
    try:
        import tensorflow.compat.v1 as tf
        from waymo_open_dataset import dataset_pb2 as open_dataset
        from waymo_open_dataset.utils import frame_utils
    except ImportError as exc:
        raise ImportError(
            "Waymo tiny preprocessing requires tensorflow and waymo_open_dataset. "
            "Install the same Waymo Open Dataset dependencies used by "
            "dataset_setup/waymo/preprocess_waymo.py before running conversion."
        ) from exc

    if not tf.executing_eagerly():
        tf.enable_eager_execution()
    return tf, open_dataset, frame_utils


def inv(mat):
    return np.linalg.inv(mat)


def make_intrinsics(intrinsic):
    f1, f2, cx, cy = intrinsic[:4]
    return np.asarray([[f1, 0.0, cx], [0.0, f2, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def resize_image_and_intrinsics(image, intrinsics, target_long_edge):
    width, height = image.size
    if target_long_edge <= 0:
        return image, intrinsics.astype(np.float64)

    long_edge = max(width, height)
    scale = float(target_long_edge) / float(long_edge)
    if abs(scale - 1.0) < 1e-8:
        return image, intrinsics.astype(np.float64)

    new_size = (int(round(width * scale)), int(round(height * scale)))
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    intrinsics = intrinsics.astype(np.float64).copy()
    intrinsics[0, :] *= scale
    intrinsics[1, :] *= scale
    return image, intrinsics


def transform_points(transform, points):
    return points @ transform[:3, :3].T + transform[:3, 3]


def points_to_depthmap(pixels, depths, width, height):
    if pixels.size == 0:
        return np.zeros((height, width), dtype=np.float32)

    xy = np.rint(pixels).astype(np.int32)
    valid = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
        & (depths > 0)
    )
    xy = xy[valid]
    depths = depths[valid]
    if xy.size == 0:
        return np.zeros((height, width), dtype=np.float32)

    depth_flat = np.full(height * width, np.inf, dtype=np.float32)
    flat_indices = xy[:, 1] * width + xy[:, 0]
    np.minimum.at(depth_flat, flat_indices, depths.astype(np.float32))
    depth_flat[~np.isfinite(depth_flat)] = 0.0
    return depth_flat.reshape(height, width)


def list_tfrecords(root):
    if not osp.isdir(root):
        return []
    return sorted(
        name
        for name in os.listdir(root)
        if name.endswith(".tfrecord") and osp.isfile(osp.join(root, name))
    )


def select_segments(waymo_root, val_count, val_seed, max_train_segments, max_val_segments):
    train_segments = list_tfrecords(osp.join(waymo_root, "train"))
    val_segments_all = list_tfrecords(osp.join(waymo_root, "val"))

    if max_train_segments is not None:
        train_segments = train_segments[:max_train_segments]

    if val_count is not None and val_count < len(val_segments_all):
        rng = random.Random(val_seed)
        val_segments = sorted(rng.sample(val_segments_all, val_count))
    else:
        val_segments = val_segments_all

    if max_val_segments is not None:
        val_segments = val_segments[:max_val_segments]

    return train_segments, val_segments, val_segments_all


def scene_name(split, segment):
    return f"{split}_{segment}"


def save_metadata(output_root, train_segments, val_segments, val_segments_all, val_count, val_seed):
    metadata_dir = osp.join(output_root, ".metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    metadata = {
        "dataset": "waymo_tiny",
        "format": "occany_kitti_like_npz",
        "has_occ_labels": False,
        "frame_id_format": "000000",
        "camera_ids": list(range(5)),
        "train_segments": train_segments,
        "val_segments": val_segments,
        "val_available_segments": val_segments_all,
        "val_count": val_count,
        "val_seed": val_seed,
        "splits": {
            "train": [scene_name("train", segment) for segment in train_segments],
            "val": [scene_name("val", segment) for segment in val_segments],
        },
    }
    with open(osp.join(metadata_dir, "splits.json"), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def process_segment(task):
    split, segment, args = task
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    tf, open_dataset, frame_utils = import_waymo_deps()

    input_path = osp.join(args.waymo_root, split, segment)
    output_scene = scene_name(split, segment)
    output_dir = osp.join(args.output_root, output_scene)
    os.makedirs(output_dir, exist_ok=True)

    dataset = tf.data.TFRecordDataset(input_path, compression_type="")
    axes_transform = np.asarray(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    camera_calib = None
    processed_frames = 0
    for frame_idx, data in enumerate(dataset):
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break

        frame = open_dataset.Frame()
        frame.ParseFromString(data.numpy())

        if camera_calib is None:
            camera_calib = {}
            for cam in frame.context.camera_calibrations:
                camera_calib[cam.name] = {
                    "intrinsics": make_intrinsics(list(cam.intrinsic)),
                    "cam_to_vehicle": np.asarray(cam.extrinsic.transform, dtype=np.float64).reshape(4, 4),
                }

        range_images, camera_projections, _, range_image_top_pose = (
            frame_utils.parse_range_image_and_camera_projection(frame)
        )
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
        )
        points_all = np.concatenate(points, axis=0)
        cp_points_all = np.concatenate(cp_points, axis=0).astype(np.int32)

        for image_proto in frame.images:
            camera_name = image_proto.name
            if camera_name not in camera_calib:
                continue

            camera_idx = camera_name - 1
            if camera_idx < 0 or camera_idx >= 5:
                continue

            output_path = osp.join(output_dir, f"{frame_idx:06d}_{camera_idx}.npz")
            if args.skip_existing and osp.exists(output_path):
                continue

            calib = camera_calib[camera_name]
            image = Image.fromarray(tf.image.decode_jpeg(image_proto.image).numpy()).convert("RGB")
            intrinsics = calib["intrinsics"]
            image, intrinsics2 = resize_image_and_intrinsics(
                image,
                intrinsics,
                args.target_long_edge,
            )
            width, height = image.size

            mask = cp_points_all[:, 0] == camera_name
            pixels = cp_points_all[mask][:, 1:3].astype(np.float64)
            pts3d_vehicle = points_all[mask].astype(np.float64)
            pts3d_camera = transform_points(axes_transform @ inv(calib["cam_to_vehicle"]), pts3d_vehicle)

            if args.target_long_edge > 0:
                scale_x = intrinsics2[0, 0] / intrinsics[0, 0]
                scale_y = intrinsics2[1, 1] / intrinsics[1, 1]
                pixels = pixels.copy()
                pixels[:, 0] *= scale_x
                pixels[:, 1] *= scale_y

            depthmap = points_to_depthmap(pixels, pts3d_camera[:, 2], width, height)
            car_to_world = np.asarray(image_proto.pose.transform, dtype=np.float64).reshape(4, 4)
            cam2world = car_to_world @ calib["cam_to_vehicle"] @ inv(axes_transform)

            np.savez_compressed(
                output_path,
                image=np.asarray(image, dtype=np.uint8),
                depthmap=depthmap.astype(np.float32),
                intrinsics=intrinsics2,
                cam2world=cam2world,
            )
        processed_frames += 1

    return output_scene, processed_frames


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess raw_data/waymo_tiny into KITTI-like OccAny npz trajectories."
    )
    parser.add_argument(
        "--waymo-root",
        default=osp.join(REPO_ROOT, "raw_data", "waymo_tiny"),
        help="Root containing train/ and val/ Waymo TFRecord files.",
    )
    parser.add_argument(
        "--output-root",
        default=osp.join(REPO_ROOT, "data", "waymo_tiny_processed"),
        help="Output root for KITTI-like trajectory folders.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of segment worker processes.")
    parser.add_argument("--val-count", type=int, default=10, help="Number of val segments to sample.")
    parser.add_argument("--val-seed", type=int, default=42, help="Seed for fixed random val sampling.")
    parser.add_argument(
        "--target-long-edge",
        type=int,
        default=1024,
        help="Resize images so their long edge is this many pixels. Use 0 to keep original size.",
    )
    parser.add_argument("--max-train-segments", type=int, default=None, help="Debug limit for train.")
    parser.add_argument("--max-val-segments", type=int, default=None, help="Debug limit for val.")
    parser.add_argument("--max-frames", type=int, default=None, help="Debug limit for frames per segment.")
    parser.add_argument("--skip-existing", action="store_true", help="Do not rewrite existing npz files.")
    return parser.parse_args()


def main():
    args = parse_args()
    check_waymo_deps_available()
    os.makedirs(args.output_root, exist_ok=True)

    train_segments, val_segments, val_segments_all = select_segments(
        args.waymo_root,
        args.val_count,
        args.val_seed,
        args.max_train_segments,
        args.max_val_segments,
    )
    save_metadata(
        args.output_root,
        train_segments,
        val_segments,
        val_segments_all,
        args.val_count,
        args.val_seed,
    )

    tasks = [("train", segment, args) for segment in train_segments]
    tasks.extend(("val", segment, args) for segment in val_segments)
    if not tasks:
        raise RuntimeError(f"No .tfrecord files found under {args.waymo_root}/train or /val")

    total_frames = 0
    if args.workers == 1:
        results = map(process_segment, tasks)
        iterator = tqdm(results, total=len(tasks), desc="Preprocessing Waymo tiny segments")
        for _, processed_frames in iterator:
            total_frames += processed_frames
    else:
        with mp.Pool(args.workers) as pool:
            results = pool.imap_unordered(process_segment, tasks)
            iterator = tqdm(results, total=len(tasks), desc="Preprocessing Waymo tiny segments")
            for _, processed_frames in iterator:
                total_frames += processed_frames

    print(
        f"Processed {len(tasks)} segments / {total_frames} frames into {args.output_root}"
    )
    print("No KITTI-style occupancy labels were generated.")


if __name__ == "__main__":
    main()
