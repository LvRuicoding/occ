"""DDAD Stage-1 depth-only adapter in the KITTI 5-frame batch format."""
from __future__ import annotations

from .. import _paths  # noqa: F401

import json
import math
import os
import os.path as osp
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from occany.utils.helpers import crop_resize_if_necessary
from occany.utils.image_util import ImgNorm

from .kitti_stage1_lidar_dense_depth import (
    collate_stage1_dense_depth,
    collate_stage1_lidar_dense_depth,
)
from .unified_occ import KITTI_GRID_CONFIG


DDAD_RAW_SPLIT_KEYS = {
    "train": "0",
    "val": "1",
}


def _quat_to_matrix(q: Dict[str, float]) -> np.ndarray:
    qw = float(q["qw"])
    qx = float(q["qx"])
    qy = float(q["qy"])
    qz = float(q["qz"])
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 0.0:
        raise ValueError(f"Invalid zero-norm quaternion: {q}")
    qw, qx, qy, qz = (qw / norm, qx / norm, qy / norm, qz / norm)
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _pose_to_matrix(pose: Dict[str, Any]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_to_matrix(pose["rotation"])
    t = pose["translation"]
    T[:3, 3] = [float(t["x"]), float(t["y"]), float(t["z"])]
    return T


def _scene_sort_key(name: str) -> Tuple[int, str]:
    try:
        return int(name.rsplit("_", 1)[1]), name
    except Exception:
        return 10**9, name


class DDAD5FrameStage1DenseDepthDataset(Dataset):
    """DDAD front-camera depth dataset using Stage-1 KITTI batch keys.

    The processed DDAD tree is expected to contain:

      processed_root/{train,val}_*/<frame>_<cam_idx>.npz
      processed_root/{train,val}_*/point_cloud/LIDAR/<frame>.npz

    Only cam_idx=0 is used by the KITTI+DDAD depth-pair experiments.
    """

    dense_depth_key = "depthmap"

    def __init__(
        self,
        processed_root: str,
        raw_root: str | None = None,
        split: str = "train",
        num_frames: int = 5,
        frame_stride: int = 4,
        output_resolution: Tuple[int, int] = (512, 160),
        cam_idx: int = 0,
        max_points_per_sweep: int = 0,
        require_lidar: bool = False,
    ) -> None:
        super().__init__()
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.processed_root = processed_root
        self.raw_root = raw_root
        self.split = split
        self.num_frames = int(num_frames)
        self.frame_stride = int(frame_stride)
        self.output_resolution = (int(output_resolution[0]), int(output_resolution[1]))
        self.cam_idx = int(cam_idx)
        self.max_points_per_sweep = int(max_points_per_sweep)
        self.require_lidar = bool(require_lidar)
        self.ignore_label = 255
        self.grid_config = KITTI_GRID_CONFIG

        if self.cam_idx != 0:
            raise ValueError("DDAD Stage-1 adapter currently supports cam_idx=0 only.")
        if self.require_lidar and not self.raw_root:
            raise ValueError("--ddad_raw_root is required for DDAD LiDAR depth experiments.")

        self._raw_splits: Dict[str, List[str]] | None = None
        self._lidar_pose_cache: Dict[str, List[np.ndarray]] = {}
        self.samples: List[Tuple[str, int]] = []
        self._index_samples()

    def _scene_dirs(self) -> List[str]:
        prefix = f"{self.split}_"
        if not osp.isdir(self.processed_root):
            raise FileNotFoundError(f"Missing DDAD processed root: {self.processed_root}")
        return sorted(
            (
                name
                for name in os.listdir(self.processed_root)
                if name.startswith(prefix) and osp.isdir(osp.join(self.processed_root, name))
            ),
            key=_scene_sort_key,
        )

    def _scene_dir(self, scene: str) -> str:
        return osp.join(self.processed_root, scene)

    def _frame_npz(self, scene: str, frame: int) -> str:
        return osp.join(self._scene_dir(scene), f"{frame:06d}_{self.cam_idx}.npz")

    def _lidar_npz(self, scene: str, frame: int) -> str:
        return osp.join(self._scene_dir(scene), "point_cloud", "LIDAR", f"{frame:06d}.npz")

    def _load_raw_splits(self) -> Dict[str, List[str]]:
        if self._raw_splits is not None:
            return self._raw_splits
        if not self.raw_root:
            raise ValueError("--ddad_raw_root is required to read DDAD scene poses.")
        path = osp.join(self.raw_root, "ddad.json")
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        scene_splits = meta.get("scene_splits", {})
        out: Dict[str, List[str]] = {}
        for split, raw_key in DDAD_RAW_SPLIT_KEYS.items():
            split_meta = scene_splits.get(raw_key, {})
            filenames = split_meta.get("filenames", [])
            out[split] = [str(v) for v in filenames]
        self._raw_splits = out
        return out

    def _raw_scene_json(self, scene: str) -> str:
        split, idx_text = scene.rsplit("_", 1)
        idx = int(idx_text)
        filenames = self._load_raw_splits().get(split, [])
        if idx < 0 or idx >= len(filenames):
            raise IndexError(
                f"DDAD raw split {split!r} has {len(filenames)} scenes; "
                f"cannot map processed scene {scene!r}."
            )
        return osp.join(str(self.raw_root), filenames[idx])

    def _lidar_poses_world_from_lidar(self, scene: str) -> List[np.ndarray]:
        if scene in self._lidar_pose_cache:
            return self._lidar_pose_cache[scene]

        path = self._raw_scene_json(scene)
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        data_by_key = {item["key"]: item for item in raw.get("data", [])}
        poses: List[np.ndarray] = []
        for sample in raw.get("samples", []):
            lidar_item = None
            for key in sample.get("datum_keys", []):
                item = data_by_key.get(key)
                if item is not None and item.get("id", {}).get("name") == "LIDAR":
                    lidar_item = item
                    break
            if lidar_item is None:
                raise RuntimeError(f"Missing LIDAR datum in DDAD raw sample for scene {scene}.")
            datum = lidar_item.get("datum", {}).get("point_cloud", {})
            pose = datum.get("pose")
            if pose is None:
                raise RuntimeError(f"Missing LIDAR pose in DDAD raw sample for scene {scene}.")
            poses.append(_pose_to_matrix(pose).astype(np.float32))

        self._lidar_pose_cache[scene] = poses
        return poses

    def _index_samples(self) -> None:
        history = (self.num_frames - 1) * self.frame_stride
        for scene in self._scene_dirs():
            scene_dir = self._scene_dir(scene)
            frames = []
            suffix = f"_{self.cam_idx}.npz"
            for name in os.listdir(scene_dir):
                if name.endswith(suffix):
                    frames.append(int(name[: -len(suffix)]))
            frame_set = set(frames)
            for t in sorted(frames):
                if t - history < 0:
                    continue
                ok = True
                for k in range(self.num_frames):
                    fid = t - k * self.frame_stride
                    if fid not in frame_set:
                        ok = False
                        break
                    if self.require_lidar and not osp.isfile(self._lidar_npz(scene, fid)):
                        ok = False
                        break
                if ok:
                    self.samples.append((scene, t))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frame(
        self,
        scene: str,
        frame: int,
        timestep_index: int,
    ) -> Tuple[Dict[str, Any], torch.Tensor, bool]:
        with np.load(self._frame_npz(scene, frame)) as npz:
            image = np.asarray(npz["image"])
            intrinsics = np.asarray(npz["intrinsics"], dtype=np.float64)
            cam2world = np.asarray(npz["cam2world"], dtype=np.float64)
            depth = np.asarray(npz[self.dense_depth_key], dtype=np.float32)

        img_pil = Image.fromarray(image)
        img_pil_out, depth_out, intr_out = crop_resize_if_necessary(
            img_pil,
            depth,
            intrinsics,
            self.output_resolution,
        )
        img_arr = np.asarray(img_pil_out)
        H, W = img_arr.shape[:2]
        depth_out = np.asarray(depth_out, dtype=np.float32)
        valid = np.isfinite(depth_out) & (depth_out > 0.0)
        has_depth = bool(valid.any())
        if has_depth:
            depth_out = np.where(valid, depth_out, 0.0).astype(np.float32)
        else:
            depth_out = np.zeros_like(depth_out, dtype=np.float32)

        view = {
            "img": ImgNorm(img_arr),
            "true_shape": np.int32((H, W)),
            "camera_pose": np.eye(4, dtype=np.float32),
            "camera_intrinsics": intr_out.astype(np.float32),
            "cam2world": cam2world.astype(np.float32),
            "timestep": int(timestep_index),
            "is_raymap": False,
            "is_metric_scale": True,
            "frame_id": int(frame),
            "label": f"{scene}_{frame:06d}_cam{self.cam_idx}",
        }
        return view, torch.from_numpy(depth_out), has_depth

    def _load_points(self, scene: str, frame: int) -> torch.Tensor:
        with np.load(self._lidar_npz(scene, frame)) as npz:
            key = "data" if "data" in npz.files else npz.files[0]
            pts = np.asarray(npz[key], dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 4:
            raise RuntimeError(f"DDAD point cloud must be (P,4+), got {pts.shape}.")
        pts = pts[:, :4]
        if self.max_points_per_sweep > 0 and pts.shape[0] > self.max_points_per_sweep:
            idx = np.linspace(0, pts.shape[0] - 1, self.max_points_per_sweep).astype(np.int64)
            pts = pts[idx]
        return torch.from_numpy(np.ascontiguousarray(pts, dtype=np.float32))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        scene, t = self.samples[index]
        frame_ids = [t - k * self.frame_stride for k in range(self.num_frames)]

        views: List[Dict[str, Any]] = []
        dense_depths: List[torch.Tensor] = []
        frame_mask: List[bool] = []
        for k, fid in enumerate(frame_ids):
            view, depth, has_depth = self._load_frame(scene, int(fid), k)
            views.append(view)
            dense_depths.append(depth)
            frame_mask.append(bool(has_depth))

        grid = self.grid_config.as_tensors()
        data: Dict[str, Any] = {
            "views": views,
            "voxel_label": torch.zeros((1, 1, 1), dtype=torch.long),
            "T_target_from_refcam": torch.eye(4, dtype=torch.float32),
            "voxel_origin": grid["voxel_origin"],
            "voxel_size": grid["voxel_size"],
            "grid_size": grid["grid_size"],
            "half_voxel_origin": grid["half_voxel_origin"],
            "half_voxel_size": grid["half_voxel_size"],
            "half_grid_size": grid["half_grid_size"],
            "fusion_vox_origin": grid["fusion_vox_origin"],
            "fusion_vox_size": grid["fusion_vox_size"],
            "fusion_vox_grid": grid["fusion_vox_grid"],
            "dense_depth": torch.stack(dense_depths, dim=0),
            "dense_depth_frame_mask": torch.tensor(frame_mask, dtype=torch.bool),
            "dataset_name": "ddad",
            "sequence": scene,
            "target_frame_id": int(t),
            "frame_ids": tuple(int(f) for f in frame_ids),
        }
        return data


class DDAD5FrameStage1LidarDenseDepthDataset(DDAD5FrameStage1DenseDepthDataset):
    """DDAD depth sample plus raw LiDAR sweeps and per-frame camera transforms."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["require_lidar"] = True
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = super().__getitem__(index)
        scene = data["sequence"]
        frame_ids = data["frame_ids"]
        poses = self._lidar_poses_world_from_lidar(scene)

        T_cam_from_lidar: List[np.ndarray] = []
        K_per_frame: List[np.ndarray] = []
        points_per_frame: List[torch.Tensor] = []
        for fid, view in zip(frame_ids, data["views"]):
            fid_int = int(fid)
            if fid_int >= len(poses):
                raise IndexError(
                    f"DDAD raw scene {scene} has {len(poses)} lidar poses; "
                    f"cannot read frame {fid_int}."
                )
            cam2world = np.asarray(view["cam2world"], dtype=np.float64)
            T_world_from_lidar = np.asarray(poses[fid_int], dtype=np.float64)
            T_cam_from_lidar.append(
                (np.linalg.inv(cam2world) @ T_world_from_lidar).astype(np.float32)
            )
            K_per_frame.append(np.asarray(view["camera_intrinsics"], dtype=np.float32))
            points_per_frame.append(self._load_points(scene, fid_int))

        image_hw = np.asarray(data["views"][0]["true_shape"], dtype=np.int32).reshape(2)
        data["points_per_frame"] = points_per_frame
        data["T_cam_from_velo"] = torch.from_numpy(np.stack(T_cam_from_lidar, axis=0))
        data["K_per_frame"] = torch.from_numpy(np.stack(K_per_frame, axis=0))
        data["image_hw"] = torch.from_numpy(image_hw)
        return data


__all__ = [
    "DDAD5FrameStage1DenseDepthDataset",
    "DDAD5FrameStage1LidarDenseDepthDataset",
    "collate_stage1_dense_depth",
    "collate_stage1_lidar_dense_depth",
]
