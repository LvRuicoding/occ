"""Occ3D-nuScenes Stage-1 adapter for shared KITTI/nuScenes fine-tuning."""
from __future__ import annotations

from .. import _paths  # noqa: F401

import os
import os.path as osp
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from occany.utils.helpers import crop_resize_if_necessary
from occany.utils.image_util import ImgNorm

from .kitti_stage1 import collate_stage1
from .unified_occ import NUSCENES_GRID_CONFIG, remap_nuscenes_labels


class NuScenes5FrameStage1LidarDataset(Dataset):
    """Front-camera 5-frame Occ3D-nuScenes dataset in the Stage-1 batch format."""

    class_names = (
        "other",
        "barrier",
        "bicycle",
        "bus",
        "car",
        "construction_vehicle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "trailer",
        "truck",
        "driveable_surface",
        "other_flat",
        "sidewalk",
        "terrain",
        "manmade",
        "vegetation",
        "free",
    )

    def __init__(
        self,
        processed_root: str,
        split: str = "train",
        num_frames: int = 5,
        frame_stride: int = 1,
        output_resolution: Tuple[int, int] = (512, 160),
        max_points_per_sweep: int = 0,
        apply_camera_mask: bool = True,
        apply_lidar_mask: bool = False,
    ) -> None:
        super().__init__()
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.processed_root = processed_root
        self.split = split
        self.num_frames = int(num_frames)
        self.frame_stride = int(frame_stride)
        self.output_resolution = (int(output_resolution[0]), int(output_resolution[1]))
        self.max_points_per_sweep = int(max_points_per_sweep)
        self.apply_camera_mask = bool(apply_camera_mask)
        self.apply_lidar_mask = bool(apply_lidar_mask)
        self.ignore_label = 255
        self.grid_config = NUSCENES_GRID_CONFIG

        self._meta_cache: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
        self.samples: List[Tuple[str, int]] = []
        self._index_samples()

    def _scene_dirs(self) -> List[str]:
        prefix = f"{self.split}_scene-"
        return sorted(
            name
            for name in os.listdir(self.processed_root)
            if name.startswith(prefix) and osp.isdir(osp.join(self.processed_root, name))
        )

    def _frame_npz(self, scene: str, frame: int) -> str:
        return osp.join(self.processed_root, scene, f"{frame:06d}_0.npz")

    def _voxel_npz(self, scene: str, frame: int) -> str:
        return osp.join(self.processed_root, scene, "voxels", f"{frame:06d}.npz")

    def _lidar_bin(self, scene: str, frame: int) -> str:
        return osp.join(self.processed_root, scene, "lidar", f"{frame:06d}.bin")

    def _meta_npz(self, scene: str, frame: int) -> str:
        return osp.join(self.processed_root, scene, "meta", f"{frame:06d}.npz")

    def _frame_meta(self, scene: str, frame: int) -> Dict[str, np.ndarray]:
        key = (scene, int(frame))
        if key not in self._meta_cache:
            path = self._meta_npz(scene, frame)
            if not osp.isfile(path):
                raise FileNotFoundError(
                    f"Missing nuScenes processed metadata: {path}. "
                    "The dataset now reads calibration and poses only from processed_root."
                )
            with np.load(path) as z:
                self._meta_cache[key] = {name: np.asarray(z[name]) for name in z.files}
        return self._meta_cache[key]

    @staticmethod
    def _decode_camera_name(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.bytes_):
            return bytes(value).decode("utf-8")
        return str(value)

    def _front_camera_index(self, meta: Dict[str, np.ndarray]) -> int:
        names = meta.get("camera_names")
        if names is None:
            return 0
        decoded = [self._decode_camera_name(v) for v in np.asarray(names).reshape(-1)]
        if "CAM_FRONT" in decoded:
            return decoded.index("CAM_FRONT")
        return 0

    @staticmethod
    def _select_matrix(
        meta: Dict[str, np.ndarray],
        key: str,
        cam_idx: int,
        shape: Tuple[int, int],
    ) -> np.ndarray:
        if key not in meta:
            raise KeyError(key)
        arr = np.asarray(meta[key], dtype=np.float64)
        if arr.shape == shape:
            return arr
        if arr.ndim == 3 and arr.shape[1:] == shape and 0 <= cam_idx < arr.shape[0]:
            return arr[cam_idx]
        raise ValueError(f"processed meta field {key!r} has unexpected shape {arr.shape}")

    def _index_samples(self) -> None:
        history = (self.num_frames - 1) * self.frame_stride
        for scene in self._scene_dirs():
            voxel_dir = osp.join(self.processed_root, scene, "voxels")
            if not osp.isdir(voxel_dir):
                continue
            for name in sorted(os.listdir(voxel_dir)):
                if not name.endswith(".npz"):
                    continue
                t = int(osp.splitext(name)[0])
                if t - history < 0:
                    continue
                ok = True
                for k in range(self.num_frames):
                    fid = t - k * self.frame_stride
                    if not osp.isfile(self._frame_npz(scene, fid)):
                        ok = False
                        break
                    if not osp.isfile(self._meta_npz(scene, fid)):
                        ok = False
                        break
                    if not osp.isfile(self._lidar_bin(scene, fid)):
                        ok = False
                        break
                if ok:
                    self.samples.append((scene, t))

    def __len__(self) -> int:
        return len(self.samples)

    def _frame_calibration(
        self, scene: str, frame: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        meta = self._frame_meta(scene, frame)
        cam_idx = self._front_camera_index(meta)
        K = self._select_matrix(meta, "camera_intrinsics", cam_idx, (3, 3))
        try:
            cam2world = self._select_matrix(meta, "cam_to_world", cam_idx, (4, 4))
        except KeyError:
            T_world_from_ego = self._select_matrix(meta, "world_from_ego", cam_idx, (4, 4))
            T_ego_from_cam = self._select_matrix(meta, "ego_from_cam", cam_idx, (4, 4))
            cam2world = T_world_from_ego @ T_ego_from_cam

        # The network expects per-dataset grid-frame points. For nuScenes this is
        # the ego frame used by Occ3D voxels; processed lidar/*.bin keeps the raw
        # LIDAR_TOP frame, so _load_points applies lidar->ego from processed meta.
        try:
            T_cam_from_ego = self._select_matrix(meta, "cam_from_ego", cam_idx, (4, 4))
        except KeyError:
            T_ego_from_cam = self._select_matrix(meta, "ego_from_cam", cam_idx, (4, 4))
            T_cam_from_ego = np.linalg.inv(T_ego_from_cam)
        if "ego_from_lidar" in meta:
            T_ego_from_lidar = self._select_matrix(meta, "ego_from_lidar", cam_idx, (4, 4))
        elif "lidar_to_ego" in meta:
            T_ego_from_lidar = self._select_matrix(meta, "lidar_to_ego", cam_idx, (4, 4))
        elif "lidar_from_ego" in meta:
            T_lidar_from_ego = self._select_matrix(meta, "lidar_from_ego", cam_idx, (4, 4))
            T_ego_from_lidar = np.linalg.inv(T_lidar_from_ego)
        else:
            raise KeyError(
                f"Missing ego_from_lidar/lidar_to_ego in processed meta for "
                f"{scene}/{frame:06d}."
            )
        return K, cam2world, T_cam_from_ego, T_ego_from_lidar

    def _load_view(self, scene: str, frame: int, timestep_index: int) -> Dict[str, Any]:
        frame_npz = np.load(self._frame_npz(scene, frame))
        image = np.asarray(frame_npz["image"])
        K_raw = np.asarray(frame_npz["intrinsics"], dtype=np.float64)
        cam2world_raw = np.asarray(frame_npz["cam2world"], dtype=np.float64)
        try:
            _K_meta, cam2world_meta, _, _ = self._frame_calibration(scene, frame)
            cam2world_raw = cam2world_meta
        except Exception:
            pass

        img_pil = Image.fromarray(image)
        placeholder_depth = np.zeros(image.shape[:2], dtype=np.float32)
        img_pil_out, _, intr_out = crop_resize_if_necessary(
            img_pil, placeholder_depth, K_raw, self.output_resolution
        )
        img_arr = np.asarray(img_pil_out)
        H, W = img_arr.shape[:2]
        return {
            "img": ImgNorm(img_arr),
            "true_shape": np.int32((H, W)),
            "camera_pose": np.eye(4, dtype=np.float32),
            "camera_intrinsics": intr_out.astype(np.float32),
            "cam2world": cam2world_raw.astype(np.float32),
            "timestep": int(timestep_index),
            "is_raymap": False,
            "is_metric_scale": True,
            "frame_id": int(frame),
            "label": f"{scene}_{frame:06d}_cam0",
        }

    def _load_points(self, scene: str, frame: int) -> torch.Tensor:
        pts = np.fromfile(self._lidar_bin(scene, frame), dtype=np.float32).reshape(-1, 4)
        if self.max_points_per_sweep > 0 and pts.shape[0] > self.max_points_per_sweep:
            idx = np.linspace(0, pts.shape[0] - 1, self.max_points_per_sweep).astype(np.int64)
            pts = pts[idx]
        _, _, _, T_ego_from_lidar = self._frame_calibration(scene, frame)
        xyz_ego = pts[:, :3].astype(np.float64) @ T_ego_from_lidar[:3, :3].T + T_ego_from_lidar[:3, 3]
        pts = pts.copy()
        pts[:, :3] = xyz_ego.astype(np.float32)
        # Match KITTI-style reflectance scale used by the shared VFE.
        pts[:, 3] = np.clip(pts[:, 3], 0.0, 255.0) / 255.0
        return torch.from_numpy(pts)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        scene, t = self.samples[index]
        frame_ids = [t - k * self.frame_stride for k in range(self.num_frames)]

        views = [self._load_view(scene, fid, k) for k, fid in enumerate(frame_ids)]
        voxel_npz = np.load(self._voxel_npz(scene, t))
        voxel_label = remap_nuscenes_labels(np.asarray(voxel_npz["voxel_label"]))
        if self.apply_camera_mask and "mask_camera" in voxel_npz:
            voxel_label = voxel_label.copy()
            voxel_label[~np.asarray(voxel_npz["mask_camera"]).astype(bool)] = self.ignore_label
        if self.apply_lidar_mask and "mask_lidar" in voxel_npz:
            voxel_label = voxel_label.copy()
            voxel_label[~np.asarray(voxel_npz["mask_lidar"]).astype(bool)] = self.ignore_label

        # Single front camera supervision: ignore cells behind the ego center.
        voxel_label = voxel_label.copy()
        voxel_label[: voxel_label.shape[0] // 2, :, :] = self.ignore_label

        _, _, T_cam0_from_ego, _ = self._frame_calibration(scene, frame_ids[0])
        K_per_frame: List[np.ndarray] = []
        T_cam_from_ego: List[np.ndarray] = []
        for fid, view in zip(frame_ids, views):
            K_per_frame.append(np.asarray(view["camera_intrinsics"], dtype=np.float32))
            try:
                _, _, T_cf, _ = self._frame_calibration(scene, fid)
            except Exception:
                T_cf = T_cam0_from_ego
            T_cam_from_ego.append(T_cf.astype(np.float32))

        image_hw = np.asarray(views[0]["true_shape"], dtype=np.int32).reshape(2)
        points_per_frame = [self._load_points(scene, int(fid)) for fid in frame_ids]
        grid = self.grid_config.as_tensors()

        return {
            "views": views,
            "voxel_label": torch.from_numpy(voxel_label.astype(np.int64)),
            "T_target_from_refcam": torch.from_numpy(
                np.linalg.inv(T_cam0_from_ego).astype(np.float32)
            ),
            "T_cam_from_velo": torch.from_numpy(np.stack(T_cam_from_ego, axis=0)),
            "K_per_frame": torch.from_numpy(np.stack(K_per_frame, axis=0)),
            "image_hw": torch.from_numpy(image_hw),
            "points_per_frame": points_per_frame,
            "dataset_name": "nuscenes",
            "sequence": scene,
            "target_frame_id": int(t),
            "frame_ids": tuple(int(f) for f in frame_ids),
            **{k: v for k, v in grid.items() if k != "dataset_name"},
        }


def collate_stage1_nuscenes_lidar(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = collate_stage1(batch)
    out["T_cam_from_velo"] = torch.stack([b["T_cam_from_velo"] for b in batch], dim=0)
    out["K_per_frame"] = torch.stack([b["K_per_frame"] for b in batch], dim=0)
    out["image_hw"] = torch.stack([b["image_hw"] for b in batch], dim=0)
    out["points_per_frame"] = [b["points_per_frame"] for b in batch]
    out["dataset_name"] = [b["dataset_name"] for b in batch]
    out["half_grid_size"] = torch.stack([b["half_grid_size"] for b in batch], dim=0)
    out["half_voxel_origin"] = torch.stack([b["half_voxel_origin"] for b in batch], dim=0)
    out["half_voxel_size"] = torch.stack([b["half_voxel_size"] for b in batch], dim=0)
    out["fusion_vox_origin"] = torch.stack([b["fusion_vox_origin"] for b in batch], dim=0)
    out["fusion_vox_size"] = torch.stack([b["fusion_vox_size"] for b in batch], dim=0)
    out["fusion_vox_grid"] = torch.stack([b["fusion_vox_grid"] for b in batch], dim=0)
    return out


__all__ = ["NuScenes5FrameStage1LidarDataset", "collate_stage1_nuscenes_lidar"]
