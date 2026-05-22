"""SemanticKITTI SSC fine-tuning dataset.

Each item is a 3-frame stereo (6 views total) sequence:
  frames = [t-2*stride, t-stride, t]  (target frame is LAST)
  views per frame = [left_cam (image_2), right_cam (image_3)]

The target frame's left camera defines the anchor coordinate system that the
rest of the pipeline operates in (camera_pose=eye(4) for that view, all other
camera_poses expressed in this anchor's frame). Voxel GT lives in the target
frame's velodyne coordinate system, which is a fixed rigid transform from the
anchor camera (lidar_to_world = T_cam_2_velo_target).
"""
from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import occany.datasets.semantic_kitti_io as SemanticKittiIO
from occany.utils.helpers import crop_resize_if_necessary
from occany.utils.image_util import ImgNorm


KITTI_SSC_CLASS_NAMES = (
    "empty", "car", "bicycle", "motorcycle", "truck", "other-vehicle",
    "person", "bicyclist", "motorcyclist", "road", "parking", "sidewalk",
    "other-ground", "building", "fence", "vegetation", "trunk", "terrain",
    "pole", "traffic-sign",
)


def _read_calib(path: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    raw: Dict[str, np.ndarray] = {}
    with open(path, "r") as f:
        for line in f.readlines():
            if not line.strip():
                continue
            key, value = line.split(":", 1)
            raw[key] = np.array([float(x) for x in value.split()])
    out["P2"] = raw["P2"].reshape(3, 4)
    out["P3"] = raw["P3"].reshape(3, 4)
    out["Tr"] = np.eye(4, dtype=np.float64)
    out["Tr"][:3, :4] = raw["Tr"].reshape(3, 4)
    return out


def _parse_poses(path: str, calib: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """Parse poses.txt; returns per-frame T_velo_to_world (left-camera origin)."""
    Tr = calib["Tr"]
    Tr_inv = np.linalg.inv(Tr)
    poses: List[np.ndarray] = []
    with open(path, "r") as f:
        for line in f:
            vals = [float(v) for v in line.strip().split()]
            T = np.eye(4, dtype=np.float64)
            T[0, :4] = vals[0:4]
            T[1, :4] = vals[4:8]
            T[2, :4] = vals[8:12]
            poses.append(Tr_inv @ T @ Tr)
    return poses


def _baseline_from_p2_p3(P2: np.ndarray, P3: np.ndarray) -> float:
    """KITTI: P3 = K @ [R|t], with R=I and t=(-baseline*fx, 0, 0).
    Returns baseline in meters (positive)."""
    fx = P2[0, 0]
    return float(-(P3[0, 3] - P2[0, 3]) / fx)


class SemanticKittiSSCDataset(Dataset):
    """SemanticKITTI SSC dataset producing 3-frame stereo sequences.

    Args:
      semkitti_root: path with `dataset/sequences/<seq>/voxels` (and labels).
      kittiodo_root: same layout, providing image_2/image_3 (KITTI Odometry).
      remap_lut_path: path to `semantic_kitti.yaml`.
      split: 'train' | 'val' | 'trainval'.
      frame_stride: temporal stride between context frames and target.
      output_resolution: (W, H), defaults to (512, 160).
      load_lidar: if True, returns raw point clouds for each frame; placeholder
        for future fusion. False by default.
      target_stride_voxels: voxel files are stored every 5 frames. Skips frames
        without a voxel label.
    """

    def __init__(
        self,
        semkitti_root: str,
        kittiodo_root: str,
        remap_lut_path: str,
        split: str,
        frame_stride: int = 5,
        output_resolution: Tuple[int, int] = (512, 160),
        load_lidar: bool = False,
        target_stride_voxels: int = 5,
        pid: int = 0,
        world: int = 1,
    ) -> None:
        super().__init__()
        self.semkitti_root = semkitti_root
        self.kittiodo_root = kittiodo_root
        self.split = split
        self.frame_stride = int(frame_stride)
        self.output_resolution = tuple(output_resolution)
        self.load_lidar = bool(load_lidar)
        self.target_stride_voxels = int(target_stride_voxels)

        self.class_names: Tuple[str, ...] = KITTI_SSC_CLASS_NAMES
        self.n_classes = len(self.class_names)
        self.empty_class = 0
        self.ignore_label = 255

        self.scene_size = (51.2, 51.2, 6.4)
        self.voxel_origin = np.array([0.0, -25.6, -2.0], dtype=np.float32)
        self.voxel_size = 0.2
        self.grid_size = (256, 256, 32)

        splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            "val": ["08"],
            "trainval": ["00", "01", "02", "03", "04", "05", "06", "07", "08",
                         "09", "10"],
        }
        if split not in splits:
            raise ValueError(f"Unknown split={split!r}; expected one of {list(splits)}")
        self.sequences = splits[split]

        self.remap_lut = SemanticKittiIO.get_remap_lut(remap_lut_path)

        self._sequence_meta: Dict[str, Dict[str, Any]] = {}
        self.samples: List[Dict[str, Any]] = []
        for seq in self.sequences:
            self._index_sequence(seq)

        if world > 1:
            self.samples = [s for i, s in enumerate(self.samples) if i % world == pid]

    def _index_sequence(self, seq: str) -> None:
        seq_dir = os.path.join(self.semkitti_root, "dataset", "sequences", seq)
        odo_seq_dir = os.path.join(self.kittiodo_root, "dataset", "sequences", seq)
        voxel_dir = os.path.join(seq_dir, "voxels")
        image_2_dir = os.path.join(odo_seq_dir, "image_2")
        image_3_dir = os.path.join(odo_seq_dir, "image_3")
        calib_path = os.path.join(odo_seq_dir, "calib.txt")
        poses_path = os.path.join(odo_seq_dir, "poses.txt")

        if not os.path.isdir(voxel_dir):
            return
        if not os.path.isfile(calib_path) or not os.path.isfile(poses_path):
            return

        calib = _read_calib(calib_path)
        poses = _parse_poses(poses_path, calib)
        baseline = _baseline_from_p2_p3(calib["P2"], calib["P3"])

        T_velo_2_cam = calib["Tr"]
        T_cam_2_velo = np.linalg.inv(T_velo_2_cam)

        self._sequence_meta[seq] = dict(
            calib=calib,
            poses=poses,
            baseline=baseline,
            T_velo_2_cam=T_velo_2_cam,
            T_cam_2_velo=T_cam_2_velo,
            image_2_dir=image_2_dir,
            image_3_dir=image_3_dir,
        )

        voxel_files = sorted(glob.glob(os.path.join(voxel_dir, "*.label")))
        target_frames = []
        for vf in voxel_files:
            stem = os.path.splitext(os.path.basename(vf))[0]
            target_frames.append(int(stem))

        for tf in target_frames:
            t0 = tf - 2 * self.frame_stride
            t1 = tf - self.frame_stride
            if t0 < 0:
                continue
            for fid in (t0, t1, tf):
                img2 = os.path.join(image_2_dir, f"{fid:06d}.png")
                img3 = os.path.join(image_3_dir, f"{fid:06d}.png")
                if not (os.path.isfile(img2) and os.path.isfile(img3)):
                    break
            else:
                self.samples.append(
                    dict(sequence=seq, frame_ids=(t0, t1, tf), target=tf)
                )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_voxel_label(label_path: str, invalid_path: str, remap_lut: np.ndarray) -> np.ndarray:
        label = SemanticKittiIO._read_label_SemKITTI(label_path)
        invalid = SemanticKittiIO._read_invalid_SemKITTI(invalid_path)
        label = remap_lut[label.astype(np.uint16)].astype(np.int64)
        label[np.isclose(invalid, 1)] = 255
        return label.reshape(256, 256, 32)

    @staticmethod
    def _read_lidar(path: str) -> np.ndarray:
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return pts

    def _build_view(
        self,
        meta: Dict[str, Any],
        frame_id: int,
        cam: int,
        anchor_world_to_anchor: np.ndarray,
    ) -> Dict[str, Any]:
        """Build a single view dict matching OccAny's per-view contract."""
        if cam == 2:
            img_path = os.path.join(meta["image_2_dir"], f"{frame_id:06d}.png")
            P = meta["calib"]["P2"]
        elif cam == 3:
            img_path = os.path.join(meta["image_3_dir"], f"{frame_id:06d}.png")
            P = meta["calib"]["P3"]
        else:
            raise ValueError(cam)

        img = Image.open(img_path).convert("RGB")
        cam_k = P[:3, :3].copy()
        # KITTI's P matrices express the right cam as a translated copy of cam_2.
        # The intrinsic is identical (same K), so cam_k from P2/P3 [:3,:3] is fine.
        place_holder_depth = np.zeros((img.height, img.width), dtype=np.float32)
        downscaled_img, _, intrinsics2 = crop_resize_if_necessary(
            img, place_holder_depth, cam_k, self.output_resolution
        )

        # cam_<cam>_2_world (target-frame's left cam = world origin).
        T_velo_2_world_frame = meta["poses"][frame_id]
        T_cam_left_2_world = T_velo_2_world_frame @ meta["T_cam_2_velo"]
        T_cam2_2_world_anchor = anchor_world_to_anchor @ T_cam_left_2_world
        if cam == 2:
            T_cam_2_world = T_cam2_2_world_anchor
        else:
            # right cam = left cam translated by (+baseline, 0, 0) in cam frame
            offset = np.eye(4, dtype=np.float64)
            offset[0, 3] = meta["baseline"]
            T_cam_2_world = T_cam2_2_world_anchor @ offset

        img_arr = np.array(downscaled_img)
        img_tensor = ImgNorm(img_arr)  # (3, H, W) in [-1, 1]
        H, W = img_arr.shape[:2]
        return dict(
            img=img_tensor,
            true_shape=np.int32((H, W)),
            camera_pose=T_cam_2_world.astype(np.float32),
            camera_intrinsics=intrinsics2.astype(np.float32),
            frame_id=int(frame_id),
            cam=int(cam),
            scene_name=meta_seq_name(meta),
            label=f"{meta_seq_name(meta)}_{frame_id:06d}_cam{cam}",
            timestep=int(frame_id),
            is_raymap=False,
            is_metric_scale=True,
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        seq = sample["sequence"]
        frame_ids = sample["frame_ids"]
        target = sample["target"]
        meta = self._sequence_meta[seq]
        meta = dict(meta, _seq_name=seq)

        # Anchor: target frame's left camera in world frame.
        T_velo_2_world_target = meta["poses"][target]
        T_cam_left_2_world_target = T_velo_2_world_target @ meta["T_cam_2_velo"]
        anchor_world_to_anchor = np.linalg.inv(T_cam_left_2_world_target)

        views: List[Dict[str, Any]] = []
        for fid in frame_ids:
            for cam in (2, 3):
                v = self._build_view(meta, fid, cam, anchor_world_to_anchor)
                views.append(v)

        # Voxel GT: lives in target's velodyne frame.
        seq_dir = os.path.join(self.semkitti_root, "dataset", "sequences", seq)
        label_path = os.path.join(seq_dir, "voxels", f"{target:06d}.label")
        invalid_path = os.path.join(seq_dir, "voxels", f"{target:06d}.invalid")
        voxel_label = self._read_voxel_label(label_path, invalid_path, self.remap_lut)

        # lidar_to_world: target velodyne -> anchor (target's left cam).
        # World here = target's left cam, so we need velo -> cam = Tr.
        lidar_to_world = meta["T_velo_2_cam"].astype(np.float32)

        # Voxel origin in lidar (velodyne) frame -> for downstream lifting modules
        # if/when they need it.
        out: Dict[str, Any] = dict(
            views=views,  # 6 dicts (3 frames x 2 cams)
            voxel_label=torch.from_numpy(voxel_label).long(),
            lidar_to_world=torch.from_numpy(lidar_to_world).float(),
            voxel_size=torch.tensor([self.voxel_size] * 3, dtype=torch.float32),
            voxel_origin=torch.from_numpy(self.voxel_origin).float(),
            grid_size=torch.tensor(self.grid_size, dtype=torch.long),
            sequence=seq,
            target_frame_id=int(target),
            frame_ids=tuple(int(f) for f in frame_ids),
            T_velo_2_cam=torch.from_numpy(meta["T_velo_2_cam"]).float(),
            anchor_pose=torch.from_numpy(T_cam_left_2_world_target).float(),
        )

        # Optional: lidar paths / point clouds. Always expose paths so future
        # fusion code can opt-in without touching dataset internals.
        lidar_paths = [
            os.path.join(seq_dir, "velodyne", f"{f:06d}.bin")
            for f in frame_ids
        ]
        out["lidar_paths"] = lidar_paths
        if self.load_lidar:
            out["lidar_points"] = [
                torch.from_numpy(self._read_lidar(p)) for p in lidar_paths
            ]

        return out


def meta_seq_name(meta: Dict[str, Any]) -> str:
    return meta.get("_seq_name", "unknown")


def collate_ssc(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function: stacks fixed-shape fields, keeps variable-shape lists."""
    n_views = len(batch[0]["views"])
    # Stack each view across batch.
    stacked_views: List[Dict[str, Any]] = []
    for view_idx in range(n_views):
        per_view = [b["views"][view_idx] for b in batch]
        view_dict: Dict[str, Any] = {}
        sample0 = per_view[0]
        for key, v in sample0.items():
            vals = [pv[key] for pv in per_view]
            if isinstance(v, torch.Tensor):
                view_dict[key] = torch.stack(vals, dim=0)
            elif isinstance(v, np.ndarray):
                view_dict[key] = torch.from_numpy(np.stack(vals, axis=0))
            elif isinstance(v, (int, float, bool)):
                view_dict[key] = torch.tensor(vals)
            else:
                view_dict[key] = vals
        stacked_views.append(view_dict)

    out: Dict[str, Any] = {
        "views": stacked_views,
        "voxel_label": torch.stack([b["voxel_label"] for b in batch], dim=0),
        "lidar_to_world": torch.stack([b["lidar_to_world"] for b in batch], dim=0),
        "voxel_size": torch.stack([b["voxel_size"] for b in batch], dim=0),
        "voxel_origin": torch.stack([b["voxel_origin"] for b in batch], dim=0),
        "grid_size": torch.stack([b["grid_size"] for b in batch], dim=0),
        "T_velo_2_cam": torch.stack([b["T_velo_2_cam"] for b in batch], dim=0),
        "anchor_pose": torch.stack([b["anchor_pose"] for b in batch], dim=0),
        "sequence": [b["sequence"] for b in batch],
        "target_frame_id": [b["target_frame_id"] for b in batch],
        "frame_ids": [b["frame_ids"] for b in batch],
        "lidar_paths": [b["lidar_paths"] for b in batch],
    }
    if "lidar_points" in batch[0]:
        out["lidar_points"] = [b["lidar_points"] for b in batch]
    return out
