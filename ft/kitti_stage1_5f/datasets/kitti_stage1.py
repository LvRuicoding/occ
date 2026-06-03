"""Stage-1 SemanticKITTI dataset: 5 consecutive left-cam frames + voxel GT.

Layout assumed:
  processed_root/<split>_<seq>/<frame:06d>_<cam_idx>.npz   (image+intrinsics+cam2world)
  processed_root/<split>_<seq>/voxels/<frame:06d>.npz      (voxel_label)
  processed_root/<split>_<seq>/calib.txt                   (calib.txt; for Tr only)

The target frame `t` is fed FIRST and treated as the OccAny reconstruction
reference. The remaining 4 frames are fed in reverse time order
(t-1, t-2, t-3, t-4). With this convention `T_target_from_refcam` is the
static velo<-cam2 rigid sensor calibration, identical across frames in a
sequence.
"""
from __future__ import annotations

from .. import _paths  # noqa: F401  (must come before any occany.* imports)

import os
import os.path as osp
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from occany.utils.helpers import crop_resize_if_necessary
from occany.utils.image_util import ImgNorm


KITTI_SSC_CLASS_NAMES = (
    "empty", "car", "bicycle", "motorcycle", "truck", "other-vehicle",
    "person", "bicyclist", "motorcyclist", "road", "parking", "sidewalk",
    "other-ground", "building", "fence", "vegetation", "trunk", "terrain",
    "pole", "traffic-sign",
)

KITTI_SPLITS: Dict[str, List[str]] = {
    "train":    ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
    "val":      ["08"],
    "trainval": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
}


def _parse_calib(calib_path: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    raw: Dict[str, np.ndarray] = {}
    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, vals = line.split(":", 1)
            arr = np.array([float(v) for v in vals.split()], dtype=np.float64)
            raw[key] = arr
    out["P2"] = raw["P2"].reshape(3, 4)
    out["P3"] = raw["P3"].reshape(3, 4)
    if "Tr" in raw:
        Tr = np.eye(4, dtype=np.float64)
        Tr[:3, :4] = raw["Tr"].reshape(3, 4)
        out["Tr"] = Tr  # T_cam0_from_velo (KITTI convention)
    return out


def _T_cami_from_cam0(P: np.ndarray) -> np.ndarray:
    """Recover T_cami_from_cam0 from P_i = K_i [I | -K_i^{-1} t_cam_in_rect].

    P[:, :3] == K_i (after KITTI rectification).
    T_cami_from_cam0 has identity rotation and translation = K_i^{-1} @ P[:, 3].
    """
    K = P[:3, :3]
    t = P[:3, 3]
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.linalg.inv(K) @ t
    return T


def _static_T_velo_from_cam2(calib: Dict[str, np.ndarray]) -> np.ndarray:
    """T mapping a point in cam_2 coords to velo coords for any frame in a seq."""
    Tr = calib["Tr"]                # T_cam0_from_velo
    T_cam2_from_cam0 = _T_cami_from_cam0(calib["P2"])
    T_velo_from_cam0 = np.linalg.inv(Tr)
    T_cam0_from_cam2 = np.linalg.inv(T_cam2_from_cam0)
    return T_velo_from_cam0 @ T_cam0_from_cam2  # T_velo_from_cam2


class Kitti5FrameStage1Dataset(Dataset):
    """5-frame monocular SemanticKITTI SSC dataset.

    Each sample = (left-cam frames [t, t-1, t-2, t-3, t-4], voxel GT at t).
    """

    def __init__(
        self,
        processed_root: str,
        kittiodo_root: str | None = None,
        split: str = "train",
        num_frames: int = 5,
        frame_stride: int = 1,
        output_resolution: Tuple[int, int] = (512, 160),
        cam_idx: int = 0,
        load_dense_depth: bool = False,
    ) -> None:
        super().__init__()
        if split not in KITTI_SPLITS:
            raise ValueError(f"split={split!r} not in {list(KITTI_SPLITS)}")
        self.processed_root = processed_root
        self.split = split
        self.num_frames = int(num_frames)
        self.frame_stride = int(frame_stride)
        self.output_resolution = (int(output_resolution[0]), int(output_resolution[1]))
        self.cam_idx = int(cam_idx)
        self.load_dense_depth = bool(load_dense_depth)

        self.class_names: Tuple[str, ...] = KITTI_SSC_CLASS_NAMES
        self.n_classes = len(self.class_names)
        self.empty_class = 0
        self.ignore_label = 255

        self.voxel_origin = np.array([0.0, -25.6, -2.0], dtype=np.float32)
        self.voxel_size = np.array([0.2, 0.2, 0.2], dtype=np.float32)
        self.grid_size = np.array([256, 256, 32], dtype=np.int64)

        # Per-sequence cache: { seq: {"calib": dict, "T_velo_from_cam2": (4,4)} }
        self._seq_cache: Dict[str, Dict[str, Any]] = {}
        self.samples: List[Tuple[str, int]] = []
        for seq in KITTI_SPLITS[split]:
            self._index_sequence(seq)

    def _seq_dir(self, seq: str) -> str:
        which = "train" if seq in KITTI_SPLITS["train"] else "val"
        return osp.join(self.processed_root, f"{which}_{seq}")

    def _resolve_calib_path(self, seq: str) -> str:
        """Resolve calib.txt from the same processed sequence directory."""
        p = osp.join(self._seq_dir(seq), "calib.txt")
        if osp.isfile(p):
            return p
        raise FileNotFoundError(
            f"Missing calib.txt for sequence {seq}: {p}. "
            "Copy per-sequence calib.txt into data/kitti_processed first, "
            "for example with tools/copy_kitti_calib_to_processed.py."
        )

    def _get_seq_calib(self, seq: str) -> Dict[str, Any]:
        if seq in self._seq_cache:
            return self._seq_cache[seq]
        calib_path = self._resolve_calib_path(seq)
        calib = _parse_calib(calib_path)
        if "Tr" not in calib:
            raise RuntimeError(
                f"calib.txt at {calib_path} is missing the 'Tr' line "
                "(velo->cam0). This is required to compute the static "
                "velo<-cam2 transform. Make sure the processed sequence "
                "directory contains the KITTI Odometry calib.txt."
            )
        T_velo_from_cam2 = _static_T_velo_from_cam2(calib).astype(np.float32)
        entry = {"calib": calib, "T_velo_from_cam2": T_velo_from_cam2}
        self._seq_cache[seq] = entry
        return entry

    def _frame_npz(self, seq: str, frame: int) -> str:
        return osp.join(self._seq_dir(seq), f"{frame:06d}_{self.cam_idx}.npz")

    def _voxel_npz(self, seq: str, frame: int) -> str:
        return osp.join(self._seq_dir(seq), "voxels", f"{frame:06d}.npz")

    def _index_sequence(self, seq: str) -> None:
        seq_dir = self._seq_dir(seq)
        voxel_dir = osp.join(seq_dir, "voxels")
        if not osp.isdir(voxel_dir):
            return
        # Touch calib once so missing calib raises now, not mid-training.
        self._get_seq_calib(seq)

        history = (self.num_frames - 1) * self.frame_stride
        for name in sorted(os.listdir(voxel_dir)):
            if not name.endswith(".npz"):
                continue
            t = int(osp.splitext(name)[0])
            if t - history < 0:
                continue
            ok = True
            for k in range(self.num_frames):
                fid = t - k * self.frame_stride
                if not osp.isfile(self._frame_npz(seq, fid)):
                    ok = False
                    break
            if ok:
                self.samples.append((seq, t))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_view(
        self,
        seq: str,
        frame: int,
        timestep_index: int,
    ) -> Dict[str, Any]:
        npz = np.load(self._frame_npz(seq, frame))
        image = np.asarray(npz["image"])           # (H, W, 3) uint8
        intrinsics = np.asarray(npz["intrinsics"], dtype=np.float64)
        # cam2world is kept around for debug/extension but not needed for lifting.
        cam2world = np.asarray(npz["cam2world"], dtype=np.float64)

        load_view_dense_depth = self.load_dense_depth and int(timestep_index) == 0
        has_dense_depth = load_view_dense_depth and "dense_depthmap" in npz.files
        if load_view_dense_depth and has_dense_depth:
            dense_depth = np.asarray(npz["dense_depthmap"], dtype=np.float32)
            if dense_depth.shape != image.shape[:2]:
                raise RuntimeError(
                    f"dense_depthmap shape {dense_depth.shape} does not match "
                    f"image shape {image.shape[:2]} for {self._frame_npz(seq, frame)}"
                )
        else:
            dense_depth = np.zeros(image.shape[:2], dtype=np.float32)
        img_pil = Image.fromarray(image)
        img_pil_out, dense_depth_out, intr_out = crop_resize_if_necessary(
            img_pil, dense_depth, intrinsics, self.output_resolution
        )
        img_arr = np.asarray(img_pil_out)
        H, W = img_arr.shape[:2]
        img_tensor = ImgNorm(img_arr)  # (3, H, W) float in [-1, 1]
        view = dict(
            img=img_tensor,
            true_shape=np.int32((H, W)),
            camera_pose=np.eye(4, dtype=np.float32),
            camera_intrinsics=intr_out.astype(np.float32),
            cam2world=cam2world.astype(np.float32),
            timestep=int(timestep_index),
            is_raymap=False,
            is_metric_scale=True,
            frame_id=int(frame),
            label=f"{seq}_{frame:06d}_cam{self.cam_idx}",
        )
        if load_view_dense_depth:
            view["dense_depth"] = np.asarray(dense_depth_out, dtype=np.float32)
            view["has_dense_depth"] = bool(has_dense_depth)
        return view

    def __getitem__(self, index: int) -> Dict[str, Any]:
        seq, t = self.samples[index]
        seq_entry = self._get_seq_calib(seq)
        T_velo_from_cam2 = seq_entry["T_velo_from_cam2"]

        # First view is the target frame (= reconstruction reference for OccAny).
        frame_ids: List[int] = [t - k * self.frame_stride for k in range(self.num_frames)]

        views: List[Dict[str, Any]] = []
        for k, fid in enumerate(frame_ids):
            views.append(self._load_view(seq, fid, timestep_index=k))

        voxel_npz = np.load(self._voxel_npz(seq, t))
        voxel_label = np.asarray(voxel_npz["voxel_label"]).astype(np.int64)
        if tuple(voxel_label.shape) != tuple(self.grid_size.tolist()):
            raise RuntimeError(
                f"voxel_label shape {voxel_label.shape} != grid_size {tuple(self.grid_size.tolist())}"
            )

        return dict(
            views=views,
            voxel_label=torch.from_numpy(voxel_label).long(),
            T_target_from_refcam=torch.from_numpy(T_velo_from_cam2.astype(np.float32)),
            voxel_origin=torch.from_numpy(self.voxel_origin.copy()),
            voxel_size=torch.from_numpy(self.voxel_size.copy()),
            grid_size=torch.from_numpy(self.grid_size.copy()),
            sequence=seq,
            target_frame_id=int(t),
            frame_ids=tuple(int(f) for f in frame_ids),
        )


def collate_stage1(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_views = len(batch[0]["views"])
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

    out: Dict[str, Any] = dict(
        views=stacked_views,
        voxel_label=torch.stack([b["voxel_label"] for b in batch], dim=0),
        T_target_from_refcam=torch.stack([b["T_target_from_refcam"] for b in batch], dim=0),
        voxel_origin=torch.stack([b["voxel_origin"] for b in batch], dim=0),
        voxel_size=torch.stack([b["voxel_size"] for b in batch], dim=0),
        grid_size=torch.stack([b["grid_size"] for b in batch], dim=0),
        sequence=[b["sequence"] for b in batch],
        target_frame_id=[b["target_frame_id"] for b in batch],
        frame_ids=[b["frame_ids"] for b in batch],
    )
    return out
