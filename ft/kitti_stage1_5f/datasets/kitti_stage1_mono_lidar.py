"""MonoScene Stage-1 SemanticKITTI dataset extended with raw LiDAR sweeps.

Same outputs as ``Kitti5FrameStage1MonoDataset``, plus, per sample:
  - ``points_per_frame``: list of 5 float32 tensors, each (P_f, 4) -> (x, y, z, intensity)
    in the *velodyne* frame of that timestep (KITTI's raw .bin layout).
  - ``T_cam_from_velo``: (4, 4) float32 — static per sequence, depends on cam_idx.
  - ``K_per_frame``: (5, 3, 3) float32 — intrinsics matching the resized image
    (already produced by ``crop_resize_if_necessary`` upstream).
  - ``image_hw``: (2,) int32 — (H, W) of each processed frame (same across frames
    in a sample by construction).

The velodyne data is read from a separate ``velodyne_root`` (the raw KITTI
Odometry layout: ``<velodyne_root>/sequences/<seq>/velodyne/<frame:06d>.bin``).
"""
from __future__ import annotations

from .. import _paths  # noqa: F401

import os.path as osp
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .kitti_stage1 import _T_cami_from_cam0
from .kitti_stage1_mono import Kitti5FrameStage1MonoDataset


def _T_cami_from_velo(calib: Dict[str, np.ndarray], cam_idx: int) -> np.ndarray:
    """Return T_cam_i_from_velo for cam_idx in {0, 1, 2, 3}.

    Convention:
        T_cami_from_velo = T_cami_from_cam0 @ T_cam0_from_velo
    where T_cam0_from_velo = calib["Tr"] and T_cami_from_cam0 is recovered from P_i.
    """
    Tr = calib["Tr"]  # (4, 4) T_cam0_from_velo
    if cam_idx == 0:
        T_cami_from_cam0 = np.eye(4, dtype=np.float64)
    elif cam_idx == 1:
        T_cami_from_cam0 = _T_cami_from_cam0(calib["P1"]) if "P1" in calib else np.eye(4)
    elif cam_idx == 2:
        T_cami_from_cam0 = _T_cami_from_cam0(calib["P2"])
    elif cam_idx == 3:
        T_cami_from_cam0 = _T_cami_from_cam0(calib["P3"])
    else:
        raise ValueError(f"cam_idx must be in 0..3, got {cam_idx}")
    return (T_cami_from_cam0 @ Tr).astype(np.float64)


class Kitti5FrameStage1MonoLidarDataset(Kitti5FrameStage1MonoDataset):
    """``Kitti5FrameStage1MonoDataset`` + per-frame raw velodyne + K + T_cam_from_velo."""

    def __init__(
        self,
        *args,
        velodyne_root: str,
        max_points_per_sweep: int = 0,  # 0 = keep all
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.velodyne_root = velodyne_root
        self.max_points_per_sweep = int(max_points_per_sweep)
        # Sanity: at least the first sample's velodyne files should exist.
        if len(self.samples) > 0:
            seq0, t0 = self.samples[0]
            for k in range(self.num_frames):
                fid = t0 - k * self.frame_stride
                p = self._velodyne_bin(seq0, fid)
                if not osp.isfile(p):
                    raise FileNotFoundError(
                        f"Missing velodyne bin for sample 0: {p}. "
                        f"Check --velodyne_root and KITTI layout."
                    )

    def _velodyne_bin(self, seq: str, frame: int) -> str:
        return osp.join(self.velodyne_root, "sequences", seq, "velodyne", f"{frame:06d}.bin")

    def _load_points(self, seq: str, frame: int) -> np.ndarray:
        """Load (P, 4) float32 [x, y, z, intensity] from raw .bin."""
        p = self._velodyne_bin(seq, frame)
        pts = np.fromfile(p, dtype=np.float32).reshape(-1, 4)
        if self.max_points_per_sweep > 0 and pts.shape[0] > self.max_points_per_sweep:
            # Deterministic stride subsample, avoids RNG state thrash in workers.
            idx = np.linspace(0, pts.shape[0] - 1, self.max_points_per_sweep).astype(np.int64)
            pts = pts[idx]
        return pts

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = super().__getitem__(index)
        seq = data["sequence"]
        seq_entry = self._get_seq_calib(seq)
        calib = seq_entry["calib"]

        T_cam_from_velo = _T_cami_from_velo(calib, cam_idx=self.cam_idx).astype(np.float32)

        # Per-frame K already lives on each view (from crop_resize_if_necessary).
        # We stack into (5, 3, 3) for convenience downstream.
        Ks: List[np.ndarray] = []
        for v in data["views"]:
            K = v["camera_intrinsics"]
            if isinstance(K, torch.Tensor):
                K = K.numpy()
            Ks.append(K.astype(np.float32))
        K_per_frame = np.stack(Ks, axis=0)  # (5, 3, 3)

        # Image shape (H, W) — same across views in a sample.
        ts0 = data["views"][0]["true_shape"]
        if isinstance(ts0, torch.Tensor):
            ts0 = ts0.numpy()
        image_hw = np.asarray(ts0, dtype=np.int32).reshape(2)

        # Points per frame (in velo coords), variable-length tensors.
        frame_ids = data["frame_ids"]
        points_per_frame: List[torch.Tensor] = []
        for fid in frame_ids:
            pts = self._load_points(seq, int(fid))
            points_per_frame.append(torch.from_numpy(pts))

        data["points_per_frame"] = points_per_frame  # list of (P_f, 4)
        data["T_cam_from_velo"] = torch.from_numpy(T_cam_from_velo)  # (4, 4)
        data["K_per_frame"] = torch.from_numpy(K_per_frame)          # (5, 3, 3)
        data["image_hw"] = torch.from_numpy(image_hw)                # (2,)
        return data


def collate_stage1_mono_lidar(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Variant of ``collate_stage1_mono`` that also stacks the lidar/K/T_cam_from_velo
    fields. Variable-length point clouds are kept as nested lists:
    ``out["points_per_frame"][b][f]`` -> tensor (P, 4) for sample b, frame f.
    """
    from .kitti_stage1_mono import collate_stage1_mono  # local import to avoid cycles

    out = collate_stage1_mono(batch)
    out["T_cam_from_velo"] = torch.stack([b["T_cam_from_velo"] for b in batch], dim=0)
    out["K_per_frame"] = torch.stack([b["K_per_frame"] for b in batch], dim=0)
    out["image_hw"] = torch.stack([b["image_hw"] for b in batch], dim=0)
    out["points_per_frame"] = [b["points_per_frame"] for b in batch]
    return out


__all__ = [
    "Kitti5FrameStage1MonoLidarDataset",
    "collate_stage1_mono_lidar",
]
