"""Stage-1 KITTI LiDAR dataset with optional per-frame dense depth maps."""
from __future__ import annotations

from .. import _paths  # noqa: F401

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from occany.utils.helpers import crop_resize_if_necessary

from .kitti_stage1 import Kitti5FrameStage1Dataset, collate_stage1
from .kitti_stage1_mono_lidar import Kitti5FrameStage1LidarDataset, collate_stage1_lidar


class _DenseDepthMixin:
    """Mixin adding per-frame dense depth maps from processed frame npz files.

    Each processed frame ``*.npz`` may contain ``dense_depthmap`` in metric
    depth units (meters). Frames without that key, or with no valid positive
    finite depth, are kept in the sample but marked invalid for the dense depth
    auxiliary loss.
    """

    dense_depth_key = "dense_depthmap"

    def _load_dense_depth(self, seq: str, frame: int) -> Tuple[np.ndarray, bool]:
        with np.load(self._frame_npz(seq, frame)) as npz:
            image = np.asarray(npz["image"])
            intrinsics = np.asarray(npz["intrinsics"], dtype=np.float64)

            if self.dense_depth_key in npz.files:
                depth = np.asarray(npz[self.dense_depth_key], dtype=np.float32)
                has_depth = bool(np.isfinite(depth).any() and np.any(depth > 0.0))
            else:
                depth = np.zeros(image.shape[:2], dtype=np.float32)
                has_depth = False

        img_pil = Image.fromarray(image)
        _img_out, depth_out, _intr_out = crop_resize_if_necessary(
            img_pil,
            depth,
            intrinsics,
            self.output_resolution,
        )
        depth_out = np.asarray(depth_out, dtype=np.float32)
        valid = np.isfinite(depth_out) & (depth_out > 0.0)
        if not bool(valid.any()):
            has_depth = False
            depth_out = np.zeros_like(depth_out, dtype=np.float32)
        else:
            depth_out = np.where(valid, depth_out, 0.0).astype(np.float32)
        return depth_out, has_depth

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = super().__getitem__(index)
        seq = data["sequence"]

        dense_depths: List[torch.Tensor] = []
        frame_mask: List[bool] = []
        for fid in data["frame_ids"]:
            depth, has_depth = self._load_dense_depth(seq, int(fid))
            dense_depths.append(torch.from_numpy(depth))
            frame_mask.append(bool(has_depth))

        data["dense_depth"] = torch.stack(dense_depths, dim=0)  # (N, H, W)
        data["dense_depth_frame_mask"] = torch.tensor(frame_mask, dtype=torch.bool)
        return data


class Kitti5FrameStage1DenseDepthDataset(_DenseDepthMixin, Kitti5FrameStage1Dataset):
    """Stage-1 sample plus optional dense depth, without raw LiDAR sweeps."""


class Kitti5FrameStage1LidarDenseDepthDataset(
    _DenseDepthMixin,
    Kitti5FrameStage1LidarDataset,
):
    """Stage-1 LiDAR sample plus optional dense depth supervision."""


def collate_stage1_dense_depth(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate Stage-1 samples with optional dense depth supervision."""
    out = collate_stage1(batch)
    out["dense_depth"] = torch.stack([b["dense_depth"] for b in batch], dim=0)
    out["dense_depth_frame_mask"] = torch.stack(
        [b["dense_depth_frame_mask"] for b in batch], dim=0
    )
    return out


def collate_stage1_lidar_dense_depth(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate Stage-1 LiDAR samples with optional dense depth supervision."""
    out = collate_stage1_lidar(batch)
    out["dense_depth"] = torch.stack([b["dense_depth"] for b in batch], dim=0)
    out["dense_depth_frame_mask"] = torch.stack(
        [b["dense_depth_frame_mask"] for b in batch], dim=0
    )
    return out


__all__ = [
    "Kitti5FrameStage1DenseDepthDataset",
    "Kitti5FrameStage1LidarDenseDepthDataset",
    "collate_stage1_dense_depth",
    "collate_stage1_lidar_dense_depth",
]
