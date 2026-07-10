"""Unified occupancy label and grid metadata for KITTI + Occ3D-nuScenes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


UNIFIED_SSC_CLASS_NAMES: Tuple[str, ...] = (
    "empty",
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
    "other",
    "barrier",
    "bus",
    "construction-vehicle",
    "traffic-cone",
    "trailer",
    "manmade",
)

NUSCENES_TO_UNIFIED = np.array(
    [
        20,  # other
        21,  # barrier
        2,   # bicycle
        22,  # bus
        1,   # car
        23,  # construction_vehicle
        3,   # motorcycle
        6,   # pedestrian
        24,  # traffic_cone
        25,  # trailer
        4,   # truck
        9,   # driveable_surface
        12,  # other_flat
        11,  # sidewalk
        17,  # terrain
        26,  # manmade
        15,  # vegetation
        0,   # free
    ],
    dtype=np.int64,
)


@dataclass(frozen=True)
class GridConfig:
    dataset_name: str
    full_grid_size: Tuple[int, int, int]
    full_voxel_origin: Tuple[float, float, float]
    full_voxel_size: Tuple[float, float, float]
    half_grid_size: Tuple[int, int, int]
    half_voxel_origin: Tuple[float, float, float]
    half_voxel_size: Tuple[float, float, float]
    fusion_vox_origin: Tuple[float, float, float]
    fusion_vox_size: Tuple[float, float, float]
    fusion_vox_grid: Tuple[int, int, int]

    def as_tensors(self) -> Dict[str, torch.Tensor | str]:
        return {
            "dataset_name": self.dataset_name,
            "grid_size": torch.tensor(self.full_grid_size, dtype=torch.long),
            "voxel_origin": torch.tensor(self.full_voxel_origin, dtype=torch.float32),
            "voxel_size": torch.tensor(self.full_voxel_size, dtype=torch.float32),
            "half_grid_size": torch.tensor(self.half_grid_size, dtype=torch.long),
            "half_voxel_origin": torch.tensor(self.half_voxel_origin, dtype=torch.float32),
            "half_voxel_size": torch.tensor(self.half_voxel_size, dtype=torch.float32),
            "fusion_vox_origin": torch.tensor(self.fusion_vox_origin, dtype=torch.float32),
            "fusion_vox_size": torch.tensor(self.fusion_vox_size, dtype=torch.float32),
            "fusion_vox_grid": torch.tensor(self.fusion_vox_grid, dtype=torch.long),
        }


KITTI_GRID_CONFIG = GridConfig(
    dataset_name="kitti",
    full_grid_size=(256, 256, 32),
    full_voxel_origin=(0.0, -25.6, -2.0),
    full_voxel_size=(0.2, 0.2, 0.2),
    half_grid_size=(128, 128, 16),
    half_voxel_origin=(0.0, -25.6, -2.0),
    half_voxel_size=(0.4, 0.4, 0.4),
    fusion_vox_origin=(-25.6, -2.0, 0.0),
    fusion_vox_size=(0.4, 0.4, 0.4),
    fusion_vox_grid=(128, 16, 128),
)

NUSCENES_GRID_CONFIG = GridConfig(
    dataset_name="nuscenes",
    full_grid_size=(200, 200, 16),
    full_voxel_origin=(-40.0, -40.0, -1.0),
    full_voxel_size=(0.4, 0.4, 0.4),
    half_grid_size=(100, 100, 8),
    half_voxel_origin=(-40.0, -40.0, -1.0),
    half_voxel_size=(0.8, 0.8, 0.8),
    fusion_vox_origin=(-40.0, -3.2, 0.0),
    fusion_vox_size=(0.8, 0.8, 0.8),
    fusion_vox_grid=(100, 8, 100),
)

GRID_CONFIGS: Dict[str, GridConfig] = {
    "kitti": KITTI_GRID_CONFIG,
    "nuscenes": NUSCENES_GRID_CONFIG,
}


def remap_nuscenes_labels(label: np.ndarray, ignore_label: int = 255) -> np.ndarray:
    out = np.full(label.shape, int(ignore_label), dtype=np.int64)
    valid = (label >= 0) & (label < len(NUSCENES_TO_UNIFIED))
    out[valid] = NUSCENES_TO_UNIFIED[label[valid]]
    out[label == ignore_label] = int(ignore_label)
    return out


def remap_kitti_labels(label: np.ndarray, ignore_label: int = 255) -> np.ndarray:
    out = label.astype(np.int64, copy=True)
    valid = ((out >= 0) & (out < 20)) | (out == ignore_label)
    out[~valid] = int(ignore_label)
    return out


__all__ = [
    "GRID_CONFIGS",
    "KITTI_GRID_CONFIG",
    "NUSCENES_GRID_CONFIG",
    "NUSCENES_TO_UNIFIED",
    "UNIFIED_SSC_CLASS_NAMES",
    "GridConfig",
    "remap_kitti_labels",
    "remap_nuscenes_labels",
]
