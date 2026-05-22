"""MonoScene-flavored Stage-1 SemanticKITTI dataset.

Same as ``Kitti5FrameStage1Dataset`` but additionally loads the 1/8
downsampled voxel label ``<frame:06d>_1_8.npy`` (produced by
``tools/preprocess_semantickitti_1_8_labels.py``) and pre-computes the
``CP_mega_matrix`` needed for MonoScene's relation loss.

The matrix computation is the function ``compute_CP_mega_matrix`` from
``monoscene/data/utils/helpers.py`` (BSD-3-Clause), vendored here so we
don't depend on the external MonoScene Python package at runtime.
"""
from __future__ import annotations

from .. import _paths  # noqa: F401  (must come before any occany.* imports)

import os
import os.path as osp
from typing import Any, Dict, List

import numpy as np
import torch

from .kitti_stage1 import (
    KITTI_SPLITS,
    KITTI_SSC_CLASS_NAMES,
    Kitti5FrameStage1Dataset,
)


def compute_CP_mega_matrix(target: np.ndarray, is_binary: bool = False) -> np.ndarray:
    """Vendored from MonoScene (monoscene/data/utils/helpers.py).

    target: (H, W, D) voxel labels at 1/8 resolution -- e.g. (32, 32, 4)
    returns: (n_relations, N, N_super) uint8 matrix, with
             N = H*W*D, N_super = (H//2)*(W//2)*(D//2),
             n_relations = 2 if is_binary else 4.
    """
    label = target.reshape(-1)
    label_row = label
    N = label.shape[0]
    super_voxel_size = [i // 2 for i in target.shape]
    if is_binary:
        matrix = np.zeros(
            (2, N, super_voxel_size[0] * super_voxel_size[1] * super_voxel_size[2]),
            dtype=np.uint8,
        )
    else:
        matrix = np.zeros(
            (4, N, super_voxel_size[0] * super_voxel_size[1] * super_voxel_size[2]),
            dtype=np.uint8,
        )

    for xx in range(super_voxel_size[0]):
        for yy in range(super_voxel_size[1]):
            for zz in range(super_voxel_size[2]):
                col_idx = (
                    xx * (super_voxel_size[1] * super_voxel_size[2])
                    + yy * super_voxel_size[2]
                    + zz
                )
                label_col_megas = np.array(
                    [
                        target[xx * 2,     yy * 2,     zz * 2],
                        target[xx * 2 + 1, yy * 2,     zz * 2],
                        target[xx * 2,     yy * 2 + 1, zz * 2],
                        target[xx * 2,     yy * 2,     zz * 2 + 1],
                        target[xx * 2 + 1, yy * 2 + 1, zz * 2],
                        target[xx * 2 + 1, yy * 2,     zz * 2 + 1],
                        target[xx * 2,     yy * 2 + 1, zz * 2 + 1],
                        target[xx * 2 + 1, yy * 2 + 1, zz * 2 + 1],
                    ]
                )
                label_col_megas = label_col_megas[label_col_megas != 255]
                for label_col_mega in label_col_megas:
                    label_col = np.ones(N) * label_col_mega
                    if not is_binary:
                        matrix[0, (label_row != 255) & (label_col == label_row) & (label_col != 0), col_idx] = 1.0
                        matrix[1, (label_row != 255) & (label_col != label_row) & (label_col != 0) & (label_row != 0), col_idx] = 1.0
                        matrix[2, (label_row != 255) & (label_row == label_col) & (label_col == 0), col_idx] = 1.0
                        matrix[3, (label_row != 255) & (label_row != label_col) & ((label_row == 0) | (label_col == 0)), col_idx] = 1.0
                    else:
                        matrix[0, (label_row != 255) & (label_col != label_row), col_idx] = 1.0
                        matrix[1, (label_row != 255) & (label_col == label_row), col_idx] = 1.0
    return matrix


class Kitti5FrameStage1MonoDataset(Kitti5FrameStage1Dataset):
    """Same as Kitti5FrameStage1Dataset, plus 1/8 label + CP_mega_matrix."""

    def __init__(self, *args, n_relations: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if n_relations not in (2, 4):
            raise ValueError(f"n_relations must be 2 or 4, got {n_relations}")
        self.n_relations = int(n_relations)
        # Pre-validate that every indexed frame has its 1_8 file.
        self._validate_voxel_1_8()

    def _voxel_1_8_npy(self, seq: str, frame: int) -> str:
        return osp.join(self._seq_dir(seq), "voxels", f"{frame:06d}_1_8.npy")

    def _validate_voxel_1_8(self) -> None:
        missing: List[str] = []
        for seq, t in self.samples:
            p = self._voxel_1_8_npy(seq, t)
            if not osp.isfile(p):
                missing.append(p)
                if len(missing) > 5:
                    break
        if missing:
            raise FileNotFoundError(
                "Missing 1/8 voxel labels required for MonoScene relation loss "
                "(produced by tools/preprocess_semantickitti_1_8_labels.py):\n"
                + "\n".join(f"  - {p}" for p in missing[:5])
                + ("\n  ... (more)" if len(missing) > 5 else "")
            )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = super().__getitem__(index)
        seq = data["sequence"]
        t = data["target_frame_id"]

        target_1_8 = np.load(self._voxel_1_8_npy(seq, t)).astype(np.float32)
        cp = compute_CP_mega_matrix(target_1_8, is_binary=(self.n_relations == 2))

        data["voxel_label_1_8"] = torch.from_numpy(target_1_8.astype(np.int64)).long()
        data["CP_mega_matrix"] = torch.from_numpy(cp)  # (n_rel, N, N_super) uint8
        return data


def collate_stage1_mono(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Same as collate_stage1, plus voxel_label_1_8 and CP_mega_matrix."""
    from .kitti_stage1 import collate_stage1  # local import to avoid cycles

    out = collate_stage1(batch)
    out["voxel_label_1_8"] = torch.stack([b["voxel_label_1_8"] for b in batch], dim=0)
    # CP_mega_matrix shape: (n_rel, N, N_super) -- same across samples in a
    # batch as long as grid_size is consistent.
    out["CP_mega_matrix"] = torch.stack([b["CP_mega_matrix"] for b in batch], dim=0)
    return out


__all__ = [
    "Kitti5FrameStage1MonoDataset",
    "collate_stage1_mono",
    "compute_CP_mega_matrix",
    "KITTI_SPLITS",
    "KITTI_SSC_CLASS_NAMES",
]
