"""Pack SemanticKITTI gt_voxels into the kitti_processed layout.

For every sequence under `--semkitti_root`/sequences/<seq>/gt_voxels we read
the .label / .invalid pair, apply the SemanticKITTI remap LUT (so labels are
in the 20-class SSC space) and mask invalid voxels as 255. Each per-frame
result is written to:

    <preprocessed_root>/<split>_<seq>/voxels/<frame:06d>.npz

with a single key `voxel_label` of shape (256, 256, 32) and dtype int16.
"""
from __future__ import annotations

import argparse
import glob
import os
import os.path as osp
from typing import Dict, List

import numpy as np
from tqdm import tqdm

import importlib.util

_REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
_SEMKITTI_IO_PATH = osp.join(_REPO_ROOT, "occany", "datasets", "semantic_kitti_io.py")
_spec = importlib.util.spec_from_file_location("semantic_kitti_io", _SEMKITTI_IO_PATH)
SemanticKittiIO = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SemanticKittiIO)


KITTI_SPLITS: Dict[str, List[str]] = {
    "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
    "val": ["08"],
}
GRID_SHAPE = (256, 256, 32)
DEFAULT_SEMKITTI_ROOT = osp.join(_REPO_ROOT, "raw_data", "semantickitti")
DEFAULT_PREPROCESSED_ROOT = osp.join(_REPO_ROOT, "data", "kitti_processed")


def _resolve_remap_lut(remap_path: str | None) -> np.ndarray:
    if remap_path is None:
        here = osp.dirname(osp.abspath(__file__))
        remap_path = osp.abspath(osp.join(here, "..", "..", "..", "occany", "datasets", "semantic_kitti.yaml"))
    return SemanticKittiIO.get_remap_lut(remap_path)


def _process_frame(label_path: str, invalid_path: str, remap_lut: np.ndarray) -> np.ndarray:
    label = SemanticKittiIO._read_label_SemKITTI(label_path)
    invalid = SemanticKittiIO._read_invalid_SemKITTI(invalid_path)
    label = remap_lut[label.astype(np.uint16)].astype(np.int32)
    label[np.isclose(invalid, 1)] = 255
    return label.reshape(GRID_SHAPE).astype(np.int16)


def _split_for_seq(seq: str) -> str:
    for split, seqs in KITTI_SPLITS.items():
        if seq in seqs:
            return split
    raise ValueError(f"Sequence {seq} is not in KITTI_SPLITS (train/val).")


def _process_sequence(seq: str, semkitti_root: str, out_root: str, remap_lut: np.ndarray) -> int:
    voxel_dir = osp.join(semkitti_root, "sequences", seq, "gt_voxels")
    if not osp.isdir(voxel_dir):
        return 0
    split = _split_for_seq(seq)
    out_dir = osp.join(out_root, f"{split}_{seq}", "voxels")
    os.makedirs(out_dir, exist_ok=True)

    label_files = sorted(glob.glob(osp.join(voxel_dir, "*.label")))
    n = 0
    for label_path in tqdm(label_files, desc=f"{split}_{seq}", leave=False):
        stem = osp.splitext(osp.basename(label_path))[0]
        invalid_path = osp.join(voxel_dir, stem + ".invalid")
        if not osp.isfile(invalid_path):
            continue
        out_path = osp.join(out_dir, stem + ".npz")
        voxel = _process_frame(label_path, invalid_path, remap_lut)
        np.savez_compressed(out_path, voxel_label=voxel)
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--semkitti_root",
        type=str,
        default=DEFAULT_SEMKITTI_ROOT,
        help="Root containing sequences/<seq>/gt_voxels/.",
    )
    parser.add_argument(
        "--preprocessed_root",
        type=str,
        default=DEFAULT_PREPROCESSED_ROOT,
        help="kitti_processed root; <split>_<seq>/voxels/ will be created inside it.",
    )
    parser.add_argument(
        "--remap_lut_path",
        type=str,
        default=None,
        help="Path to occany/datasets/semantic_kitti.yaml; default: auto.",
    )
    args = parser.parse_args()

    remap_lut = _resolve_remap_lut(args.remap_lut_path)
    total = 0
    for split, seqs in KITTI_SPLITS.items():
        for seq in seqs:
            total += _process_sequence(seq, args.semkitti_root, args.preprocessed_root, remap_lut)
    print(f"Wrote {total} voxel npz files under {args.preprocessed_root}")


if __name__ == "__main__":
    main()
