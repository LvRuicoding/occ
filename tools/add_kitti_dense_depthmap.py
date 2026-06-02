#!/usr/bin/env python3
"""Add KITTI depth-completion groundtruth to kitti_processed .npz files.

The existing kitti_processed files are organized as:

    data/kitti_processed/<split>_<seq>/<frame:06d>_<cam_idx>.npz

where cam_idx 0 is image_2 / image_02 and cam_idx 1 is image_3 / image_03.
This script reads KITTI depth completion groundtruth PNGs, resizes them to the
processed image resolution, converts KITTI uint16 depth encoding to meters, and
writes a new ``dense_depthmap`` array into each .npz.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PROCESSED_ROOT = REPO_ROOT / "data" / "kitti_processed"
DEFAULT_DEPTH_ROOT = (
    REPO_ROOT
    / "raw_data"
    / "OpenDataLab___KITTI_depth_completion"
    / "KITTI_depth_completion"
)
DEFAULT_LOG_PATH = DEFAULT_PROCESSED_ROOT / "dense_depthmap_import.log"

ODOMETRY_SEQ_TO_RAW: Dict[str, Tuple[str, int]] = {
    "00": ("2011_10_03_drive_0027_sync", 0),
    "01": ("2011_10_03_drive_0042_sync", 0),
    "02": ("2011_10_03_drive_0034_sync", 0),
    "03": ("2011_09_26_drive_0067_sync", 0),
    "04": ("2011_09_30_drive_0016_sync", 0),
    "05": ("2011_09_30_drive_0018_sync", 0),
    "06": ("2011_09_30_drive_0020_sync", 0),
    "07": ("2011_09_30_drive_0027_sync", 0),
    "08": ("2011_09_30_drive_0028_sync", 1100),
    "09": ("2011_09_30_drive_0033_sync", 0),
    "10": ("2011_09_30_drive_0034_sync", 0),
}

CAM_IDX_TO_DEPTH_CAM = {
    "0": "image_02",
    "1": "image_03",
}


try:
    NEAREST = Image.Resampling.NEAREST
except AttributeError:  # Pillow < 9
    NEAREST = Image.NEAREST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Insert KITTI depth-completion groundtruth as dense_depthmap into "
            "existing kitti_processed .npz files."
        )
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help="Root of kitti_processed.",
    )
    parser.add_argument(
        "--depth-root",
        type=Path,
        default=DEFAULT_DEPTH_ROOT,
        help="Root of KITTI_depth_completion.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path for the import log.",
    )
    parser.add_argument(
        "--seq-drive-map",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping from odometry sequence ids to raw drive names, "
            "or to objects with drive/start_frame. Examples: "
            "{\"00\": \"2011_10_03_drive_0027_sync\"}, "
            "{\"08\": {\"drive\": \"2011_09_30_drive_0028_sync\", \"start_frame\": 1100}}."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help=(
            "Optional processed split directories to process, e.g. train_00 val_08. "
            "Defaults to all train_*/val_* directories."
        ),
    )
    parser.add_argument(
        "--field-name",
        default="dense_depthmap",
        help="Field name to add to each .npz.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite field-name if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be written.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap for quick checks.",
    )
    return parser.parse_args()


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )


def load_seq_drive_map(mapping_path: Optional[Path]) -> Dict[str, Tuple[str, int]]:
    mapping = dict(ODOMETRY_SEQ_TO_RAW)
    if mapping_path is None:
        return mapping

    with mapping_path.open("r", encoding="utf-8") as f:
        user_mapping = json.load(f)
    for seq, value in user_mapping.items():
        seq_key = f"{int(seq):02d}"
        default_start = mapping.get(seq_key, ("", 0))[1]
        if isinstance(value, str):
            mapping[seq_key] = (value, default_start)
        elif isinstance(value, dict):
            drive = value.get("drive") or value.get("raw_drive")
            if not drive:
                raise ValueError(f"missing drive/raw_drive for sequence {seq_key}")
            start_frame = int(value.get("start_frame", value.get("start", default_start)))
            mapping[seq_key] = (str(drive), start_frame)
        else:
            raise TypeError(f"unsupported mapping value for sequence {seq_key}: {value!r}")
    return mapping


def iter_processed_files(processed_root: Path, splits: Optional[Iterable[str]]) -> Iterable[Path]:
    if splits:
        split_dirs = [processed_root / split for split in splits]
    else:
        split_dirs = sorted(
            p
            for p in processed_root.iterdir()
            if p.is_dir() and (p.name.startswith("train_") or p.name.startswith("val_"))
        )

    for split_dir in split_dirs:
        if not split_dir.is_dir():
            logging.warning("skip missing processed split directory: %s", split_dir)
            continue
        yield from sorted(split_dir.glob("*.npz"))


def parse_processed_path(npz_path: Path) -> Optional[Tuple[str, str, str]]:
    split_name = npz_path.parent.name
    if "_" not in split_name:
        return None
    seq = split_name.rsplit("_", 1)[-1]

    stem_parts = npz_path.stem.rsplit("_", 1)
    if len(stem_parts) != 2:
        return None
    frame, cam_idx = stem_parts
    if not frame.isdigit() or cam_idx not in CAM_IDX_TO_DEPTH_CAM:
        return None
    return seq, frame, cam_idx


def build_depth_index(depth_root: Path) -> Dict[Tuple[str, str, str], Path]:
    """Index (raw_drive, depth_cam, frame_10digit) -> PNG path."""
    index: Dict[Tuple[str, str, str], Path] = {}

    for depth_split in ("train", "val"):
        split_root = depth_root / depth_split
        if not split_root.is_dir():
            logging.warning("skip missing depth split directory: %s", split_root)
            continue

        for drive_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
            gt_root = drive_dir / "proj_depth" / "groundtruth"
            if not gt_root.is_dir():
                continue

            for depth_cam in CAM_IDX_TO_DEPTH_CAM.values():
                cam_dir = gt_root / depth_cam
                if not cam_dir.is_dir():
                    continue
                for png_path in cam_dir.glob("*.png"):
                    key = (drive_dir.name, depth_cam, png_path.stem)
                    if key in index:
                        logging.warning(
                            "duplicate depth PNG for %s; keeping first: %s",
                            key,
                            index[key],
                        )
                        continue
                    index[key] = png_path

    return index


def read_depth_png_meters(png_path: Path, target_hw: Tuple[int, int]) -> np.ndarray:
    """Read KITTI uint16 depth PNG, resize to target H/W, return float32 meters."""
    target_h, target_w = target_hw

    with Image.open(png_path) as img:
        raw = np.asarray(img)

    if raw.ndim != 2:
        raise ValueError(f"expected single-channel depth PNG, got {raw.shape}: {png_path}")

    raw = raw.astype(np.uint16, copy=False)
    if raw.shape != (target_h, target_w):
        raw_img = Image.fromarray(raw)
        raw = np.asarray(raw_img.resize((target_w, target_h), resample=NEAREST)).astype(
            np.uint16,
            copy=False,
        )

    return (raw.astype(np.float32) / 256.0).astype(np.float32, copy=False)


def write_npz_atomic(npz_path: Path, arrays: Dict[str, np.ndarray]) -> None:
    tmp_path = npz_path.with_suffix(npz_path.suffix + ".tmp")
    try:
        with tmp_path.open("wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp_path, npz_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def add_dense_depthmaps(args: argparse.Namespace) -> None:
    processed_root = args.processed_root.resolve()
    depth_root = args.depth_root.resolve()
    seq_drive_map = load_seq_drive_map(args.seq_drive_map)

    logging.info("processed_root=%s", processed_root)
    logging.info("depth_root=%s", depth_root)
    logging.info("field_name=%s overwrite=%s dry_run=%s", args.field_name, args.overwrite, args.dry_run)

    if not processed_root.is_dir():
        raise FileNotFoundError(f"processed root does not exist: {processed_root}")
    if not depth_root.is_dir():
        raise FileNotFoundError(f"depth root does not exist: {depth_root}")

    logging.info("building depth PNG index")
    depth_index = build_depth_index(depth_root)
    logging.info("indexed %d depth PNG files", len(depth_index))

    processed_files = list(iter_processed_files(processed_root, args.splits))
    if args.max_files is not None:
        processed_files = processed_files[: args.max_files]
    logging.info("processing %d npz files", len(processed_files))

    written = 0
    would_write = 0
    skipped_existing = 0
    skipped_missing = 0
    skipped_bad_name = 0
    errors = 0

    for npz_path in tqdm(processed_files, desc="dense_depthmap"):
        parsed = parse_processed_path(npz_path)
        if parsed is None:
            skipped_bad_name += 1
            logging.warning("skip unrecognized processed path: %s", npz_path)
            continue

        seq, frame, cam_idx = parsed
        raw_info = seq_drive_map.get(seq)
        if raw_info is None:
            skipped_missing += 1
            logging.warning("skip %s: no raw drive mapping for sequence %s", npz_path, seq)
            continue

        raw_drive, raw_start_frame = raw_info
        depth_cam = CAM_IDX_TO_DEPTH_CAM[cam_idx]
        raw_frame = int(frame) + raw_start_frame
        depth_key = (raw_drive, depth_cam, f"{raw_frame:010d}")
        depth_path = depth_index.get(depth_key)
        if depth_path is None:
            skipped_missing += 1
            logging.info(
                "missing depth: npz=%s raw_drive=%s cam=%s frame=%s raw_frame=%010d",
                npz_path,
                raw_drive,
                depth_cam,
                frame,
                raw_frame,
            )
            continue

        try:
            with np.load(npz_path, allow_pickle=False) as npz:
                if args.field_name in npz.files and not args.overwrite:
                    skipped_existing += 1
                    continue
                arrays = {key: npz[key] for key in npz.files}

            image = arrays.get("image")
            if image is None or image.ndim < 2:
                raise ValueError(f"{npz_path} does not contain a valid image array")
            target_hw = (int(image.shape[0]), int(image.shape[1]))

            dense_depthmap = read_depth_png_meters(depth_path, target_hw)

            if args.dry_run:
                would_write += 1
                continue

            arrays[args.field_name] = dense_depthmap
            write_npz_atomic(npz_path, arrays)
            written += 1
        except Exception:
            errors += 1
            logging.exception("failed to process %s", npz_path)

    logging.info(
        "done written=%d would_write=%d skipped_existing=%d skipped_missing=%d "
        "skipped_bad_name=%d errors=%d",
        written,
        would_write,
        skipped_existing,
        skipped_missing,
        skipped_bad_name,
        errors,
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_path)
    add_dense_depthmaps(args)


if __name__ == "__main__":
    main()
