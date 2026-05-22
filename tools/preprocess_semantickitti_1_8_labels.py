#!/usr/bin/env python3
"""Generate MonoScene-style 1/8 SemanticKITTI voxel labels for kitti_processed."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SK_ROOT = REPO_ROOT / "raw_data" / "semantickitti"
DEFAULT_PROCESSED_ROOT = REPO_ROOT / "data" / "kitti_processed"
DEFAULT_YAML = REPO_ROOT / "occany" / "datasets" / "semantic_kitti.yaml"
SCENE_SIZE = (256, 256, 32)
TRAIN_SEQS = ("00", "01", "02", "03", "04", "05", "06", "07", "09", "10")
VAL_SEQS = ("08",)
ALL_SEQS = TRAIN_SEQS + VAL_SEQS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read SemanticKITTI .label/.invalid voxel files, apply the MonoScene "
            "label remap and 1/8 downsampling, then write <frame>_1_8.npy into "
            "data/kitti_processed/<split>_<seq>/voxels/."
        )
    )
    parser.add_argument("--semantic-kitti-root", type=Path, default=DEFAULT_SK_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--semantic-kitti-yaml", type=Path, default=DEFAULT_YAML)
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=list(ALL_SEQS),
        help="Sequence ids to process, e.g. 00 01 08.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing <frame>_1_8.npy files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs without writing files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of frames per sequence, useful for quick checks.",
    )
    return parser.parse_args()


def normalize_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def unpack(compressed: np.ndarray) -> np.ndarray:
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1
    return uncompressed


def get_remap_lut(config_path: Path) -> np.ndarray:
    with config_path.open("r", encoding="utf-8") as f:
        dataset_config = yaml.safe_load(f)
    learning_map = dataset_config["learning_map"]
    max_key = max(learning_map.keys())
    remap_lut = np.zeros((max_key + 100), dtype=np.int32)
    remap_lut[list(learning_map.keys())] = list(learning_map.values())
    remap_lut[remap_lut == 0] = 255
    remap_lut[0] = 0
    return remap_lut


def read_label(path: Path) -> np.ndarray:
    label = np.fromfile(path, dtype=np.uint16).astype(np.float32)
    expected = int(np.prod(SCENE_SIZE))
    if label.size != expected:
        raise ValueError(f"{path} has {label.size} voxels, expected {expected}")
    return label


def read_invalid(path: Path) -> np.ndarray:
    invalid = unpack(np.fromfile(path, dtype=np.uint8))
    expected = int(np.prod(SCENE_SIZE))
    if invalid.size != expected:
        raise ValueError(f"{path} has {invalid.size} voxels, expected {expected}")
    return invalid


def downsample_label(label: np.ndarray, voxel_size: tuple[int, int, int], downscale: int) -> np.ndarray:
    """MonoScene/NYU _downsample_label logic, used with downscale=8 here."""
    if downscale == 1:
        return label
    ds = downscale
    small_size = (
        voxel_size[0] // ds,
        voxel_size[1] // ds,
        voxel_size[2] // ds,
    )
    label_downscale = np.zeros(small_size, dtype=np.uint8)
    empty_t = 0.95 * ds * ds * ds
    s01 = small_size[0] * small_size[1]
    label_i = np.zeros((ds, ds, ds), dtype=np.int32)

    for i in range(small_size[0] * small_size[1] * small_size[2]):
        z = int(i / s01)
        y = int((i - z * s01) / small_size[0])
        x = int(i - z * s01 - y * small_size[0])

        label_i[:, :, :] = label[
            x * ds : (x + 1) * ds,
            y * ds : (y + 1) * ds,
            z * ds : (z + 1) * ds,
        ]
        label_bin = label_i.flatten()

        zero_count_0 = np.array(np.where(label_bin == 0)).size
        zero_count_255 = np.array(np.where(label_bin == 255)).size

        zero_count = zero_count_0 + zero_count_255
        if zero_count > empty_t:
            label_downscale[x, y, z] = 0 if zero_count_0 > zero_count_255 else 255
        else:
            label_i_s = label_bin[
                np.where(np.logical_and(label_bin > 0, label_bin < 255))
            ]
            label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
    return label_downscale


def sequences_root(root: Path) -> Path:
    candidates = (root / "dataset" / "sequences", root / "sequences")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"could not find dataset/sequences or sequences under {root}")


def processed_seq_dir(processed_root: Path, seq: str) -> Path:
    split = "val" if seq in VAL_SEQS else "train"
    return processed_root / f"{split}_{seq}"


def source_frame_paths(seq_dir: Path, frame_id: str) -> tuple[Path, Path]:
    label_candidates = (
        seq_dir / "gt_voxels" / f"{frame_id}.label",
        seq_dir / "voxels" / f"{frame_id}.label",
    )
    invalid_candidates = (
        seq_dir / "gt_voxels" / f"{frame_id}.invalid",
        seq_dir / "voxels" / f"{frame_id}.invalid",
    )
    label_path = next((p for p in label_candidates if p.is_file()), None)
    invalid_path = next((p for p in invalid_candidates if p.is_file()), None)
    if label_path is None:
        raise FileNotFoundError(f"missing label for frame {frame_id}: {label_candidates}")
    if invalid_path is None:
        raise FileNotFoundError(f"missing invalid mask for frame {frame_id}: {invalid_candidates}")
    return label_path, invalid_path


def target_frame_ids(seq_out_dir: Path) -> list[str]:
    voxel_dir = seq_out_dir / "voxels"
    if not voxel_dir.is_dir():
        raise FileNotFoundError(f"missing processed voxel directory: {voxel_dir}")
    return [p.stem for p in sorted(voxel_dir.glob("*.npz"))]


def process_frame(
    label_path: Path,
    invalid_path: Path,
    out_path: Path,
    remap_lut: np.ndarray,
    overwrite: bool,
    dry_run: bool,
) -> str:
    if out_path.exists() and not overwrite:
        return "exists"
    if dry_run:
        return "would_write"

    label = read_label(label_path)
    invalid = read_invalid(invalid_path)
    label = remap_lut[label.astype(np.uint16)].astype(np.float32)
    label[np.isclose(invalid, 1)] = 255
    label = label.reshape(SCENE_SIZE)
    label_1_8 = downsample_label(label, SCENE_SIZE, 8)
    np.save(out_path, label_1_8)
    return "wrote"


def main() -> None:
    args = parse_args()
    sk_root = normalize_path(args.semantic_kitti_root)
    processed_root = normalize_path(args.processed_root)
    yaml_path = normalize_path(args.semantic_kitti_yaml)
    seqs_root = sequences_root(sk_root)
    remap_lut = get_remap_lut(yaml_path)

    counts = {"wrote": 0, "would_write": 0, "exists": 0}
    for raw_seq in args.sequences:
        seq = f"{int(raw_seq):02d}" if raw_seq.isdigit() else raw_seq
        seq_dir = seqs_root / seq
        seq_out_dir = processed_seq_dir(processed_root, seq)
        frame_ids = target_frame_ids(seq_out_dir)
        if args.limit is not None:
            frame_ids = frame_ids[: args.limit]

        for frame_id in frame_ids:
            label_path, invalid_path = source_frame_paths(seq_dir, frame_id)
            out_path = seq_out_dir / "voxels" / f"{frame_id}_1_8.npy"
            status = process_frame(
                label_path,
                invalid_path,
                out_path,
                remap_lut,
                args.overwrite,
                args.dry_run,
            )
            counts[status] += 1
            print(f"{status}: {label_path} + {invalid_path} -> {out_path}")

    print(
        "summary: "
        f"wrote={counts['wrote']}, "
        f"would_write={counts['would_write']}, "
        f"exists={counts['exists']}"
    )
    if counts["exists"]:
        print("existing outputs were left unchanged; use --overwrite to replace them.")


if __name__ == "__main__":
    main()
