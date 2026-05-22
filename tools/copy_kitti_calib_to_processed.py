#!/usr/bin/env python3
"""Copy per-sequence KITTI calib.txt files into data/kitti_processed."""

from __future__ import annotations

import argparse
import filecmp
import re
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIB_ROOT = REPO_ROOT / "raw_data" / "semantickitti_occany_root"
DEFAULT_PROCESSED_ROOT = REPO_ROOT / "data" / "kitti_processed"
SEQ_DIR_RE = re.compile(r"^(?:train|val)_(\d{2})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy calib.txt from a KITTI/SemanticKITTI root into matching "
            "<split>_<seq> directories under data/kitti_processed."
        )
    )
    parser.add_argument(
        "--calib-root",
        type=Path,
        default=DEFAULT_CALIB_ROOT,
        help=(
            "Root containing sequence calib files. Supported layouts include "
            "<root>/dataset/sequences/<seq>/calib.txt, "
            "<root>/sequences/<seq>/calib.txt, and <root>/<seq>/calib.txt."
        ),
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help="Root containing train_00, val_08, ... directories.",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        help="Optional sequence ids to copy, e.g. 00 01 08. Defaults to all processed dirs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing destination calib.txt.",
    )
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help="Skip missing source calib.txt files instead of failing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without writing files.",
    )
    return parser.parse_args()


def normalize_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def processed_sequence_dirs(processed_root: Path) -> dict[str, list[Path]]:
    if not processed_root.is_dir():
        raise FileNotFoundError(f"processed root does not exist: {processed_root}")

    seq_dirs: dict[str, list[Path]] = {}
    for path in sorted(processed_root.iterdir()):
        if not path.is_dir():
            continue
        match = SEQ_DIR_RE.match(path.name)
        if match:
            seq_dirs.setdefault(match.group(1), []).append(path)
    return seq_dirs


def resolve_source_calib(calib_root: Path, seq: str) -> Path | None:
    candidates = (
        calib_root / "dataset" / "sequences" / seq / "calib.txt",
        calib_root / "sequences" / seq / "calib.txt",
        calib_root / seq / "calib.txt",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def selected_sequence_dirs(args: argparse.Namespace) -> dict[str, list[Path]]:
    seq_dirs = processed_sequence_dirs(args.processed_root)
    if not args.sequences:
        return seq_dirs

    selected: dict[str, list[Path]] = {}
    missing = []
    for seq in sorted(args.sequences):
        seq = f"{int(seq):02d}" if seq.isdigit() else seq
        if seq in seq_dirs:
            selected[seq] = seq_dirs[seq]
        else:
            missing.append(seq)
    if missing:
        raise FileNotFoundError(
            "requested sequences are not present under processed root: "
            + ", ".join(missing)
        )
    return selected


def copy_one(src: Path, dst: Path, overwrite: bool, dry_run: bool) -> str:
    if dst.exists():
        if filecmp.cmp(src, dst, shallow=False):
            return "same"
        if not overwrite:
            return "exists"

    if dry_run:
        return "would_copy"

    shutil.copy2(src, dst)
    return "copied"


def main() -> None:
    args = parse_args()
    args.calib_root = normalize_path(args.calib_root)
    args.processed_root = normalize_path(args.processed_root)

    seq_dirs = selected_sequence_dirs(args)
    if not seq_dirs:
        raise RuntimeError(f"no train_XX/val_XX sequence dirs found in {args.processed_root}")

    counts = {"copied": 0, "would_copy": 0, "same": 0, "exists": 0, "missing": 0}
    missing: list[str] = []

    for seq, dirs in sorted(seq_dirs.items()):
        src = resolve_source_calib(args.calib_root, seq)
        if src is None:
            counts["missing"] += len(dirs)
            missing.append(seq)
            if args.ignore_missing:
                print(f"missing source calib for sequence {seq}; skipped")
                continue
            continue

        for seq_dir in dirs:
            dst = seq_dir / "calib.txt"
            status = copy_one(src, dst, args.overwrite, args.dry_run)
            counts[status] += 1
            print(f"{status}: {src} -> {dst}")

    if missing and not args.ignore_missing:
        raise FileNotFoundError(
            "missing source calib.txt for sequences: "
            + ", ".join(sorted(missing))
            + f"\ncalib root checked: {args.calib_root}"
        )

    print(
        "summary: "
        f"copied={counts['copied']}, "
        f"would_copy={counts['would_copy']}, "
        f"same={counts['same']}, "
        f"exists={counts['exists']}, "
        f"missing={counts['missing']}"
    )
    if counts["exists"]:
        print("existing destination files were left unchanged; use --overwrite to replace them.")


if __name__ == "__main__":
    main()
