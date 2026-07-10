"""Remove selected keys from nuScenes processed camera npz files."""
from __future__ import annotations

import argparse
import os
import os.path as osp
import tempfile
import zipfile
from typing import Dict, Iterable, Sequence

import numpy as np


def _raw_scene_name(processed_scene: str) -> str:
    if processed_scene.startswith("train_"):
        return processed_scene[len("train_") :]
    if processed_scene.startswith("val_"):
        return processed_scene[len("val_") :]
    return processed_scene


def _iter_processed_scenes(
    processed_root: str,
    splits: Sequence[str],
    scenes: Sequence[str] | None,
) -> Iterable[str]:
    wanted = set(scenes or [])
    for name in sorted(os.listdir(processed_root)):
        path = osp.join(processed_root, name)
        if not osp.isdir(path):
            continue
        if wanted and name not in wanted and _raw_scene_name(name) not in wanted:
            continue
        if splits and not any(name.startswith(f"{split}_scene-") for split in splits):
            continue
        yield name


def _is_camera_npz(name: str) -> bool:
    if not name.endswith(".npz") or "_" not in name:
        return False
    frame, cam_ext = name.split("_", 1)
    cam = cam_ext[:-4]
    return frame.isdigit() and cam.isdigit()


def _npz_uses_compression(path: str) -> bool:
    with zipfile.ZipFile(path, "r") as zf:
        return any(info.compress_type != zipfile.ZIP_STORED for info in zf.infolist())


def _write_npz(path: str, data: Dict[str, np.ndarray], compressed: bool) -> None:
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{osp.basename(path)}.", suffix=".tmp.npz", dir=osp.dirname(path)
    )
    os.close(fd)
    try:
        if compressed:
            np.savez_compressed(tmp_path, **data)
        else:
            np.savez(tmp_path, **data)
        os.replace(tmp_path, path)
    finally:
        if osp.exists(tmp_path):
            os.unlink(tmp_path)


def _remove_keys(path: str, keys: set[str], compression: str, dry_run: bool) -> str:
    with np.load(path) as z:
        present = set(z.files)
        remove = present & keys
        if not remove:
            return "unchanged"
        data = {key: z[key] for key in z.files if key not in remove}

    if dry_run:
        return "would_update"

    if compression == "preserve":
        compressed = _npz_uses_compression(path)
    else:
        compressed = compression == "compressed"
    _write_npz(path, data, compressed=compressed)
    return "updated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove extra keys, such as cam_to_world, from processed nuScenes camera npz files."
    )
    parser.add_argument("--processed-root", default="data/nuscenes_processed")
    parser.add_argument("--keys", nargs="+", default=["cam_to_world"])
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Optional scene filter, e.g. train_scene-0001 or scene-0001.",
    )
    parser.add_argument(
        "--compression",
        choices=("preserve", "compressed", "stored"),
        default="preserve",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-scenes", type=int, default=0)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = set(args.keys)
    counters = {"updated": 0, "would_update": 0, "unchanged": 0, "errors": 0}
    scenes_seen = 0
    files_seen = 0

    for scene in _iter_processed_scenes(args.processed_root, args.splits, args.scenes):
        scenes_seen += 1
        if args.limit_scenes > 0 and scenes_seen > args.limit_scenes:
            break
        scene_dir = osp.join(args.processed_root, scene)
        for name in sorted(os.listdir(scene_dir)):
            if not _is_camera_npz(name):
                continue
            if args.limit_files > 0 and files_seen >= args.limit_files:
                break
            path = osp.join(scene_dir, name)
            try:
                status = _remove_keys(path, keys, args.compression, dry_run=args.dry_run)
                counters[status] += 1
            except Exception as exc:
                counters["errors"] += 1
                print(f"[error] {path}: {exc}")
            files_seen += 1
            if args.log_every > 0 and files_seen % args.log_every == 0:
                print(f"processed={files_seen}, counters={counters}")

    action = "dry-run" if args.dry_run else "done"
    print(f"{action}: scenes_seen={scenes_seen}, files_seen={files_seen}, keys={sorted(keys)}, counters={counters}")


if __name__ == "__main__":
    main()
