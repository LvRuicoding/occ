#!/usr/bin/env python3
import argparse
import os
import pathlib
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

WAYMO_ROOT_URL = "https://drive.google.com/drive/folders/13WxRl9Zb_AshEwvD96Uwz8cHjRNrtfQk"
NUSC_VOXEL04_URL = "https://drive.google.com/drive/folders/1Xarc91cNCNN3h8Vum-REbI-f0UlSf5Fc"


def run(cmd):
    print("+ " + " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def require_gdown():
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown is not installed in this Python environment. Run:\n"
            f"  {sys.executable} -m pip install -U gdown"
        ) from exc
    return gdown


def gdown_list_folder(gdown, url, proxy):
    print(f"+ list gdown folder {url}", flush=True)
    files = gdown.download_folder(
        url=url,
        quiet=True,
        proxy=proxy,
        use_cookies=False,
        skip_download=True,
        remaining_ok=True,
    )
    if files is None:
        raise RuntimeError(f"Failed to list Google Drive folder: {url}")
    return files


def item_name(item):
    return os.path.basename(str(item.path).rstrip("/"))


def download_drive_file(gdown, item, output_path, continue_download, proxy):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"+ gdown file {item.path} -> {output_path}", flush=True)
    result = gdown.download(
        id=item.id,
        output=str(output_path),
        quiet=False,
        proxy=proxy,
        use_cookies=False,
        resume=continue_download,
    )
    if result is None:
        raise RuntimeError(f"Failed to download {item.path}")


def download_drive_files(gdown, items, output_root, continue_download, proxy):
    for item in items:
        download_drive_file(gdown, item, output_root / item.path, continue_download, proxy)


def download_waymo_voxel04(gdown, output_root, continue_download, proxy):
    output_root.mkdir(parents=True, exist_ok=True)
    items = gdown_list_folder(gdown, WAYMO_ROOT_URL, proxy)
    voxel04_items = [item for item in items if str(item.path).startswith("voxel04/")]
    if not voxel04_items:
        names = [str(item.path) for item in items]
        raise RuntimeError(f"Could not find files under Waymo voxel04. Root contains: {names}")

    if len(voxel04_items) == 50:
        print(
            "Warning: gdown listed exactly 50 Waymo voxel04 files. "
            "Google Drive folder listing may be truncated by gdown.",
            flush=True,
        )
    download_drive_files(gdown, voxel04_items, output_root, continue_download, proxy)


def download_nuscenes_labels(gdown, output_root, continue_download, keep_archive, proxy):
    output_root.mkdir(parents=True, exist_ok=True)
    items = gdown_list_folder(gdown, NUSC_VOXEL04_URL, proxy)
    wanted = {"annotations.json", "gts.tar.gz"}
    found = set()

    for item in items:
        name = item_name(item)
        if name not in wanted:
            continue
        found.add(name)
        download_drive_file(gdown, item, output_root / name, continue_download, proxy)

    missing = wanted - found
    if missing:
        names = [item_name(item) for item in items]
        raise RuntimeError(f"Missing nuScenes files {sorted(missing)}. Folder contains: {names}")

    archive = output_root / "gts.tar.gz"
    gts_dir = output_root / "gts"
    if archive.exists():
        if gts_dir.exists():
            print(f"Skip extraction because {gts_dir} already exists.", flush=True)
        else:
            run(["tar", "-xzf", archive, "-C", output_root])
        if not keep_archive:
            archive.unlink()


def print_summary(waymo_root, nuscenes_root):
    for path in (waymo_root, nuscenes_root):
        if path.exists():
            run(["du", "-sh", path])
    print("\nCheck:", flush=True)
    print(f"  Waymo labels:    {waymo_root / 'voxel04'}", flush=True)
    print(f"  nuScenes labels: {nuscenes_root / 'gts'}", flush=True)
    print(f"  nuScenes anno:   {nuscenes_root / 'annotations.json'}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Download Occ3D Waymo and nuScenes 3D labels.")
    parser.add_argument("--repo-root", type=pathlib.Path, default=REPO_ROOT)
    parser.add_argument(
        "--waymo-output",
        type=pathlib.Path,
        default=None,
        help="Default: <repo-root>/data/Occ3D-Waymo",
    )
    parser.add_argument(
        "--nuscenes-output",
        type=pathlib.Path,
        default=None,
        help="Default: <repo-root>/raw_data/nuscenes",
    )
    parser.add_argument("--skip-waymo", action="store_true")
    parser.add_argument("--skip-nuscenes", action="store_true")
    parser.add_argument("--no-continue", action="store_true", help="Do not resume partial gdown downloads.")
    parser.add_argument("--keep-nuscenes-archive", action="store_true")
    parser.add_argument(
        "--proxy",
        default=None,
        help="Proxy URL for gdown, e.g. http://127.0.0.1:7890 or socks5://127.0.0.1:1080.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    gdown = require_gdown()

    waymo_root = args.waymo_output or (args.repo_root / "data" / "Occ3D-Waymo")
    nuscenes_root = args.nuscenes_output or (args.repo_root / "raw_data" / "nuscenes")
    waymo_root = waymo_root.resolve()
    nuscenes_root = nuscenes_root.resolve()
    continue_download = not args.no_continue
    proxy = args.proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        print(f"Using proxy: {proxy}", flush=True)

    if not args.skip_waymo:
        download_waymo_voxel04(gdown, waymo_root, continue_download, proxy)
    if not args.skip_nuscenes:
        download_nuscenes_labels(
            gdown,
            nuscenes_root,
            continue_download,
            args.keep_nuscenes_archive,
            proxy,
        )

    print_summary(waymo_root, nuscenes_root)


if __name__ == "__main__":
    main()
