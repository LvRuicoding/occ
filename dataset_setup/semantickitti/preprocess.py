"""Preprocess SemanticKITTI sequences into the OccAny .npz format used for DDAD.

For each frame of each KITTI odometry sequence (00-10) and for both color
cameras (image_2 / image_3), we produce a single .npz file containing:
    image     : (H, W, 3) uint8, RGB, resized so that long side == target_resolution
    depthmap  : (H, W) float32, sparse LiDAR depth in meters at the resized resolution
    intrinsics: (3, 3) float64, camera matrix in the resized image
    cam2world : (4, 4) float64, camera (cam_i) to world transform

Output layout (split = official KITTI Odometry train/val):
    save_dir/<split>_<seq>/<frame:06d>_<cam_idx>.npz
    e.g. save_dir/train_00/000123_0.npz   (image_2)
         save_dir/train_00/000123_1.npz   (image_3)
"""
import argparse
import os
import os.path as osp

import cv2
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC


# ---------------------------------------------------------------------------
# Resize / intrinsics helpers (mirrors dataset_setup/ddad/preprocess.py)
# ---------------------------------------------------------------------------
class ImageList:
    def __init__(self, images):
        if not isinstance(images, (tuple, list, set)):
            images = [images]
        self.images = []
        for image in images:
            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def to_pil(self):
        return tuple(self.images) if len(self.images) > 1 else self.images[0]

    @property
    def size(self):
        sizes = [im.size for im in self.images]
        assert all(sizes[0] == s for s in sizes)
        return sizes[0]

    def resize(self, *args, **kwargs):
        return ImageList(self._dispatch('resize', *args, **kwargs))

    def _dispatch(self, func, *args, **kwargs):
        return [getattr(im, func)(*args, **kwargs) for im in self.images]


def opencv_to_colmap_intrinsics(K):
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K


def colmap_to_opencv_intrinsics(K):
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def camera_matrix_of_crop(input_camera_matrix, input_resolution, output_resolution,
                          scaling=1, offset_factor=0.5, offset=None):
    margins = np.asarray(input_resolution) * scaling - output_resolution
    assert np.all(margins >= 0.0)
    if offset is None:
        offset = offset_factor * margins
    out = opencv_to_colmap_intrinsics(input_camera_matrix)
    out[:2, :] *= scaling
    out[:2, 2] -= offset
    return colmap_to_opencv_intrinsics(out)


def rescale_image_depthmap(image, depthmap, camera_intrinsics, output_resolution, force=True):
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (W, H)
    output_resolution = np.array(output_resolution)
    if depthmap is not None:
        assert tuple(depthmap.shape[:2]) == image.size[::-1]

    assert output_resolution.shape == (2,)
    scale_final = max(output_resolution / image.size) + 1e-8
    if scale_final >= 1 and not force:
        return image.to_pil(), depthmap, camera_intrinsics
    output_resolution = np.floor(input_resolution * scale_final).astype(int)

    image = image.resize(tuple(output_resolution),
                         resample=lanczos if scale_final < 1 else bicubic)
    if depthmap is not None:
        depthmap = cv2.resize(depthmap, output_resolution, fx=scale_final,
                              fy=scale_final, interpolation=cv2.INTER_NEAREST)
    intrinsics2 = camera_matrix_of_crop(camera_intrinsics, input_resolution,
                                        output_resolution, scaling=scale_final)
    return image.to_pil(), depthmap, intrinsics2


def geotrf(Trf, pts):
    """Apply 3x3 / 4x4 transform to (N, d) or (N, d) homogeneous-friendly points."""
    pts = np.asarray(pts)
    if Trf.shape[-1] == pts.shape[-1] + 1:
        return pts @ Trf[:pts.shape[-1], :pts.shape[-1]].T + Trf[:pts.shape[-1], -1]
    return pts @ Trf.T


# ---------------------------------------------------------------------------
# KITTI calib / poses parsing
# ---------------------------------------------------------------------------
def parse_calib(calib_path):
    """Parse KITTI odometry calib.txt.

    Returns dict with P0..P3 (3x4) and Tr (4x4 velo->cam0).
    """
    out = {}
    with open(calib_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, vals = line.split(':', 1)
            arr = np.fromstring(vals, sep=' ')
            if key.startswith('P'):
                out[key] = arr.reshape(3, 4)
            elif key == 'Tr':
                Tr = np.eye(4)
                Tr[:3, :4] = arr.reshape(3, 4)
                out['Tr'] = Tr
    assert 'Tr' in out, f"Missing Tr in {calib_path}"
    return out


def parse_poses(poses_path):
    """Parse poses.txt (cam0 in world). Returns (N, 4, 4)."""
    raw = np.loadtxt(poses_path).reshape(-1, 3, 4)
    poses = np.tile(np.eye(4), (raw.shape[0], 1, 1))
    poses[:, :3, :4] = raw
    return poses


def cam_from_P(P):
    """Decompose P = K [I | t_cam_in_rect] to recover K and the cam pose in cam0.

    KITTI rectified projection matrix is P_i = K_i [R_rect | -K_i^{-1} * b]. After
    rectification all R_i = I, so we have P_i[:, :3] = K_i and the camera origin in
    the cam0 (rectified) frame is C_i = -K_i^{-1} * P_i[:, 3].
    cam_i_to_cam0 is therefore [I | C_i].
    """
    K = P[:3, :3].copy()
    t = P[:3, 3].copy()
    C = -np.linalg.inv(K) @ t
    T_cami_to_cam0 = np.eye(4)
    T_cami_to_cam0[:3, 3] = C
    return K, T_cami_to_cam0


# ---------------------------------------------------------------------------
# Depth generation
# ---------------------------------------------------------------------------
def lidar_depth_for_camera(velo_path, T_velo_to_cami, K, image_wh):
    """Project a Velodyne .bin into camera i and return a sparse depth map (HxW, m)."""
    pts = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    pts_cam = geotrf(T_velo_to_cami, pts)  # (N, 3)
    z = pts_cam[:, 2]
    valid = z > 0
    pts_cam = pts_cam[valid]
    z = z[valid]

    uv = (K @ pts_cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    u = np.round(uv[:, 0]).astype(np.int64)
    v = np.round(uv[:, 1]).astype(np.int64)

    W, H = image_wh
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[in_img], v[in_img], z[in_img]
    pts_cam = pts_cam[in_img]

    # If multiple LiDAR points fall on the same pixel, keep the closest one.
    order = np.argsort(-z)  # write farthest first so closest overwrites
    u, v, z = u[order], v[order], z[order]
    pts_cam = pts_cam[order]

    depth = np.zeros((H, W), dtype=np.float32)
    depth[v, u] = z
    # Pixel coords (still on original-resolution grid) and corresponding 3D points
    pixels = np.stack([u, v], axis=-1).astype(np.int64)
    return depth, pts_cam, pixels


# ---------------------------------------------------------------------------
# Per-sequence worker
# ---------------------------------------------------------------------------
KITTI_SPLITS = {
    'train': ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10'],
    'val':   ['08'],
}
CAM_TO_IDX = {'image_2': '0', 'image_3': '1'}
CAM_TO_P_KEY = {'image_2': 'P2', 'image_3': 'P3'}


def process_sequence(seq, split, root, save_dir, target_resolution, pid, nproc):
    seq_dir = osp.join(root, 'sequences', seq)
    calib = parse_calib(osp.join(seq_dir, 'calib.txt'))
    poses_cam0 = parse_poses(osp.join(seq_dir, 'poses.txt'))  # (N, 4, 4)

    Tr_velo_to_cam0 = calib['Tr']

    cam_info = {}
    for cam_name in ('image_2', 'image_3'):
        K_i, T_cami_to_cam0 = cam_from_P(calib[CAM_TO_P_KEY[cam_name]])
        T_cam0_to_cami = np.linalg.inv(T_cami_to_cam0)
        T_velo_to_cami = T_cam0_to_cami @ Tr_velo_to_cam0
        cam_info[cam_name] = dict(
            K=K_i,
            T_cami_to_cam0=T_cami_to_cam0,
            T_velo_to_cami=T_velo_to_cami,
        )

    out_scene_dir = osp.join(save_dir, f"{split}_{seq}")
    os.makedirs(out_scene_dir, exist_ok=True)

    velo_dir = osp.join(seq_dir, 'velodyne')
    n_frames = len(poses_cam0)
    indices = list(range(pid, n_frames, nproc))

    pbar = tqdm(indices, desc=f"{split}_{seq} pid={pid}", leave=False)
    for fi in pbar:
        frame = f"{fi:06d}"
        velo_path = osp.join(velo_dir, frame + '.bin')
        if not osp.isfile(velo_path):
            continue
        pose_cam0 = poses_cam0[fi]

        for cam_name, info in cam_info.items():
            img_path = osp.join(seq_dir, cam_name, frame + '.png')
            if not osp.isfile(img_path):
                continue
            pil = PIL.Image.open(img_path).convert('RGB')
            image = np.array(pil)
            H, W = image.shape[:2]

            depth, pts_cam, pixels = lidar_depth_for_camera(
                velo_path, info['T_velo_to_cami'], info['K'], (W, H))

            output_resolution = (target_resolution, 1) if W > H else (1, target_resolution)
            image_resized, _, intrinsics2 = rescale_image_depthmap(
                image, None, info['K'], output_resolution)

            W2, H2 = image_resized.size
            downscaled_depth = np.zeros((H2, W2), dtype=np.float32)
            if pixels.shape[0] > 0:
                M = intrinsics2 @ np.linalg.inv(info['K'])
                pos2d = geotrf(M, np.concatenate(
                    [pixels.astype(np.float64), np.ones((pixels.shape[0], 1))], axis=1)
                )[:, :2]
                pos2d = np.round(pos2d).astype(np.int64)
                x = np.clip(pos2d[:, 0], 0, W2 - 1)
                y = np.clip(pos2d[:, 1], 0, H2 - 1)
                downscaled_depth[y, x] = pts_cam[:, 2].astype(np.float32)

            cam2world = pose_cam0 @ info['T_cami_to_cam0']

            frame_id = f"{frame}_{CAM_TO_IDX[cam_name]}"
            np.savez_compressed(
                osp.join(out_scene_dir, frame_id + '.npz'),
                image=np.array(image_resized),
                depthmap=downscaled_depth,
                intrinsics=intrinsics2,
                cam2world=cam2world,
            )


# ---------------------------------------------------------------------------
# Torch DataLoader wrapper so we can fan out workers like the DDAD script
# ---------------------------------------------------------------------------
class _SeqJobDataset(torch.utils.data.Dataset):
    def __init__(self, jobs):
        self.jobs = jobs

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, idx):
        process_sequence(**self.jobs[idx])
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, default=0)
    parser.add_argument('--nproc', type=int, default=1)
    parser.add_argument('--n_workers', type=int, default=2)
    parser.add_argument('--kitti_root', type=str,
                        default='/home/dataset-local/lr/code/OccAny/raw_data/semantickitti_occany_root/dataset')
    parser.add_argument('--preprocessed_root', type=str,
                        default='/home/dataset-local/lr/code/OccAny/data/kitti_processed')
    parser.add_argument('--target_resolution', type=int, default=1024)
    args = parser.parse_args()

    os.makedirs(args.preprocessed_root, exist_ok=True)

    jobs = []
    for split, seqs in KITTI_SPLITS.items():
        for seq in seqs:
            jobs.append(dict(
                seq=seq,
                split=split,
                root=args.kitti_root,
                save_dir=args.preprocessed_root,
                target_resolution=args.target_resolution,
                pid=args.pid,
                nproc=args.nproc,
            ))

    dataset = _SeqJobDataset(jobs)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, collate_fn=lambda x: x,
        num_workers=args.n_workers, shuffle=False,
    )
    for _ in tqdm(loader, desc='sequences'):
        pass


if __name__ == '__main__':
    main()
