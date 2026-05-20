"""Visualize a few processed KITTI samples to sanity-check GT depth and cam2world.

For each sampled (scene, cam) pair we pick a frame t and a target frame t+gap
in the same scene/camera and produce a single figure with:

  (1) GT depth overlay on RGB at frame t
      -> sanity-checks the LiDAR depth (alignment with the image content,
         scale in meters).
  (2) RGB at frame t+gap.
  (3) Frame t depth back-projected to world, then projected into camera
      at frame t+gap and overlaid on (2).
      -> sanity-checks cam2world: if cam2world_t and cam2world_{t+gap} are
         right, the reprojected points should land on the same physical
         structures (cars, road markings, poles) as in (2).

Output: /home/dataset-local/lr/code/OccAny/visuals/<scene>_<frame_t>_<cam>.png
plus a small text summary "summary.txt".
"""
import argparse
import os
import os.path as osp
import random
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


FRAME_RE = re.compile(r'^(\d{6})_(\d)\.npz$')


def list_scenes(root):
    return sorted([d for d in os.listdir(root)
                   if osp.isdir(osp.join(root, d))])


def list_frames(root, scene):
    """Return dict cam_idx -> sorted list of frame ids (int)."""
    out = {}
    for fn in os.listdir(osp.join(root, scene)):
        m = FRAME_RE.match(fn)
        if not m:
            continue
        fi, cam = int(m.group(1)), m.group(2)
        out.setdefault(cam, []).append(fi)
    for cam in out:
        out[cam].sort()
    return out


def load_sample(root, scene, frame_id, cam):
    fn = osp.join(root, scene, f"{frame_id:06d}_{cam}.npz")
    d = np.load(fn)
    return dict(image=d['image'], depth=d['depthmap'],
                K=d['intrinsics'], cam2world=d['cam2world'])


def backproject(depth, K):
    H, W = depth.shape
    v, u = np.indices((H, W))
    valid = depth > 0
    u = u[valid]; v = v[valid]; z = depth[valid]
    Kinv = np.linalg.inv(K)
    pix = np.stack([u, v, np.ones_like(u)], axis=-1).astype(np.float64)  # (N, 3)
    rays = pix @ Kinv.T
    pts_cam = rays * z[:, None]
    return pts_cam, z


def project(pts_cam, K, image_wh):
    z = pts_cam[:, 2]
    front = z > 0
    pts_cam = pts_cam[front]
    z = z[front]
    uv = (pts_cam @ K.T)
    uv = uv[:, :2] / uv[:, 2:3]
    W, H = image_wh
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return uv[inb], z[inb]


def overlay_depth(ax, image, depth, title, vmax=None):
    ax.imshow(image)
    valid = depth > 0
    if valid.any():
        v, u = np.where(valid)
        d = depth[valid]
        if vmax is None:
            vmax = float(np.percentile(d, 95))
        ax.scatter(u, v, c=d, s=1.0, cmap='turbo',
                   vmin=0.5, vmax=vmax, alpha=0.85)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def overlay_points(ax, image, uv, z, title, vmax=None):
    ax.imshow(image)
    if uv.shape[0] > 0:
        if vmax is None:
            vmax = float(np.percentile(z, 95))
        ax.scatter(uv[:, 0], uv[:, 1], c=z, s=1.0, cmap='turbo',
                   vmin=0.5, vmax=vmax, alpha=0.85)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def visualize_sample(root, out_dir, scene, frame_t, cam, gap):
    s_t = load_sample(root, scene, frame_t, cam)
    frame_q = frame_t + gap
    s_q = load_sample(root, scene, frame_q, cam)

    # Back-project frame t into world and forward-project into frame q.
    pts_cam_t, z_t = backproject(s_t['depth'], s_t['K'])
    pts_h = np.concatenate([pts_cam_t, np.ones((pts_cam_t.shape[0], 1))], axis=1)
    pts_world = pts_h @ s_t['cam2world'].T
    pts_q_cam = pts_world @ np.linalg.inv(s_q['cam2world']).T
    pts_q_cam = pts_q_cam[:, :3]

    H_q, W_q = s_q['image'].shape[:2]
    uv, z = project(pts_q_cam, s_q['K'], (W_q, H_q))

    # Pose translation between the two frames (sanity number).
    dpose = np.linalg.inv(s_t['cam2world']) @ s_q['cam2world']
    dt = np.linalg.norm(dpose[:3, 3])

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    overlay_depth(axes[0], s_t['image'], s_t['depth'],
                  f"{scene} cam{cam} frame {frame_t:06d}: GT depth overlay")
    axes[1].imshow(s_q['image']); axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].set_title(f"{scene} cam{cam} frame {frame_q:06d}: target image", fontsize=9)
    overlay_points(axes[2], s_q['image'], uv, z,
                   f"reprojection of frame {frame_t:06d} -> {frame_q:06d} "
                   f"(translation {dt:.2f} m)")
    fig.tight_layout()

    out_path = osp.join(out_dir, f"{scene}_{frame_t:06d}_cam{cam}.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    return dict(out=out_path, scene=scene, cam=cam,
                frame_t=frame_t, frame_q=frame_q,
                translation=float(dt),
                n_depth_points=int((s_t['depth'] > 0).sum()),
                n_reprojected=int(uv.shape[0]))


def pick_samples(root, n, gap, seed):
    rng = random.Random(seed)
    scenes = list_scenes(root)
    samples = []
    attempts = 0
    while len(samples) < n and attempts < 500:
        attempts += 1
        scene = rng.choice(scenes)
        cams = list_frames(root, scene)
        if not cams:
            continue
        cam = rng.choice(list(cams.keys()))
        frames = cams[cam]
        if len(frames) < gap + 1:
            continue
        frame_t = rng.choice(frames[:-gap])
        if (frame_t + gap) not in cams[cam]:
            continue
        samples.append((scene, frame_t, cam))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        default='/home/dataset-local/lr/code/OccAny/data/kitti_processed')
    parser.add_argument('--out_dir', type=str,
                        default='/home/dataset-local/lr/code/OccAny/visuals')
    parser.add_argument('--n', type=int, default=6)
    parser.add_argument('--gap', type=int, default=10,
                        help='frame gap for reprojection sanity check')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    samples = pick_samples(args.root, args.n, args.gap, args.seed)
    if not samples:
        raise SystemExit('no samples found')

    summary = []
    for scene, frame_t, cam in samples:
        info = visualize_sample(args.root, args.out_dir, scene, frame_t, cam, args.gap)
        summary.append(info)
        print(f"saved {info['out']}  | trans={info['translation']:.2f}m"
              f"  depth_pts={info['n_depth_points']}  reproj_pts={info['n_reprojected']}")

    with open(osp.join(args.out_dir, 'summary.txt'), 'w') as f:
        for s in summary:
            f.write(f"{s['scene']}  cam{s['cam']}  frame {s['frame_t']:06d}->{s['frame_q']:06d}  "
                    f"translation={s['translation']:.3f}m  "
                    f"depth_pts={s['n_depth_points']}  reproj_pts={s['n_reprojected']}\n")


if __name__ == '__main__':
    main()
