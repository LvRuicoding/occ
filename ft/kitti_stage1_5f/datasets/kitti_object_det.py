"""KITTI Object detection dataset for Stage-1 OccAny experiments."""
from __future__ import annotations

import math
import os.path as osp
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from occany.utils.helpers import crop_resize_if_necessary
from occany.utils.image_util import ImgNorm

from .kitti_stage1 import _T_cami_from_cam0
from .unified_occ import KITTI_GRID_CONFIG, GridConfig


KITTI_OBJECT_CLASS_NAMES: Tuple[str, ...] = ("Car", "Pedestrian", "Cyclist")
KITTI_OBJECT_NAME_TO_LABEL = {name: i for i, name in enumerate(KITTI_OBJECT_CLASS_NAMES)}
KITTI_OBJECT_LEGACY_DET_PC_RANGE: Tuple[float, float, float, float, float, float] = (
    0.0,
    -25.6,
    -2.0,
    51.2,
    25.6,
    4.4,
)
KITTI_OBJECT_DET_PC_RANGE: Tuple[float, float, float, float, float, float] = (
    0.0,
    -40.0,
    -3.0,
    70.4,
    40.0,
    3.4,
)
KITTI_OBJECT_DET_VOXEL_SIZE: Tuple[float, float, float] = (0.4, 0.4, 0.4)
KITTI_OBJECT_LEGACY_DET_DEPTH_BOUND: Tuple[float, float, float] = (1.0, 52.0, 0.4)
KITTI_OBJECT_DET_DEPTH_BOUND: Tuple[float, float, float] = (1.0, 80.0, 0.4)


def _grid_dim(span: float, voxel: float, name: str) -> int:
    dim = int(round(float(span) / float(voxel)))
    if dim <= 0 or abs(dim * float(voxel) - float(span)) > 1e-4:
        raise ValueError(f"{name} span={span:g} must be divisible by voxel={voxel:g}.")
    return dim


def make_kitti_object_det_grid_config(
    pc_range: Tuple[float, float, float, float, float, float] = KITTI_OBJECT_DET_PC_RANGE,
    voxel_size: Tuple[float, float, float] = KITTI_OBJECT_DET_VOXEL_SIZE,
) -> GridConfig:
    x_min, y_min, z_min, x_max, y_max, z_max = (float(v) for v in pc_range)
    vx, vy, vz = (float(v) for v in voxel_size)
    if not (x_max > x_min and y_max > y_min and z_max > z_min):
        raise ValueError(f"Invalid detection pc_range={pc_range!r}.")

    half_grid = (
        _grid_dim(x_max - x_min, vx, "x"),
        _grid_dim(y_max - y_min, vy, "y"),
        _grid_dim(z_max - z_min, vz, "z"),
    )
    full_size = (vx * 0.5, vy * 0.5, vz * 0.5)
    full_grid = tuple(int(v) * 2 for v in half_grid)
    return GridConfig(
        dataset_name="kitti_object_det",
        full_grid_size=full_grid,
        full_voxel_origin=(x_min, y_min, z_min),
        full_voxel_size=full_size,
        half_grid_size=half_grid,
        half_voxel_origin=(x_min, y_min, z_min),
        half_voxel_size=(vx, vy, vz),
        # LidarImageFusionModule voxelizes in camera coords:
        # x_cam right ~= y_lidar, y_cam down ~= z_lidar, z_cam forward ~= x_lidar.
        fusion_vox_origin=(y_min, z_min, x_min),
        fusion_vox_size=(vy, vz, vx),
        fusion_vox_grid=(half_grid[1], half_grid[2], half_grid[0]),
    )


def _normalize_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _parse_object_calib(calib_path: str) -> Dict[str, np.ndarray]:
    raw: Dict[str, np.ndarray] = {}
    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, vals = line.split(":", 1)
            raw[key] = np.array([float(v) for v in vals.split()], dtype=np.float64)

    P2 = raw["P2"].reshape(3, 4)
    R0 = np.eye(4, dtype=np.float64)
    R0[:3, :3] = raw["R0_rect"].reshape(3, 3)
    Tr = np.eye(4, dtype=np.float64)
    Tr[:3, :4] = raw["Tr_velo_to_cam"].reshape(3, 4)
    T_cam2_from_velo = _T_cami_from_cam0(P2) @ R0 @ Tr
    return {
        "P2": P2,
        "K2": P2[:3, :3].copy(),
        "T_cam_from_velo": T_cam2_from_velo.astype(np.float32),
        "T_velo_from_cam": np.linalg.inv(T_cam2_from_velo).astype(np.float32),
    }


def _read_train_rand(path: str) -> List[int]:
    text = open(path, "r").read().replace("\n", ",")
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _read_mapping(path: str) -> Tuple[List[Tuple[str, str, int]], Dict[Tuple[str, str, int], int]]:
    items: List[Tuple[str, str, int]] = []
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            items.append((parts[0], parts[1], int(parts[2])))
    lookup = {(date, drive, frame): idx for idx, (date, drive, frame) in enumerate(items)}
    return items, lookup


class KittiObject5FrameDetDataset(Dataset):
    """KITTI Object samples with reliable same-drive 5-frame history.

    The target frame is first, followed by historical frames
    ``t - k * frame_stride``. Samples missing any requested historical frame in
    ``mapping/train_mapping.txt`` are filtered out.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        num_frames: int = 5,
        frame_stride: int = 4,
        output_resolution: Tuple[int, int] = (512, 160),
        max_points_per_sweep: int = 0,
        train_count: int = 3712,
        grid_config: GridConfig | None = None,
    ) -> None:
        super().__init__()
        if split not in ("train", "val", "trainval"):
            raise ValueError(f"split must be train/val/trainval, got {split!r}")
        self.root = root
        self.split = split
        self.num_frames = int(num_frames)
        self.frame_stride = int(frame_stride)
        self.output_resolution = (int(output_resolution[0]), int(output_resolution[1]))
        self.max_points_per_sweep = int(max_points_per_sweep)
        self.class_names = KITTI_OBJECT_CLASS_NAMES
        self.grid_config = grid_config if grid_config is not None else KITTI_GRID_CONFIG

        mapping_path = osp.join(root, "mapping", "train_mapping.txt")
        rand_path = osp.join(root, "mapping", "train_rand.txt")
        self.mapping, lookup = _read_mapping(mapping_path)
        shuffled_ids = _read_train_rand(rand_path)
        if split == "train":
            candidate_ids = shuffled_ids[: int(train_count)]
        elif split == "val":
            candidate_ids = shuffled_ids[int(train_count):]
        else:
            candidate_ids = shuffled_ids

        self.samples: List[Tuple[int, Tuple[int, ...]]] = []
        for sample_id in candidate_ids:
            if sample_id < 0 or sample_id >= len(self.mapping):
                continue
            date, drive, frame = self.mapping[sample_id]
            frame_ids: List[int] = []
            ok = True
            for k in range(self.num_frames):
                key = (date, drive, frame - k * self.frame_stride)
                hist_id = lookup.get(key)
                if hist_id is None:
                    ok = False
                    break
                frame_ids.append(int(hist_id))
            if ok and all(self._has_required_files(fid) for fid in frame_ids):
                self.samples.append((int(sample_id), tuple(frame_ids)))
        if not self.samples:
            raise RuntimeError(
                f"No reliable {self.num_frames}-frame KITTI Object samples found "
                f"under {root} for split={split!r} stride={self.frame_stride}."
            )

    def _sample_name(self, sample_id: int) -> str:
        return f"{int(sample_id):06d}"

    def _image_path(self, sample_id: int) -> str:
        return osp.join(self.root, "training", "image_2", f"{self._sample_name(sample_id)}.png")

    def _velodyne_path(self, sample_id: int) -> str:
        return osp.join(self.root, "training", "velodyne", f"{self._sample_name(sample_id)}.bin")

    def _calib_path(self, sample_id: int) -> str:
        return osp.join(self.root, "training", "calib", f"{self._sample_name(sample_id)}.txt")

    def _label_path(self, sample_id: int) -> str:
        return osp.join(self.root, "training", "label_2", f"{self._sample_name(sample_id)}.txt")

    def _has_required_files(self, sample_id: int) -> bool:
        return (
            osp.isfile(self._image_path(sample_id))
            and osp.isfile(self._velodyne_path(sample_id))
            and osp.isfile(self._calib_path(sample_id))
            and osp.isfile(self._label_path(sample_id))
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_view(
        self,
        sample_id: int,
        timestep_index: int,
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        calib = _parse_object_calib(self._calib_path(sample_id))
        image = np.asarray(Image.open(self._image_path(sample_id)).convert("RGB"))
        depth = np.zeros(image.shape[:2], dtype=np.float32)
        img_out, _, K_out = crop_resize_if_necessary(
            Image.fromarray(image),
            depth,
            calib["K2"],
            self.output_resolution,
        )
        img_arr = np.asarray(img_out)
        H, W = img_arr.shape[:2]
        view = dict(
            img=ImgNorm(img_arr),
            true_shape=np.int32((H, W)),
            camera_pose=np.eye(4, dtype=np.float32),
            camera_intrinsics=K_out.astype(np.float32),
            cam2world=np.eye(4, dtype=np.float32),
            timestep=int(timestep_index),
            is_raymap=False,
            is_metric_scale=True,
            frame_id=int(sample_id),
            label=f"kitti_object_{sample_id:06d}_cam2",
        )
        return view, calib["T_cam_from_velo"]

    def _load_points(self, sample_id: int) -> torch.Tensor:
        pts = np.fromfile(self._velodyne_path(sample_id), dtype=np.float32).reshape(-1, 4)
        if self.max_points_per_sweep > 0 and pts.shape[0] > self.max_points_per_sweep:
            idx = np.linspace(0, pts.shape[0] - 1, self.max_points_per_sweep).astype(np.int64)
            pts = pts[idx]
        return torch.from_numpy(pts)

    def _load_labels(self, sample_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        calib = _parse_object_calib(self._calib_path(sample_id))
        T_velo_from_cam = calib["T_velo_from_cam"]
        boxes: List[List[float]] = []
        labels: List[int] = []
        with open(self._label_path(sample_id), "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 15:
                    continue
                name = parts[0]
                if name not in KITTI_OBJECT_NAME_TO_LABEL:
                    continue
                h, w, l = (float(parts[8]), float(parts[9]), float(parts[10]))
                x, y, z = (float(parts[11]), float(parts[12]), float(parts[13]))
                ry = float(parts[14])
                center_cam = np.array([x, y - h * 0.5, z, 1.0], dtype=np.float32)
                center_velo = T_velo_from_cam @ center_cam
                yaw = _normalize_angle(-ry - math.pi * 0.5)
                boxes.append(
                    [
                        float(center_velo[0]),
                        float(center_velo[1]),
                        float(center_velo[2]),
                        float(l),
                        float(w),
                        float(h),
                        float(yaw),
                    ]
                )
                labels.append(KITTI_OBJECT_NAME_TO_LABEL[name])
        if not boxes:
            return torch.zeros((0, 7), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        target_id, frame_ids = self.samples[index]
        views: List[Dict[str, Any]] = []
        Ts: List[np.ndarray] = []
        points: List[torch.Tensor] = []
        for k, sid in enumerate(frame_ids):
            view, T_cam = self._load_view(sid, timestep_index=k)
            views.append(view)
            Ts.append(T_cam.astype(np.float32))
            points.append(self._load_points(sid))

        Ks = np.stack([v["camera_intrinsics"].astype(np.float32) for v in views], axis=0)
        image_hw = np.asarray(views[0]["true_shape"], dtype=np.int32).reshape(2)
        gt_boxes, gt_labels = self._load_labels(target_id)
        grid = self.grid_config.as_tensors()
        target_T_cam = Ts[0]
        target_T_velo_from_cam = np.linalg.inv(target_T_cam).astype(np.float32)
        date, drive, raw_frame = self.mapping[target_id]
        return dict(
            views=views,
            points_per_frame=points,
            T_cam_from_velo=torch.from_numpy(np.stack(Ts, axis=0)),
            T_target_from_refcam=torch.from_numpy(target_T_velo_from_cam),
            K_per_frame=torch.from_numpy(Ks),
            image_hw=torch.from_numpy(image_hw),
            gt_bboxes_3d=gt_boxes,
            gt_labels_3d=gt_labels,
            voxel_origin=grid["voxel_origin"],
            voxel_size=grid["voxel_size"],
            grid_size=grid["grid_size"],
            half_voxel_origin=grid["half_voxel_origin"],
            half_voxel_size=grid["half_voxel_size"],
            half_grid_size=grid["half_grid_size"],
            fusion_vox_origin=grid["fusion_vox_origin"],
            fusion_vox_size=grid["fusion_vox_size"],
            fusion_vox_grid=grid["fusion_vox_grid"],
            dataset_name="kitti_object",
            sample_id=int(target_id),
            frame_ids=tuple(int(f) for f in frame_ids),
            raw_mapping=(date, drive, int(raw_frame)),
        )


def collate_kitti_object_det(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_views = len(batch[0]["views"])
    stacked_views: List[Dict[str, Any]] = []
    for view_idx in range(n_views):
        per_view = [b["views"][view_idx] for b in batch]
        view_dict: Dict[str, Any] = {}
        sample0 = per_view[0]
        for key, value in sample0.items():
            vals = [pv[key] for pv in per_view]
            if isinstance(value, torch.Tensor):
                view_dict[key] = torch.stack(vals, dim=0)
            elif isinstance(value, np.ndarray):
                view_dict[key] = torch.from_numpy(np.stack(vals, axis=0))
            elif isinstance(value, (int, float, bool)):
                view_dict[key] = torch.tensor(vals)
            else:
                view_dict[key] = vals
        stacked_views.append(view_dict)

    return dict(
        views=stacked_views,
        points_per_frame=[b["points_per_frame"] for b in batch],
        T_cam_from_velo=torch.stack([b["T_cam_from_velo"] for b in batch], dim=0),
        T_target_from_refcam=torch.stack([b["T_target_from_refcam"] for b in batch], dim=0),
        K_per_frame=torch.stack([b["K_per_frame"] for b in batch], dim=0),
        image_hw=torch.stack([b["image_hw"] for b in batch], dim=0),
        gt_bboxes_3d=[b["gt_bboxes_3d"] for b in batch],
        gt_labels_3d=[b["gt_labels_3d"] for b in batch],
        voxel_origin=torch.stack([b["voxel_origin"] for b in batch], dim=0),
        voxel_size=torch.stack([b["voxel_size"] for b in batch], dim=0),
        grid_size=torch.stack([b["grid_size"] for b in batch], dim=0),
        half_voxel_origin=torch.stack([b["half_voxel_origin"] for b in batch], dim=0),
        half_voxel_size=torch.stack([b["half_voxel_size"] for b in batch], dim=0),
        half_grid_size=torch.stack([b["half_grid_size"] for b in batch], dim=0),
        fusion_vox_origin=torch.stack([b["fusion_vox_origin"] for b in batch], dim=0),
        fusion_vox_size=torch.stack([b["fusion_vox_size"] for b in batch], dim=0),
        fusion_vox_grid=torch.stack([b["fusion_vox_grid"] for b in batch], dim=0),
        dataset_name=[b.get("dataset_name", "kitti_object") for b in batch],
        sample_id=[b["sample_id"] for b in batch],
        frame_ids=[b["frame_ids"] for b in batch],
        raw_mapping=[b["raw_mapping"] for b in batch],
    )


def _axis_aligned_iou_2d(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = box[0] - box[3] * 0.5
    x2 = box[0] + box[3] * 0.5
    y1 = box[1] - box[4] * 0.5
    y2 = box[1] + box[4] * 0.5
    bx1 = boxes[:, 0] - boxes[:, 3] * 0.5
    bx2 = boxes[:, 0] + boxes[:, 3] * 0.5
    by1 = boxes[:, 1] - boxes[:, 4] * 0.5
    by2 = boxes[:, 1] + boxes[:, 4] * 0.5
    inter_w = np.maximum(0.0, np.minimum(x2, bx2) - np.maximum(x1, bx1))
    inter_h = np.maximum(0.0, np.minimum(y2, by2) - np.maximum(y1, by1))
    inter = inter_w * inter_h
    area = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    barea = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    return inter / np.maximum(area + barea - inter, 1e-6)


def _axis_aligned_iou_3d(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = box[0] - box[3] * 0.5
    x2 = box[0] + box[3] * 0.5
    y1 = box[1] - box[4] * 0.5
    y2 = box[1] + box[4] * 0.5
    z1 = box[2] - box[5] * 0.5
    z2 = box[2] + box[5] * 0.5
    bx1 = boxes[:, 0] - boxes[:, 3] * 0.5
    bx2 = boxes[:, 0] + boxes[:, 3] * 0.5
    by1 = boxes[:, 1] - boxes[:, 4] * 0.5
    by2 = boxes[:, 1] + boxes[:, 4] * 0.5
    bz1 = boxes[:, 2] - boxes[:, 5] * 0.5
    bz2 = boxes[:, 2] + boxes[:, 5] * 0.5
    inter_x = np.maximum(0.0, np.minimum(x2, bx2) - np.maximum(x1, bx1))
    inter_y = np.maximum(0.0, np.minimum(y2, by2) - np.maximum(y1, by1))
    inter_z = np.maximum(0.0, np.minimum(z2, bz2) - np.maximum(z1, bz1))
    inter = inter_x * inter_y * inter_z
    vol = np.maximum(box[3] * box[4] * box[5], 0.0)
    bvol = np.maximum(boxes[:, 3] * boxes[:, 4] * boxes[:, 5], 0.0)
    return inter / np.maximum(vol + bvol - inter, 1e-6)


def _ap40(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    ap = 0.0
    for r in np.linspace(0.0, 1.0, 40):
        mask = recalls >= r
        ap += float(np.max(precisions[mask])) if np.any(mask) else 0.0
    return ap / 40.0


def evaluate_lidar_det_ap40(
    gt_boxes_list: List[torch.Tensor],
    gt_labels_list: List[torch.Tensor],
    pred_boxes_list: List[torch.Tensor],
    pred_scores_list: List[torch.Tensor],
    pred_labels_list: List[torch.Tensor],
) -> Dict[str, float]:
    """Lightweight LiDAR-frame AP40 proxy for branch comparison."""
    thresholds = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}
    stats: Dict[str, float] = {}
    bev_aps: List[float] = []
    ap3d: List[float] = []
    total_gt = 0
    total_pred = 0
    for class_id, class_name in enumerate(KITTI_OBJECT_CLASS_NAMES):
        gt_by_sample = []
        pred_records = []
        n_gt = 0
        for sample_idx, (gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels) in enumerate(
            zip(gt_boxes_list, gt_labels_list, pred_boxes_list, pred_scores_list, pred_labels_list)
        ):
            gt_mask = gt_labels.cpu().numpy() == class_id
            gt_np = gt_boxes.cpu().numpy()[gt_mask]
            gt_by_sample.append(gt_np)
            n_gt += int(gt_np.shape[0])
            pred_mask = pred_labels.cpu().numpy() == class_id
            pboxes = pred_boxes.cpu().numpy()[pred_mask]
            pscores = pred_scores.cpu().numpy()[pred_mask]
            for box, score in zip(pboxes, pscores):
                pred_records.append((float(score), sample_idx, box.astype(np.float32)))
        total_gt += n_gt
        total_pred += len(pred_records)
        pred_records.sort(key=lambda x: x[0], reverse=True)
        if n_gt == 0:
            stats[f"det_bev_ap40_{class_name.lower()}"] = 0.0
            stats[f"det_3d_ap40_{class_name.lower()}"] = 0.0
            continue

        class_bev = []
        class_3d = []
        for iou_fn in (_axis_aligned_iou_2d, _axis_aligned_iou_3d):
            matched = [np.zeros((g.shape[0],), dtype=bool) for g in gt_by_sample]
            tp = np.zeros((len(pred_records),), dtype=np.float32)
            fp = np.zeros_like(tp)
            for i, (_score, sample_idx, box) in enumerate(pred_records):
                gt = gt_by_sample[sample_idx]
                if gt.shape[0] == 0:
                    fp[i] = 1.0
                    continue
                ious = iou_fn(box, gt)
                best = int(np.argmax(ious)) if ious.size else -1
                if best >= 0 and ious[best] >= thresholds[class_name] and not matched[sample_idx][best]:
                    tp[i] = 1.0
                    matched[sample_idx][best] = True
                else:
                    fp[i] = 1.0
            ctp = np.cumsum(tp)
            cfp = np.cumsum(fp)
            recalls = ctp / max(float(n_gt), 1.0)
            precisions = ctp / np.maximum(ctp + cfp, 1e-6)
            if iou_fn is _axis_aligned_iou_2d:
                class_bev.append(_ap40(recalls, precisions))
            else:
                class_3d.append(_ap40(recalls, precisions))
        bev = class_bev[0]
        d3 = class_3d[0]
        stats[f"det_bev_ap40_{class_name.lower()}"] = float(bev)
        stats[f"det_3d_ap40_{class_name.lower()}"] = float(d3)
        bev_aps.append(float(bev))
        ap3d.append(float(d3))

    stats["det_map_bev"] = float(np.mean(bev_aps)) if bev_aps else 0.0
    stats["det_map_3d"] = float(np.mean(ap3d)) if ap3d else 0.0
    stats["det_gt_count"] = float(total_gt)
    stats["det_pred_count"] = float(total_pred)
    return stats


__all__ = [
    "KITTI_OBJECT_CLASS_NAMES",
    "KittiObject5FrameDetDataset",
    "collate_kitti_object_det",
    "evaluate_lidar_det_ap40",
]
