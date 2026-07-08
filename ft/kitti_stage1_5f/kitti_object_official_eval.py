"""Pure-Python KITTI Object AP_R40 evaluation.

This mirrors the KITTI/OpenMMLab evaluation policy for Car/Pedestrian/Cyclist:
easy/moderate/hard filtering, DontCare suppression for 2D bbox AP, strict and
loose overlap settings, and AP_R40 interpolation. It avoids the numba/CUDA
dependency used by the OpenMMLab evaluator.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box as shapely_box


CLASS_NAMES: Tuple[str, ...] = ("Car", "Pedestrian", "Cyclist")
DIFFICULTIES: Tuple[str, ...] = ("easy", "moderate", "hard")
MIN_HEIGHT = (40.0, 25.0, 25.0)
MAX_OCCLUSION = (0, 1, 2)
MAX_TRUNCATION = (0.15, 0.30, 0.50)


def _get_thresholds(scores: np.ndarray, num_gt: int, num_sample_pts: int = 41) -> List[float]:
    if scores.size == 0 or num_gt <= 0:
        return []
    scores = np.sort(scores)[::-1]
    thresholds: List[float] = []
    current_recall = 0.0
    for i, score in enumerate(scores):
        l_recall = float(i + 1) / float(num_gt)
        r_recall = float(i + 2) / float(num_gt) if i < len(scores) - 1 else l_recall
        if ((r_recall - current_recall) < (current_recall - l_recall)) and i < len(scores) - 1:
            continue
        thresholds.append(float(score))
        current_recall += 1.0 / float(num_sample_pts - 1)
    return thresholds[:num_sample_pts]


def _bbox_overlap(boxes: np.ndarray, qboxes: np.ndarray, criterion: int = -1) -> np.ndarray:
    out = np.zeros((boxes.shape[0], qboxes.shape[0]), dtype=np.float64)
    for i, b in enumerate(boxes):
        bw = max(float(b[2] - b[0]), 0.0)
        bh = max(float(b[3] - b[1]), 0.0)
        b_area = bw * bh
        for j, q in enumerate(qboxes):
            iw = max(min(float(b[2]), float(q[2])) - max(float(b[0]), float(q[0])), 0.0)
            ih = max(min(float(b[3]), float(q[3])) - max(float(b[1]), float(q[1])), 0.0)
            inter = iw * ih
            if inter <= 0.0:
                continue
            q_area = max(float(q[2] - q[0]), 0.0) * max(float(q[3] - q[1]), 0.0)
            if criterion == -1:
                denom = b_area + q_area - inter
            elif criterion == 0:
                denom = b_area
            elif criterion == 1:
                denom = q_area
            else:
                denom = inter
            out[i, j] = inter / max(denom, 1e-12)
    return out


def _bev_poly(box7: np.ndarray) -> Polygon:
    # box7 is camera layout: x, y, z, h, w, l, rotation_y.
    x, z = float(box7[0]), float(box7[2])
    w, l, ry = float(box7[4]), float(box7[5]), float(box7[6])
    poly = shapely_box(-w * 0.5, -l * 0.5, w * 0.5, l * 0.5)
    poly = rotate(poly, -ry * 180.0 / np.pi, origin=(0.0, 0.0), use_radians=False)
    return translate(poly, xoff=x, yoff=z)


def _bev_overlap(dt_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    out = np.zeros((dt_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float64)
    dt_polys = [_bev_poly(b) for b in dt_boxes]
    gt_polys = [_bev_poly(b) for b in gt_boxes]
    for i, dp in enumerate(dt_polys):
        d_area = dp.area
        for j, gp in enumerate(gt_polys):
            inter = dp.intersection(gp).area
            if inter <= 0.0:
                continue
            out[i, j] = inter / max(d_area + gp.area - inter, 1e-12)
    return out


def _d3_overlap(dt_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    out = np.zeros((dt_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float64)
    dt_polys = [_bev_poly(b) for b in dt_boxes]
    gt_polys = [_bev_poly(b) for b in gt_boxes]
    for i, (db, dp) in enumerate(zip(dt_boxes, dt_polys)):
        d_top = float(db[1] - db[3])
        d_bottom = float(db[1])
        d_vol = max(dp.area * float(db[3]), 0.0)
        for j, (gb, gp) in enumerate(zip(gt_boxes, gt_polys)):
            inter_bev = dp.intersection(gp).area
            if inter_bev <= 0.0:
                continue
            g_top = float(gb[1] - gb[3])
            g_bottom = float(gb[1])
            inter_h = max(min(d_bottom, g_bottom) - max(d_top, g_top), 0.0)
            inter = inter_bev * inter_h
            g_vol = max(gp.area * float(gb[3]), 0.0)
            out[i, j] = inter / max(d_vol + g_vol - inter, 1e-12)
    return out


def _anno_camera_boxes(anno: Dict) -> np.ndarray:
    if len(anno["name"]) == 0:
        return np.zeros((0, 7), dtype=np.float64)
    return np.concatenate(
        [
            np.asarray(anno["location"], dtype=np.float64),
            np.asarray(anno["dimensions"], dtype=np.float64),
            np.asarray(anno["rotation_y"], dtype=np.float64).reshape(-1, 1),
        ],
        axis=1,
    )


def _overlap_matrix(gt_anno: Dict, dt_anno: Dict, metric: str) -> np.ndarray:
    if len(gt_anno["name"]) == 0 or len(dt_anno["name"]) == 0:
        return np.zeros((len(dt_anno["name"]), len(gt_anno["name"])), dtype=np.float64)
    if metric == "bbox":
        return _bbox_overlap(
            np.asarray(dt_anno["bbox"], dtype=np.float64),
            np.asarray(gt_anno["bbox"], dtype=np.float64),
            criterion=-1,
        )
    gt_boxes = _anno_camera_boxes(gt_anno)
    dt_boxes = _anno_camera_boxes(dt_anno)
    if metric == "bev":
        return _bev_overlap(dt_boxes, gt_boxes)
    if metric == "3d":
        return _d3_overlap(dt_boxes, gt_boxes)
    raise ValueError(f"Unsupported metric={metric!r}.")


def _clean_data(gt_anno: Dict, dt_anno: Dict, class_name: str, difficulty: int):
    current = class_name.lower()
    ignored_gt: List[int] = []
    ignored_dt: List[int] = []
    dc_bboxes: List[np.ndarray] = []
    num_valid_gt = 0

    for i, gt_name_raw in enumerate(gt_anno["name"]):
        gt_name = str(gt_name_raw).lower()
        bbox = gt_anno["bbox"][i]
        height = float(bbox[3] - bbox[1])
        if gt_name == current:
            valid_class = 1
        elif current == "pedestrian" and gt_name == "person_sitting":
            valid_class = 0
        elif current == "car" and gt_name == "van":
            valid_class = 0
        else:
            valid_class = -1

        ignore = (
            int(gt_anno["occluded"][i]) > MAX_OCCLUSION[difficulty]
            or float(gt_anno["truncated"][i]) > MAX_TRUNCATION[difficulty]
            or height <= MIN_HEIGHT[difficulty]
        )
        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif valid_class == 0 or (valid_class == 1 and ignore):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
        if str(gt_name_raw) == "DontCare":
            dc_bboxes.append(np.asarray(bbox, dtype=np.float64))

    for i, dt_name_raw in enumerate(dt_anno["name"]):
        dt_name = str(dt_name_raw).lower()
        bbox = dt_anno["bbox"][i]
        height = abs(float(bbox[3] - bbox[1]))
        valid_class = 1 if dt_name == current else -1
        if height < MIN_HEIGHT[difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    dc = (
        np.stack(dc_bboxes, axis=0).astype(np.float64)
        if dc_bboxes
        else np.zeros((0, 4), dtype=np.float64)
    )
    return num_valid_gt, np.asarray(ignored_gt, dtype=np.int64), np.asarray(ignored_dt, dtype=np.int64), dc


def _compute_statistics(
    overlaps: np.ndarray,
    gt_anno: Dict,
    dt_anno: Dict,
    ignored_gt: np.ndarray,
    ignored_dt: np.ndarray,
    dc_bboxes: np.ndarray,
    metric: str,
    min_overlap: float,
    thresh: float = 0.0,
    compute_fp: bool = False,
):
    det_size = len(dt_anno["name"])
    gt_size = len(gt_anno["name"])
    dt_scores = np.asarray(dt_anno.get("score", np.zeros((det_size,), dtype=np.float64)), dtype=np.float64)
    assigned = np.zeros((det_size,), dtype=bool)
    tp = fp = fn = 0
    thresholds: List[float] = []
    no_detection = -1e10

    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = no_detection
        max_overlap = 0.0
        assigned_ignored_det = False
        for j in range(det_size):
            if ignored_dt[j] == -1 or assigned[j]:
                continue
            if compute_fp and dt_scores[j] < thresh:
                continue
            overlap = overlaps[j, i] if overlaps.size else 0.0
            score = dt_scores[j]
            if (not compute_fp) and overlap > min_overlap and score > valid_detection:
                det_idx = j
                valid_detection = score
            elif (
                compute_fp
                and overlap > min_overlap
                and (overlap > max_overlap or assigned_ignored_det)
                and ignored_dt[j] == 0
            ):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1.0
                assigned_ignored_det = False
            elif (
                compute_fp
                and overlap > min_overlap
                and valid_detection == no_detection
                and ignored_dt[j] == 1
            ):
                det_idx = j
                valid_detection = 1.0
                assigned_ignored_det = True

        if valid_detection == no_detection and ignored_gt[i] == 0:
            fn += 1
        elif valid_detection != no_detection and (ignored_gt[i] == 1 or ignored_dt[det_idx] == 1):
            assigned[det_idx] = True
        elif valid_detection != no_detection:
            tp += 1
            thresholds.append(float(dt_scores[det_idx]))
            assigned[det_idx] = True

    if compute_fp:
        for i in range(det_size):
            if assigned[i] or ignored_dt[i] in (-1, 1) or dt_scores[i] < thresh:
                continue
            fp += 1
        if metric == "bbox" and dc_bboxes.shape[0] > 0 and det_size > 0:
            dc_overlap = _bbox_overlap(np.asarray(dt_anno["bbox"], dtype=np.float64), dc_bboxes, criterion=0)
            suppressed = 0
            for dc_idx in range(dc_bboxes.shape[0]):
                for dt_idx in range(det_size):
                    if assigned[dt_idx] or ignored_dt[dt_idx] in (-1, 1) or dt_scores[dt_idx] < thresh:
                        continue
                    if dc_overlap[dt_idx, dc_idx] > min_overlap:
                        assigned[dt_idx] = True
                        suppressed += 1
            fp -= suppressed
    return tp, fp, fn, thresholds


def _eval_one(
    gt_annos: Sequence[Dict],
    dt_annos: Sequence[Dict],
    class_name: str,
    difficulty: int,
    metric: str,
    min_overlap: float,
) -> float:
    cleaned = [
        _clean_data(gt, dt, class_name, difficulty)
        for gt, dt in zip(gt_annos, dt_annos)
    ]
    total_valid_gt = sum(item[0] for item in cleaned)
    if total_valid_gt <= 0:
        return 0.0

    overlaps = [_overlap_matrix(gt, dt, metric) for gt, dt in zip(gt_annos, dt_annos)]
    threshold_scores: List[float] = []
    for ov, gt, dt, (_n, ignored_gt, ignored_dt, dc_bboxes) in zip(overlaps, gt_annos, dt_annos, cleaned):
        _tp, _fp, _fn, scores = _compute_statistics(
            ov,
            gt,
            dt,
            ignored_gt,
            ignored_dt,
            dc_bboxes,
            metric,
            min_overlap,
            thresh=0.0,
            compute_fp=False,
        )
        threshold_scores.extend(scores)
    thresholds = _get_thresholds(np.asarray(threshold_scores, dtype=np.float64), total_valid_gt)
    precision = np.zeros((41,), dtype=np.float64)

    for tidx, thresh in enumerate(thresholds[:41]):
        tp = fp = fn = 0
        for ov, gt, dt, (_n, ignored_gt, ignored_dt, dc_bboxes) in zip(overlaps, gt_annos, dt_annos, cleaned):
            stp, sfp, sfn, _scores = _compute_statistics(
                ov,
                gt,
                dt,
                ignored_gt,
                ignored_dt,
                dc_bboxes,
                metric,
                min_overlap,
                thresh=thresh,
                compute_fp=True,
            )
            tp += stp
            fp += sfp
            fn += sfn
        precision[tidx] = float(tp) / max(float(tp + fp), 1e-12)
        _recall = float(tp) / max(float(tp + fn), 1e-12)

    valid_len = min(len(thresholds), 41)
    for i in range(valid_len):
        precision[i] = np.max(precision[i:valid_len])
    return float(np.sum(precision[1:]) / 40.0 * 100.0)


def kitti_object_eval(
    gt_annos: Sequence[Dict],
    dt_annos: Sequence[Dict],
    current_classes: Iterable[str] = CLASS_NAMES,
    eval_types: Sequence[str] = ("bbox", "bev", "3d"),
) -> Tuple[str, Dict[str, float]]:
    current_classes = tuple(current_classes)
    strict = {"bbox": (0.7, 0.5, 0.5), "bev": (0.7, 0.5, 0.5), "3d": (0.7, 0.5, 0.5)}
    loose = {"bbox": (0.7, 0.5, 0.5), "bev": (0.5, 0.25, 0.25), "3d": (0.5, 0.25, 0.25)}
    overlap_sets = (("strict", strict), ("loose", loose))
    lines: List[str] = ["", "----------- AP40 Results ------------", ""]
    ret: Dict[str, float] = {}

    for cls_name in current_classes:
        cls_idx = CLASS_NAMES.index(cls_name)
        for set_name, overlaps in overlap_sets:
            lines.append(
                f"{cls_name} AP40@"
                f"{overlaps['bbox'][cls_idx]:.2f}, {overlaps['bev'][cls_idx]:.2f}, "
                f"{overlaps['3d'][cls_idx]:.2f} ({set_name}):"
            )
            for metric in eval_types:
                vals = [
                    _eval_one(gt_annos, dt_annos, cls_name, diff_idx, metric, overlaps[metric][cls_idx])
                    for diff_idx in range(3)
                ]
                ret_name = "2D" if metric == "bbox" else ("BEV" if metric == "bev" else "3D")
                for diff_name, value in zip(DIFFICULTIES, vals):
                    ret[f"KITTI/{cls_name}_{ret_name}_AP40_{diff_name}_{set_name}"] = value
                lines.append(f"{metric:<4} AP40:{vals[0]:.4f}, {vals[1]:.4f}, {vals[2]:.4f}")
            lines.append("")

    if len(current_classes) > 1:
        for set_name, _overlaps in overlap_sets:
            lines.append(f"Overall AP40 ({set_name}):")
            for metric in eval_types:
                ret_name = "2D" if metric == "bbox" else ("BEV" if metric == "bev" else "3D")
                vals = []
                for diff_name in DIFFICULTIES:
                    keys = [
                        f"KITTI/{cls_name}_{ret_name}_AP40_{diff_name}_{set_name}"
                        for cls_name in current_classes
                    ]
                    vals.append(float(np.mean([ret[k] for k in keys])))
                    ret[f"KITTI/Overall_{ret_name}_AP40_{diff_name}_{set_name}"] = vals[-1]
                lines.append(f"{metric:<4} AP40:{vals[0]:.4f}, {vals[1]:.4f}, {vals[2]:.4f}")
            lines.append("")
    return "\n".join(lines), ret


__all__ = ["kitti_object_eval"]
