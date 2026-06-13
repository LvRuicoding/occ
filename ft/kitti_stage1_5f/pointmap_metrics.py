"""Pointmap quality metrics shared by Stage-1 training/evaluation tools."""
from __future__ import annotations

import argparse
import math
from typing import Dict, List, Optional, Tuple

import torch

from .models.stage1_ssc_bevdetocc_lidar_pointmap import _pointmap_targets_from_depth


def add_pointmap_metric_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--pointmap_eval_chamfer_max_points", type=int, default=8192)
    p.add_argument("--pointmap_eval_chamfer_chunk_size", type=int, default=2048)
    p.add_argument("--pointmap_eval_fscore_threshold", type=float, default=0.2)
    p.add_argument(
        "--pointmap_eval_cross_view_pairs",
        choices=["adjacent", "all", "none"],
        default="adjacent",
    )
    p.add_argument("--pointmap_eval_cross_view_max_points", type=int, default=4096)
    p.add_argument("--pointmap_eval_stat_sample_max_points", type=int, default=2_000_000)
    p.add_argument("--pointmap_eval_confidence_max_points", type=int, default=2_000_000)
    p.add_argument("--pointmap_eval_confidence_good_threshold", type=float, default=None)


def _metric_arg(args, short_name: str, default):
    prefixed = f"pointmap_eval_{short_name}"
    if hasattr(args, prefixed):
        return getattr(args, prefixed)
    return getattr(args, short_name, default)


class Reservoir:
    """Approximate streaming sample buffer for median-like statistics."""

    def __init__(self, max_size: int, seed: int = 0) -> None:
        self.max_size = int(max_size)
        self.values: Optional[torch.Tensor] = None
        self.seen = 0
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(int(seed))

    def add(self, values: torch.Tensor) -> None:
        if self.max_size <= 0:
            self.seen += int(values.numel())
            return
        v = values.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        n = int(v.numel())
        if n == 0:
            return
        if self.values is None:
            take = min(self.max_size, n)
            if take < n:
                idx = torch.randperm(n, generator=self.gen)[:take]
                self.values = v[idx].clone()
            else:
                self.values = v.clone()
            self.seen += n
            return

        cur = int(self.values.numel())
        if cur < self.max_size:
            take = min(self.max_size - cur, n)
            self.values = torch.cat([self.values, v[:take].clone()], dim=0)
            v = v[take:]
            n = int(v.numel())
            self.seen += take
            if n == 0:
                return

        positions = torch.arange(
            self.seen + 1,
            self.seen + n + 1,
            dtype=torch.float32,
        )
        keep = torch.rand(n, generator=self.gen) < (float(self.max_size) / positions)
        if bool(keep.any()):
            dst = torch.randint(self.max_size, (int(keep.sum()),), generator=self.gen)
            self.values[dst] = v[keep]
        self.seen += n

    def median(self) -> float:
        if self.values is None or self.values.numel() == 0:
            return float("nan")
        return float(torch.median(self.values).item())


class PairReservoir:
    """Reservoir for paired confidence/error samples."""

    def __init__(self, max_size: int, seed: int = 0) -> None:
        self.max_size = int(max_size)
        self.conf: Optional[torch.Tensor] = None
        self.err: Optional[torch.Tensor] = None
        self.seen = 0
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(int(seed))

    def add(self, conf: torch.Tensor, err: torch.Tensor) -> None:
        if self.max_size <= 0:
            self.seen += int(err.numel())
            return
        c = conf.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        e = err.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        valid = torch.isfinite(c) & torch.isfinite(e)
        c = c[valid]
        e = e[valid]
        n = int(e.numel())
        if n == 0:
            return
        if self.err is None:
            take = min(self.max_size, n)
            if take < n:
                idx = torch.randperm(n, generator=self.gen)[:take]
                c = c[idx]
                e = e[idx]
            self.conf = c[:take].clone()
            self.err = e[:take].clone()
            self.seen += n
            return

        cur = int(self.err.numel())
        if cur < self.max_size:
            take = min(self.max_size - cur, n)
            self.conf = torch.cat([self.conf, c[:take].clone()], dim=0)
            self.err = torch.cat([self.err, e[:take].clone()], dim=0)
            c = c[take:]
            e = e[take:]
            n = int(e.numel())
            self.seen += take
            if n == 0:
                return

        positions = torch.arange(
            self.seen + 1,
            self.seen + n + 1,
            dtype=torch.float32,
        )
        keep = torch.rand(n, generator=self.gen) < (float(self.max_size) / positions)
        if bool(keep.any()):
            dst = torch.randint(self.max_size, (int(keep.sum()),), generator=self.gen)
            self.conf[dst] = c[keep]
            self.err[dst] = e[keep]
        self.seen += n

    def aucs(self, good_threshold: float) -> Dict[str, float]:
        if self.conf is None or self.err is None or self.err.numel() < 2:
            return {}
        order = torch.argsort(self.conf, descending=True)
        err = self.err[order]
        coverage = torch.arange(1, err.numel() + 1, dtype=torch.float32) / float(err.numel())
        risk = torch.cumsum(err, dim=0) / torch.arange(1, err.numel() + 1, dtype=torch.float32)
        confidence_auc = float(torch.trapz(risk, coverage).item())

        labels = (self.err <= float(good_threshold)).to(torch.int64)
        n_pos = int(labels.sum().item())
        n_neg = int(labels.numel() - n_pos)
        roc_auc = float("nan")
        if n_pos > 0 and n_neg > 0:
            order_asc = torch.argsort(self.conf)
            ranks = torch.empty_like(order_asc, dtype=torch.float32)
            ranks[order_asc] = torch.arange(1, labels.numel() + 1, dtype=torch.float32)
            pos_rank_sum = ranks[labels.bool()].sum()
            roc_auc = float(((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)).item())

        return {
            "confidence_auc": confidence_auc,
            "confidence_roc_auc": roc_auc,
            "confidence_samples": float(self.err.numel()),
            "confidence_good_threshold": float(good_threshold),
        }


class PointmapMetricAccumulator:
    _STATE_SCALARS = (
        "count",
        "pts_l1_sum",
        "pts_l2_sum",
        "scale_count",
        "scale_l1_sum",
        "scale_l2_sum",
        "depth_count",
        "depth_absrel_sum",
        "depth_sq_sum",
        "depth_delta_sum",
        "reproj_count",
        "reproj_sum",
        "chamfer_f_sum",
        "chamfer_b_sum",
        "chamfer_f_count",
        "chamfer_b_count",
        "fscore_prec_hits",
        "fscore_rec_hits",
        "cross_count",
        "cross_sum",
    )

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        seed = int(getattr(args, "seed", 0))
        self.count = 0.0
        self.pts_l1_sum = 0.0
        self.pts_l2_sum = 0.0
        self.pts_l2_med = Reservoir(
            _metric_arg(args, "stat_sample_max_points", 2_000_000),
            seed + 11,
        )

        self.scale_count = 0.0
        self.scale_l1_sum = 0.0
        self.scale_l2_sum = 0.0
        self.scale_l2_med = Reservoir(
            _metric_arg(args, "stat_sample_max_points", 2_000_000),
            seed + 12,
        )

        self.depth_count = 0.0
        self.depth_absrel_sum = 0.0
        self.depth_sq_sum = 0.0
        self.depth_delta_sum = 0.0

        self.reproj_count = 0.0
        self.reproj_sum = 0.0
        self.reproj_med = Reservoir(
            _metric_arg(args, "stat_sample_max_points", 2_000_000),
            seed + 13,
        )

        self.chamfer_f_sum = 0.0
        self.chamfer_b_sum = 0.0
        self.chamfer_f_count = 0.0
        self.chamfer_b_count = 0.0
        self.fscore_prec_hits = 0.0
        self.fscore_rec_hits = 0.0

        self.cross_count = 0.0
        self.cross_sum = 0.0
        self.cross_med = Reservoir(
            _metric_arg(args, "stat_sample_max_points", 2_000_000),
            seed + 14,
        )

        self.conf_pairs = PairReservoir(
            _metric_arg(args, "confidence_max_points", 2_000_000),
            seed + 15,
        )

    def update_point_errors(
        self,
        pred_ref: torch.Tensor,
        gt_ref: torch.Tensor,
        valid: torch.Tensor,
        pred_conf: Optional[torch.Tensor],
    ) -> None:
        diff = pred_ref.float() - gt_ref.float()
        finite = torch.isfinite(diff).all(dim=-1)
        mask = valid & finite
        if not bool(mask.any().item()):
            return
        l1 = diff.abs().sum(dim=-1)[mask]
        l2 = torch.linalg.norm(diff, dim=-1)[mask]
        self.count += float(l2.numel())
        self.pts_l1_sum += float(l1.sum().item())
        self.pts_l2_sum += float(l2.sum().item())
        self.pts_l2_med.add(l2)
        if pred_conf is not None:
            self.conf_pairs.add(pred_conf[mask], l2)

    def update_scale_aligned(
        self,
        pred_ref: torch.Tensor,
        gt_ref: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        pred = pred_ref.float()
        gt = gt_ref.float()
        finite = torch.isfinite(pred).all(dim=-1) & torch.isfinite(gt).all(dim=-1)
        mask = valid & finite
        if not bool(mask.any().item()):
            return
        mask_f = mask.to(dtype=pred.dtype)
        dot = (pred * gt).sum(dim=-1)
        denom = (pred * pred).sum(dim=-1)
        dims = tuple(range(1, dot.ndim))
        dot_sum = (dot * mask_f).sum(dim=dims, keepdim=True)
        denom_sum = (denom * mask_f).sum(dim=dims, keepdim=True).clamp(min=1e-8)
        scale = (dot_sum / denom_sum).clamp(min=1e-6, max=1e6)
        diff = pred * scale.unsqueeze(-1) - gt
        l1 = diff.abs().sum(dim=-1)[mask]
        l2 = torch.linalg.norm(diff, dim=-1)[mask]
        self.scale_count += float(l2.numel())
        self.scale_l1_sum += float(l1.sum().item())
        self.scale_l2_sum += float(l2.sum().item())
        self.scale_l2_med.add(l2)

    def update_depth(
        self,
        pred_local: torch.Tensor,
        dense_depth: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        pred_z = pred_local[..., 2].float()
        gt_z = dense_depth.float()
        mask = valid & torch.isfinite(pred_z) & torch.isfinite(gt_z) & (gt_z > 0.0) & (pred_z > 0.0)
        if not bool(mask.any().item()):
            return
        p = pred_z[mask]
        g = gt_z[mask]
        abs_rel = (p - g).abs() / g.clamp(min=1e-6)
        sq = (p - g).pow(2)
        ratio = torch.maximum(p / g.clamp(min=1e-6), g / p.clamp(min=1e-6))
        self.depth_count += float(p.numel())
        self.depth_absrel_sum += float(abs_rel.sum().item())
        self.depth_sq_sum += float(sq.sum().item())
        self.depth_delta_sum += float((ratio < 1.25).to(torch.float32).sum().item())

    def update_reprojection(
        self,
        pred_local: torch.Tensor,
        K_per_frame: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        B, N, H, W, _ = pred_local.shape
        device = pred_local.device
        ys = torch.arange(H, device=device, dtype=torch.float32)
        xs = torch.arange(W, device=device, dtype=torch.float32)
        v0, u0 = torch.meshgrid(ys, xs, indexing="ij")
        u0 = u0.view(1, 1, H, W)
        v0 = v0.view(1, 1, H, W)

        pts = pred_local.float()
        z = pts[..., 2]
        K = K_per_frame.to(device=device, dtype=torch.float32)
        fx = K[..., 0, 0].view(B, N, 1, 1)
        fy = K[..., 1, 1].view(B, N, 1, 1)
        cx = K[..., 0, 2].view(B, N, 1, 1)
        cy = K[..., 1, 2].view(B, N, 1, 1)
        u = pts[..., 0] / z.clamp(min=1e-6) * fx + cx
        v = pts[..., 1] / z.clamp(min=1e-6) * fy + cy
        err = torch.sqrt((u - u0).pow(2) + (v - v0).pow(2))
        mask = valid & torch.isfinite(err) & torch.isfinite(pts).all(dim=-1) & (z > 1e-6)
        if not bool(mask.any().item()):
            return
        vals = err[mask]
        self.reproj_count += float(vals.numel())
        self.reproj_sum += float(vals.sum().item())
        self.reproj_med.add(vals)

    def update_chamfer(
        self,
        pred_ref: torch.Tensor,
        gt_ref: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        max_points = int(_metric_arg(self.args, "chamfer_max_points", 8192))
        if max_points <= 0:
            return
        B, N = pred_ref.shape[:2]
        gen = torch.Generator(device=pred_ref.device)
        gen.manual_seed(int(getattr(self.args, "seed", 0)) + int(self.count) % 1000003)
        for b in range(B):
            for n in range(N):
                mask = (
                    valid[b, n]
                    & torch.isfinite(pred_ref[b, n]).all(dim=-1)
                    & torch.isfinite(gt_ref[b, n]).all(dim=-1)
                )
                if not bool(mask.any().item()):
                    continue
                pred = _sample_points(pred_ref[b, n][mask].float(), max_points, gen)
                gt = _sample_points(gt_ref[b, n][mask].float(), max_points, gen)
                if pred.numel() == 0 or gt.numel() == 0:
                    continue
                d_pg = _nearest_distances(
                    pred,
                    gt,
                    int(_metric_arg(self.args, "chamfer_chunk_size", 2048)),
                )
                d_gp = _nearest_distances(
                    gt,
                    pred,
                    int(_metric_arg(self.args, "chamfer_chunk_size", 2048)),
                )
                thr = float(_metric_arg(self.args, "fscore_threshold", 0.2))
                self.chamfer_f_sum += float(d_pg.sum().item())
                self.chamfer_b_sum += float(d_gp.sum().item())
                self.chamfer_f_count += float(d_pg.numel())
                self.chamfer_b_count += float(d_gp.numel())
                self.fscore_prec_hits += float((d_pg < thr).to(torch.float32).sum().item())
                self.fscore_rec_hits += float((d_gp < thr).to(torch.float32).sum().item())

    def update_cross_view(
        self,
        pred_ref: torch.Tensor,
        pred_local: torch.Tensor,
        K_per_frame: torch.Tensor,
        cam2world: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        pair_mode = _metric_arg(self.args, "cross_view_pairs", "adjacent")
        if pair_mode == "none":
            return
        B, N, H, W, _ = pred_ref.shape
        pairs = _view_pairs(N, pair_mode)
        max_points = int(_metric_arg(self.args, "cross_view_max_points", 4096))
        if not pairs or max_points <= 0:
            return
        gen = torch.Generator(device=pred_ref.device)
        gen.manual_seed(
            int(getattr(self.args, "seed", 0)) + 7919 + int(self.cross_count) % 1000003
        )

        T_ref_from_world = torch.linalg.inv(cam2world[:, 0].float())
        T_ref_from_cam = T_ref_from_world[:, None] @ cam2world.float()
        T_cam_from_ref = torch.linalg.inv(T_ref_from_cam)
        K = K_per_frame.to(device=pred_ref.device, dtype=torch.float32)

        for b in range(B):
            for src, dst in pairs:
                src_mask = (
                    valid[b, src]
                    & torch.isfinite(pred_ref[b, src]).all(dim=-1)
                    & torch.isfinite(pred_local[b, src]).all(dim=-1)
                )
                if not bool(src_mask.any().item()):
                    continue
                src_ref = _sample_points(pred_ref[b, src][src_mask].float(), max_points, gen)
                if src_ref.numel() == 0:
                    continue

                T = T_cam_from_ref[b, dst]
                dst_local = src_ref @ T[:3, :3].T + T[:3, 3]
                z = dst_local[:, 2]
                ok_z = torch.isfinite(dst_local).all(dim=-1) & (z > 1e-6)
                if not bool(ok_z.any().item()):
                    continue
                src_ref = src_ref[ok_z]
                dst_local = dst_local[ok_z]
                z = z[ok_z]

                K_bd = K[b, dst]
                u = dst_local[:, 0] / z * K_bd[0, 0] + K_bd[0, 2]
                v = dst_local[:, 1] / z * K_bd[1, 1] + K_bd[1, 2]
                x = torch.round(u).long()
                y = torch.round(v).long()
                in_img = (x >= 0) & (x < W) & (y >= 0) & (y < H)
                if not bool(in_img.any().item()):
                    continue
                src_ref = src_ref[in_img]
                x = x[in_img]
                y = y[in_img]
                target_valid = valid[b, dst, y, x]
                if not bool(target_valid.any().item()):
                    continue
                src_ref = src_ref[target_valid]
                x = x[target_valid]
                y = y[target_valid]
                dst_ref = pred_ref[b, dst, y, x].float()
                finite = torch.isfinite(dst_ref).all(dim=-1)
                if not bool(finite.any().item()):
                    continue
                err = torch.linalg.norm(src_ref[finite] - dst_ref[finite], dim=-1)
                self.cross_count += float(err.numel())
                self.cross_sum += float(err.sum().item())
                self.cross_med.add(err)

    def state_dict(self) -> Dict:
        state = {name: float(getattr(self, name)) for name in self._STATE_SCALARS}
        state.update(
            {
                "pts_l2_med_values": self.pts_l2_med.values,
                "pts_l2_med_seen": int(self.pts_l2_med.seen),
                "scale_l2_med_values": self.scale_l2_med.values,
                "scale_l2_med_seen": int(self.scale_l2_med.seen),
                "reproj_med_values": self.reproj_med.values,
                "reproj_med_seen": int(self.reproj_med.seen),
                "cross_med_values": self.cross_med.values,
                "cross_med_seen": int(self.cross_med.seen),
                "conf_values": self.conf_pairs.conf,
                "conf_err_values": self.conf_pairs.err,
                "conf_seen": int(self.conf_pairs.seen),
            }
        )
        return state

    @classmethod
    def from_states(cls, args: argparse.Namespace, states: List[Dict]) -> "PointmapMetricAccumulator":
        merged = cls(args)
        for name in cls._STATE_SCALARS:
            setattr(merged, name, float(sum(float(s.get(name, 0.0)) for s in states)))

        def merge_values(key: str, max_size: int, seed: int) -> Optional[torch.Tensor]:
            values = [
                s[key].reshape(-1).to(device="cpu", dtype=torch.float32)
                for s in states
                if s.get(key) is not None and int(s[key].numel()) > 0
            ]
            if not values or max_size <= 0:
                return None
            out = torch.cat(values, dim=0)
            if out.numel() > max_size:
                gen = torch.Generator(device="cpu")
                gen.manual_seed(int(seed))
                idx = torch.randperm(out.numel(), generator=gen)[:max_size]
                out = out[idx]
            return out.contiguous()

        max_stats = int(_metric_arg(args, "stat_sample_max_points", 2_000_000))
        seed = int(getattr(args, "seed", 0))
        merged.pts_l2_med.values = merge_values("pts_l2_med_values", max_stats, seed + 1011)
        merged.pts_l2_med.seen = int(sum(int(s.get("pts_l2_med_seen", 0)) for s in states))
        merged.scale_l2_med.values = merge_values("scale_l2_med_values", max_stats, seed + 1012)
        merged.scale_l2_med.seen = int(sum(int(s.get("scale_l2_med_seen", 0)) for s in states))
        merged.reproj_med.values = merge_values("reproj_med_values", max_stats, seed + 1013)
        merged.reproj_med.seen = int(sum(int(s.get("reproj_med_seen", 0)) for s in states))
        merged.cross_med.values = merge_values("cross_med_values", max_stats, seed + 1014)
        merged.cross_med.seen = int(sum(int(s.get("cross_med_seen", 0)) for s in states))

        conf_values = [
            s["conf_values"].reshape(-1).to(device="cpu", dtype=torch.float32)
            for s in states
            if s.get("conf_values") is not None and int(s["conf_values"].numel()) > 0
        ]
        err_values = [
            s["conf_err_values"].reshape(-1).to(device="cpu", dtype=torch.float32)
            for s in states
            if s.get("conf_err_values") is not None and int(s["conf_err_values"].numel()) > 0
        ]
        max_conf = int(_metric_arg(args, "confidence_max_points", 2_000_000))
        if conf_values and err_values and max_conf > 0:
            conf = torch.cat(conf_values, dim=0)
            err = torch.cat(err_values, dim=0)
            if conf.numel() != err.numel():
                raise RuntimeError(
                    f"Confidence reservoir merge mismatch: conf={conf.numel()} err={err.numel()}"
                )
            if conf.numel() > max_conf:
                gen = torch.Generator(device="cpu")
                gen.manual_seed(seed + 1015)
                idx = torch.randperm(conf.numel(), generator=gen)[:max_conf]
                conf = conf[idx]
                err = err[idx]
            merged.conf_pairs.conf = conf.contiguous()
            merged.conf_pairs.err = err.contiguous()
        merged.conf_pairs.seen = int(sum(int(s.get("conf_seen", 0)) for s in states))
        return merged

    def finalize(self) -> Dict[str, float]:
        out: Dict[str, float] = {
            "pts3d_valid": self.count,
            "pts3d_l1": self.pts_l1_sum / max(self.count, 1.0),
            "pts3d_l2": self.pts_l2_sum / max(self.count, 1.0),
            "pts3d_median_error": self.pts_l2_med.median(),
            "scale_aligned_pts3d_l1": self.scale_l1_sum / max(self.scale_count, 1.0),
            "scale_aligned_pts3d_l2": self.scale_l2_sum / max(self.scale_count, 1.0),
            "scale_aligned_pts3d_median_error": self.scale_l2_med.median(),
            "depth_absrel": self.depth_absrel_sum / max(self.depth_count, 1.0),
            "depth_rmse": math.sqrt(self.depth_sq_sum / max(self.depth_count, 1.0)),
            "depth_delta_lt_1_25": self.depth_delta_sum / max(self.depth_count, 1.0),
            "depth_valid": self.depth_count,
            "reprojection_error_px": self.reproj_sum / max(self.reproj_count, 1.0),
            "reprojection_median_error_px": self.reproj_med.median(),
            "reprojection_valid": self.reproj_count,
        }

        chamfer_f = self.chamfer_f_sum / max(self.chamfer_f_count, 1.0)
        chamfer_b = self.chamfer_b_sum / max(self.chamfer_b_count, 1.0)
        precision = self.fscore_prec_hits / max(self.chamfer_f_count, 1.0)
        recall = self.fscore_rec_hits / max(self.chamfer_b_count, 1.0)
        fscore = 2.0 * precision * recall / max(precision + recall, 1e-12)
        out.update(
            {
                "chamfer_distance": 0.5 * (chamfer_f + chamfer_b),
                "chamfer_forward": chamfer_f,
                "chamfer_backward": chamfer_b,
                "fscore": fscore,
                "fscore_precision": precision,
                "fscore_recall": recall,
                "fscore_threshold": float(_metric_arg(self.args, "fscore_threshold", 0.2)),
                "chamfer_pred_points": self.chamfer_f_count,
                "chamfer_gt_points": self.chamfer_b_count,
                "cross_view_consistency_l2": self.cross_sum / max(self.cross_count, 1.0),
                "cross_view_consistency_median": self.cross_med.median(),
                "cross_view_valid": self.cross_count,
            }
        )
        conf_good = _metric_arg(self.args, "confidence_good_threshold", None)
        good_thr = (
            float(conf_good)
            if conf_good is not None
            else float(_metric_arg(self.args, "fscore_threshold", 0.2))
        )
        out.update(self.conf_pairs.aucs(good_thr))
        return out


def _sample_points(points: torch.Tensor, max_points: int, gen: torch.Generator) -> torch.Tensor:
    if points.shape[0] <= int(max_points):
        return points
    idx = torch.randperm(points.shape[0], device=points.device, generator=gen)[: int(max_points)]
    return points[idx]


def _nearest_distances(src: torch.Tensor, dst: torch.Tensor, chunk_size: int) -> torch.Tensor:
    outs: List[torch.Tensor] = []
    chunk = max(int(chunk_size), 1)
    dst = dst.float()
    for start in range(0, src.shape[0], chunk):
        d = torch.cdist(src[start:start + chunk].float(), dst)
        outs.append(d.min(dim=1).values)
    return torch.cat(outs, dim=0)


def _view_pairs(n_views: int, mode: str) -> List[Tuple[int, int]]:
    if mode == "none":
        return []
    if mode == "adjacent":
        pairs: List[Tuple[int, int]] = []
        for i in range(n_views - 1):
            pairs.append((i, i + 1))
            pairs.append((i + 1, i))
        return pairs
    return [(i, j) for i in range(n_views) for j in range(n_views) if i != j]


def valid_pointmap_mask(
    pred_ref: torch.Tensor,
    pred_local: torch.Tensor,
    dense_depth: torch.Tensor,
    K_per_frame: torch.Tensor,
    cam2world: torch.Tensor,
    frame_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gt_ref, gt_local, valid = _pointmap_targets_from_depth(
        dense_depth.float(),
        K_per_frame.float(),
        cam2world.float(),
    )
    if pred_ref.shape != gt_ref.shape or pred_local.shape != gt_local.shape:
        raise RuntimeError(
            f"Pointmap shape mismatch: pred_ref={tuple(pred_ref.shape)} "
            f"pred_local={tuple(pred_local.shape)} gt_ref={tuple(gt_ref.shape)} "
            f"gt_local={tuple(gt_local.shape)}."
        )
    if frame_mask is not None:
        valid = valid & frame_mask.to(device=valid.device, dtype=torch.bool).view(
            valid.shape[0], valid.shape[1], 1, 1
        )
    return gt_ref, gt_local, valid


def update_pointmap_metrics_from_batch(
    accum: PointmapMetricAccumulator,
    out: Dict[str, torch.Tensor],
    batch: Dict,
    device: torch.device,
    cam2world_per_frame: torch.Tensor,
) -> None:
    required = ("pointmap_pts3d", "pointmap_pts3d_local")
    missing = [k for k in required if k not in out]
    if missing:
        raise RuntimeError(f"Model output is missing pointmap keys: {missing}")

    pred_ref = out["pointmap_pts3d"].float()
    pred_local = out["pointmap_pts3d_local"].float()
    pred_conf = out.get("pointmap_conf")
    if pred_conf is not None:
        pred_conf = pred_conf.float()

    dense_depth = batch["dense_depth"].to(device=device, dtype=torch.float32, non_blocking=True)
    if "K_per_frame" in batch:
        K_per_frame = batch["K_per_frame"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
    else:
        K_per_frame = torch.stack(
            [
                v["camera_intrinsics"].to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                for v in batch["views"]
            ],
            dim=1,
        )
    frame_mask = batch.get("dense_depth_frame_mask")
    if frame_mask is not None:
        frame_mask = frame_mask.to(device=device, non_blocking=True)

    gt_ref, _gt_local, valid = valid_pointmap_mask(
        pred_ref,
        pred_local,
        dense_depth,
        K_per_frame,
        cam2world_per_frame,
        frame_mask,
    )

    accum.update_point_errors(pred_ref, gt_ref, valid, pred_conf)
    accum.update_scale_aligned(pred_ref, gt_ref, valid)
    accum.update_depth(pred_local, dense_depth, valid)
    accum.update_reprojection(pred_local, K_per_frame, valid)
    accum.update_chamfer(pred_ref, gt_ref, valid)
    accum.update_cross_view(pred_ref, pred_local, K_per_frame, cam2world_per_frame, valid)


def json_safe(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj
