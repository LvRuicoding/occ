"""Stage-1 KITTI Object detection models with a local CenterHead."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lidar_fusion import LidarImageFusionModule
from .stage1_pointmap_ablation import _make_recon_backbone
from .stage1_ssc_bevdetocc_lidar import LSSDepthLift, OccAnyTokenProjector


def conv_bn_relu_2d(
    c_in: int,
    c_out: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(int(c_in), int(c_out), kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(int(c_out)),
        nn.ReLU(inplace=True),
    )


class BasicBlock2D(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = conv_bn_relu_2d(c_in, c_out, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(int(c_out), int(c_out), 3, padding=1, bias=False),
            nn.BatchNorm2d(int(c_out)),
        )
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(int(c_in), int(c_out), 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(int(c_out)),
            )
            if stride != 1 or int(c_in) != int(c_out)
            else None
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv2(self.conv1(x))
        return self.relu(out + identity)


class CustomResNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        num_layer: Tuple[int, ...] = (1, 2, 4),
        num_channels: Tuple[int, ...] = (128, 256, 512),
        stride: Tuple[int, ...] = (1, 2, 2),
        output_ids: Tuple[int, ...] = (0, 1, 2),
    ) -> None:
        super().__init__()
        self.output_ids = tuple(int(i) for i in output_ids)
        layers: List[nn.Module] = []
        c_cur = int(in_channels)
        for n_blocks, c_out, s in zip(num_layer, num_channels, stride):
            blocks: List[nn.Module] = [BasicBlock2D(c_cur, int(c_out), stride=int(s))]
            c_cur = int(c_out)
            for _ in range(int(n_blocks) - 1):
                blocks.append(BasicBlock2D(c_cur, c_cur))
            layers.append(nn.Sequential(*blocks))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        out = x
        for idx, layer in enumerate(self.layers):
            out = layer(out)
            if idx in self.output_ids:
                feats.append(out)
        return feats


class FPNLSS2D(nn.Module):
    def __init__(self, in_channels: int = 640, out_channels: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            conv_bn_relu_2d(in_channels, out_channels, kernel_size=3, padding=1),
            conv_bn_relu_2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        if len(feats) != 3:
            raise RuntimeError(f"FPNLSS2D expects 3 feature maps, got {len(feats)}.")
        x0, _x1, x2 = feats
        x2 = F.interpolate(x2, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x0, x2], dim=1))


def _gaussian_radius(det_size: Tuple[float, float], min_overlap: float = 0.1) -> float:
    height, width = det_size
    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
    sq1 = math.sqrt(max(0.0, b1**2 - 4.0 * a1 * c1))
    r1 = (b1 + sq1) / 2.0

    a2 = 4.0
    b2 = 2.0 * (height + width)
    c2 = (1.0 - min_overlap) * width * height
    sq2 = math.sqrt(max(0.0, b2**2 - 4.0 * a2 * c2))
    r2 = (b2 + sq2) / 2.0

    a3 = 4.0 * min_overlap
    b3 = -2.0 * min_overlap * (height + width)
    c3 = (min_overlap - 1.0) * width * height
    sq3 = math.sqrt(max(0.0, b3**2 - 4.0 * a3 * c3))
    r3 = (b3 + sq3) / (2.0 * max(a3, 1e-6))
    return min(r1, r2, r3)


def _draw_gaussian(heatmap: torch.Tensor, center: Tuple[int, int], radius: int) -> None:
    diameter = 2 * int(radius) + 1
    x = torch.arange(diameter, device=heatmap.device, dtype=torch.float32) - radius
    y = torch.arange(diameter, device=heatmap.device, dtype=torch.float32) - radius
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    sigma = max(float(diameter) / 6.0, 1e-6)
    gaussian = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))

    cx, cy = int(center[0]), int(center[1])
    height, width = heatmap.shape[-2:]
    left = min(cx, radius)
    right = min(height - cx - 1, radius)
    top = min(cy, radius)
    bottom = min(width - cy - 1, radius)
    if left < 0 or right < 0 or top < 0 or bottom < 0:
        return
    patch = heatmap[cx - left: cx + right + 1, cy - top: cy + bottom + 1]
    gpatch = gaussian[radius - left: radius + right + 1, radius - top: radius + bottom + 1]
    torch.maximum(patch, gpatch.to(dtype=patch.dtype), out=patch)


def _gather_feat(feat: torch.Tensor, inds: torch.Tensor) -> torch.Tensor:
    B, C, H, W = feat.shape
    flat = feat.view(B, C, H * W).permute(0, 2, 1).contiguous()
    gather_inds = inds.unsqueeze(-1).expand(-1, -1, C)
    return flat.gather(1, gather_inds)


def _gaussian_focal_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.clamp(min=1e-4, max=1.0 - 1e-4)
    pos = target.eq(1.0).float()
    neg = target.lt(1.0).float()
    neg_weights = torch.pow(1.0 - target, 4.0)
    pos_loss = -torch.log(pred) * torch.pow(1.0 - pred, 2.0) * pos
    neg_loss = -torch.log(1.0 - pred) * torch.pow(pred, 2.0) * neg_weights * neg
    num_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def _axis_aligned_nms_bev(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)
    order = scores.argsort(descending=True)
    keep: List[torch.Tensor] = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        cur = boxes[i]
        rest = boxes[order[1:]]
        x1 = cur[0] - cur[3] * 0.5
        x2 = cur[0] + cur[3] * 0.5
        y1 = cur[1] - cur[4] * 0.5
        y2 = cur[1] + cur[4] * 0.5
        rx1 = rest[:, 0] - rest[:, 3] * 0.5
        rx2 = rest[:, 0] + rest[:, 3] * 0.5
        ry1 = rest[:, 1] - rest[:, 4] * 0.5
        ry2 = rest[:, 1] + rest[:, 4] * 0.5
        inter = (torch.minimum(x2, rx2) - torch.maximum(x1, rx1)).clamp(min=0.0)
        inter = inter * (torch.minimum(y2, ry2) - torch.maximum(y1, ry1)).clamp(min=0.0)
        area = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
        rarea = (rx2 - rx1).clamp(min=0.0) * (ry2 - ry1).clamp(min=0.0)
        iou = inter / (area + rarea - inter).clamp(min=1e-6)
        order = order[1:][iou <= float(iou_threshold)]
    return torch.stack(keep)


class SimpleCenterHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 3,
        pc_range: Tuple[float, float, float, float, float, float] = (0.0, -25.6, -2.0, 51.2, 25.6, 4.4),
        voxel_size: Tuple[float, float] = (0.4, 0.4),
        max_objs: int = 100,
        score_threshold: float = 0.05,
        nms_iou_threshold: float = 0.2,
        max_per_img: int = 100,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.pc_range = tuple(float(v) for v in pc_range)
        self.voxel_size = tuple(float(v) for v in voxel_size)
        self.max_objs = int(max_objs)
        self.score_threshold = float(score_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.max_per_img = int(max_per_img)
        self.shared_conv = conv_bn_relu_2d(in_channels, in_channels, kernel_size=3, padding=1)

        def head(out_channels: int, final_bias: float = 0.0) -> nn.Sequential:
            seq = nn.Sequential(
                conv_bn_relu_2d(in_channels, 64, kernel_size=3, padding=1),
                nn.Conv2d(64, int(out_channels), kernel_size=1),
            )
            nn.init.constant_(seq[-1].bias, float(final_bias))
            return seq

        self.heatmap = head(self.num_classes, final_bias=-2.19)
        self.reg = head(2)
        self.height = head(1)
        self.dim = head(3)
        self.rot = head(2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.shared_conv(x)
        return {
            "heatmap": self.heatmap(x),
            "reg": self.reg(x),
            "height": self.height(x),
            "dim": self.dim(x),
            "rot": self.rot(x),
        }

    def _get_targets(
        self,
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        feat_shape: Tuple[int, int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = len(gt_boxes)
        H, W = int(feat_shape[0]), int(feat_shape[1])
        heatmap = torch.zeros((B, self.num_classes, H, W), device=device)
        anno_box = torch.zeros((B, self.max_objs, 8), device=device)
        inds = torch.zeros((B, self.max_objs), dtype=torch.long, device=device)
        mask = torch.zeros((B, self.max_objs), dtype=torch.bool, device=device)
        x_min, y_min = self.pc_range[0], self.pc_range[1]
        vx, vy = self.voxel_size
        for b, (boxes_b, labels_b) in enumerate(zip(gt_boxes, gt_labels)):
            boxes_b = boxes_b.to(device=device, dtype=torch.float32)
            labels_b = labels_b.to(device=device, dtype=torch.long)
            obj_count = 0
            for box, label in zip(boxes_b, labels_b):
                if obj_count >= self.max_objs:
                    break
                cls = int(label.item())
                if cls < 0 or cls >= self.num_classes:
                    continue
                x, y, z, length, width, height, yaw = [float(v) for v in box.detach().cpu()]
                if length <= 0.0 or width <= 0.0 or height <= 0.0:
                    continue
                coor_x = (x - x_min) / vx
                coor_y = (y - y_min) / vy
                if not (0.0 <= coor_x < H and 0.0 <= coor_y < W):
                    continue
                cx = int(coor_x)
                cy = int(coor_y)
                radius = _gaussian_radius((length / vx, width / vy), min_overlap=0.1)
                radius_i = max(1, int(radius))
                _draw_gaussian(heatmap[b, cls], (cx, cy), radius_i)
                inds[b, obj_count] = cx * W + cy
                anno_box[b, obj_count] = torch.tensor(
                    [
                        coor_x - cx,
                        coor_y - cy,
                        z,
                        math.log(max(length, 1e-3)),
                        math.log(max(width, 1e-3)),
                        math.log(max(height, 1e-3)),
                        math.sin(yaw),
                        math.cos(yaw),
                    ],
                    device=device,
                    dtype=torch.float32,
                )
                mask[b, obj_count] = True
                obj_count += 1
        return heatmap, anno_box, inds, mask

    def loss(
        self,
        preds: Dict[str, torch.Tensor],
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        heatmap_pred = torch.sigmoid(preds["heatmap"].float())
        B, _C, H, W = heatmap_pred.shape
        heatmap_t, anno_t, inds, mask = self._get_targets(gt_boxes, gt_labels, (H, W), heatmap_pred.device)
        hm_loss = _gaussian_focal_loss(heatmap_pred, heatmap_t)
        box_pred = torch.cat(
            [preds["reg"], preds["height"], preds["dim"], preds["rot"]],
            dim=1,
        ).float()
        gathered = _gather_feat(box_pred, inds)
        mask_f = mask.unsqueeze(-1).float()
        pos = mask_f.sum().clamp(min=1.0)
        box_loss = (F.l1_loss(gathered * mask_f, anno_t * mask_f, reduction="sum") / pos)
        loss = hm_loss + 0.25 * box_loss
        return loss, {
            "det_heatmap": float(hm_loss.detach()),
            "det_bbox": float(box_loss.detach()),
            "det_pos": float(mask.sum().detach()),
        }

    @torch.no_grad()
    def decode(self, preds: Dict[str, torch.Tensor], topk: int = 200) -> List[Dict[str, torch.Tensor]]:
        heatmap = torch.sigmoid(preds["heatmap"].float())
        heatmap = heatmap * (F.max_pool2d(heatmap, 3, stride=1, padding=1) == heatmap).float()
        B, C, H, W = heatmap.shape
        outputs: List[Dict[str, torch.Tensor]] = []
        x_min, y_min = self.pc_range[0], self.pc_range[1]
        vx, vy = self.voxel_size
        for b in range(B):
            k = min(int(topk), C * H * W)
            scores, flat_inds = torch.topk(heatmap[b].reshape(-1), k=k)
            labels = flat_inds // (H * W)
            rem = flat_inds % (H * W)
            xs = rem // W
            ys = rem % W
            reg = preds["reg"][b].float().permute(1, 2, 0).reshape(H * W, 2)[rem]
            z = preds["height"][b].float().permute(1, 2, 0).reshape(H * W, 1)[rem, 0]
            dims = torch.exp(preds["dim"][b].float().permute(1, 2, 0).reshape(H * W, 3)[rem])
            rot = preds["rot"][b].float().permute(1, 2, 0).reshape(H * W, 2)[rem]
            yaw = torch.atan2(rot[:, 0], rot[:, 1])
            centers_x = (xs.float() + reg[:, 0]) * vx + x_min
            centers_y = (ys.float() + reg[:, 1]) * vy + y_min
            boxes = torch.stack(
                [centers_x, centers_y, z, dims[:, 0], dims[:, 1], dims[:, 2], yaw],
                dim=1,
            )
            valid = (
                (scores >= self.score_threshold)
                & (boxes[:, 0] >= self.pc_range[0])
                & (boxes[:, 0] <= self.pc_range[3])
                & (boxes[:, 1] >= self.pc_range[1])
                & (boxes[:, 1] <= self.pc_range[4])
            )
            boxes = boxes[valid]
            scores_b = scores[valid]
            labels_b = labels[valid].long()
            keep_all: List[torch.Tensor] = []
            for cls in range(C):
                cls_inds = torch.nonzero(labels_b == cls, as_tuple=False).view(-1)
                if cls_inds.numel() == 0:
                    continue
                keep_rel = _axis_aligned_nms_bev(
                    boxes[cls_inds],
                    scores_b[cls_inds],
                    self.nms_iou_threshold,
                )
                keep_all.append(cls_inds[keep_rel])
            if keep_all:
                keep = torch.cat(keep_all, dim=0)
                keep = keep[scores_b[keep].argsort(descending=True)]
                keep = keep[: self.max_per_img]
                boxes = boxes[keep]
                scores_b = scores_b[keep]
                labels_b = labels_b[keep]
            else:
                boxes = boxes.new_zeros((0, 7))
                scores_b = boxes.new_zeros((0,))
                labels_b = boxes.new_zeros((0,), dtype=torch.long)
            outputs.append({"boxes_3d": boxes, "scores_3d": scores_b, "labels_3d": labels_b})
        return outputs


class _Stage1DetBaseModel(nn.Module):
    def __init__(
        self,
        *,
        use_lidar_fusion: bool,
        occany_ckpt: Optional[str] = None,
        c_lift: int = 64,
        patch_size: int = 16,
        token_dim: int = 768,
        backbone_img_size: Tuple[int, int] = (512, 512),
        backbone_dtype: torch.dtype = torch.bfloat16,
        num_frames: int = 5,
        freeze_backbone: bool = False,
        backbone: str = "must3r",
        fusion_vox_origin: Tuple[float, float, float] = (-25.6, -2.0, 0.0),
        fusion_vox_size: Tuple[float, float, float] = (0.4, 0.4, 0.4),
        fusion_vox_grid: Tuple[int, int, int] = (128, 16, 128),
        fusion_num_heads: int = 8,
        fusion_window: int = 4,
        fusion_d_voxel: int = 128,
        fusion_pe_num_freqs: int = 8,
        fusion_attn_type: str = "cross",
        lss_in_channels: int = 256,
        lss_out_channels: int = 32,
        depth_bound: Tuple[float, float, float] = (1.0, 52.0, 0.4),
        det_score_threshold: float = 0.05,
        **_unused,
    ) -> None:
        super().__init__()
        del c_lift
        if use_lidar_fusion and fusion_attn_type != "cross":
            raise ValueError(f"Detection post-fusion uses cross attention, got {fusion_attn_type!r}.")
        self.num_frames = int(num_frames)
        self.freeze_backbone = bool(freeze_backbone)
        self.use_lidar_fusion = bool(use_lidar_fusion)
        self.half_grid_size = (128, 128, 16)
        self.backbone = _make_recon_backbone(
            backbone=backbone,
            img_size=backbone_img_size,
            embed_dim=token_dim,
            patch_size=patch_size,
            backbone_dtype=backbone_dtype,
            freeze=self.freeze_backbone,
        )
        if occany_ckpt is not None:
            self.backbone.load_checkpoint(occany_ckpt)

        H_t = backbone_img_size[0] // int(patch_size)
        W_t = backbone_img_size[1] // int(patch_size)
        self.fusion = None
        if self.use_lidar_fusion:
            self.fusion = LidarImageFusionModule(
                d_model=token_dim,
                H_t=H_t,
                W_t=W_t,
                patch_size=patch_size,
                num_heads=fusion_num_heads,
                window=fusion_window,
                vox_origin=fusion_vox_origin,
                vox_size=fusion_vox_size,
                vox_grid=fusion_vox_grid,
                vfe_d_voxel=fusion_d_voxel,
                pe_num_freqs=fusion_pe_num_freqs,
                attn_type="cross",
                fusion3d_enabled=False,
            )
        self.token_projector = OccAnyTokenProjector(token_dim=token_dim, out_channels=lss_in_channels)
        self.lss = LSSDepthLift(
            in_channels=lss_in_channels,
            out_channels=lss_out_channels,
            depth_bound=depth_bound,
            voxel_origin=(0.0, -25.6, -2.0),
            voxel_size=(0.4, 0.4, 0.4),
            grid_size=self.half_grid_size,
        )
        self.volume_to_bev = conv_bn_relu_2d(lss_out_channels * self.half_grid_size[2], 64)
        self.bev_backbone = CustomResNet2D(in_channels=64)
        self.bev_neck = FPNLSS2D(in_channels=128 + 512, out_channels=256)
        self.det_head = SimpleCenterHead(
            in_channels=256,
            num_classes=3,
            score_threshold=float(det_score_threshold),
        )

    def set_freeze_backbone(self, freeze: bool = True) -> None:
        self.freeze_backbone = bool(freeze)
        self.backbone.set_frozen(self.freeze_backbone)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    @staticmethod
    def _grid_tuple(
        grid_config: Optional[Dict[str, torch.Tensor | Tuple[int, int, int]]],
        name: str,
        default: Tuple[int, int, int],
    ) -> Tuple[int, int, int]:
        if grid_config is None or name not in grid_config:
            return default
        value = grid_config[name]
        if isinstance(value, torch.Tensor):
            if value.ndim == 2:
                if value.shape[0] > 1 and not torch.equal(value, value[:1].expand_as(value)):
                    raise RuntimeError(f"{name} must be identical within a batch; got {value.tolist()}")
                value = value[0]
            return tuple(int(v) for v in value.detach().cpu().tolist())
        return tuple(int(v) for v in value)

    @staticmethod
    def _grid_tensor(
        grid_config: Optional[Dict[str, torch.Tensor | Tuple[int, int, int]]],
        name: str,
        default: torch.Tensor,
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        if grid_config is None or name not in grid_config:
            value = default
        else:
            value = grid_config[name]
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32)
        value = value.to(device=device, dtype=torch.float32)
        if value.ndim == 1:
            value = value.view(1, 3)
        if value.shape[0] == 1 and batch_size > 1:
            value = value.expand(batch_size, -1)
        return value

    def forward(
        self,
        views: List[Dict[str, torch.Tensor]],
        T_target_from_refcam: torch.Tensor,
        points_per_frame: Optional[List[List[torch.Tensor]]] = None,
        T_cam_from_velo: Optional[torch.Tensor] = None,
        K_per_frame: Optional[torch.Tensor] = None,
        image_hw: Optional[torch.Tensor] = None,
        gt_depth: Optional[torch.Tensor] = None,
        return_depth: bool = False,
        grid_config: Optional[Dict[str, torch.Tensor | Tuple[int, int, int]]] = None,
    ) -> Dict[str, torch.Tensor]:
        del T_target_from_refcam, gt_depth, return_depth
        if T_cam_from_velo is None or K_per_frame is None or image_hw is None:
            raise RuntimeError("Detection models require T_cam_from_velo, K_per_frame and image_hw.")
        backbone_out = self.backbone(views)
        t_rec = backbone_out["t_rec"]
        if t_rec.shape[1] != self.num_frames:
            raise RuntimeError(f"model was built for num_frames={self.num_frames}, got N={t_rec.shape[1]}.")

        B = int(t_rec.shape[0])
        fusion_origin = None
        fusion_size = None
        fusion_grid = None
        if self.use_lidar_fusion:
            if points_per_frame is None:
                raise RuntimeError("det_postfusion_only requires points_per_frame.")
            if grid_config is not None:
                fusion_origin = grid_config.get("fusion_vox_origin")
                fusion_size = grid_config.get("fusion_vox_size")
                fusion_grid = self._grid_tuple(grid_config, "fusion_vox_grid", self.fusion.vfe.vox_grid)
            t_rec = self.fusion(
                t_rec,
                points_per_frame=points_per_frame,
                T_cam_from_velo=T_cam_from_velo,
                K_per_frame=K_per_frame,
                image_hw=image_hw,
                p_rec_local=backbone_out.get("p_rec_local"),
                c_rec=backbone_out["c_rec"],
                fusion_vox_origin=fusion_origin,
                fusion_vox_size=fusion_size,
                fusion_vox_grid=fusion_grid,
            )

        feat_2d = self.token_projector(t_rec)
        half_grid = self._grid_tuple(grid_config, "half_grid_size", self.half_grid_size)
        half_origin = self._grid_tensor(
            grid_config,
            "half_voxel_origin",
            self.lss.voxel_origin,
            feat_2d.device,
            B,
        )
        half_size = self._grid_tensor(
            grid_config,
            "half_voxel_size",
            self.lss.voxel_size,
            feat_2d.device,
            B,
        )
        T_cam = T_cam_from_velo.to(device=feat_2d.device)
        lss_volume, _depth_logits = self.lss(
            feat_2d=feat_2d,
            K_per_frame=K_per_frame.to(device=feat_2d.device),
            T_cam_from_velo=T_cam,
            image_hw=image_hw.to(device=feat_2d.device),
            voxel_origin=half_origin,
            voxel_size=half_size,
            grid_size=half_grid,
        )
        # KITTI Object has reliable historical images/LiDAR but no per-frame ego
        # poses in this dataset root, so LSS is applied directly after fusion and
        # detection uses the target-frame volume without temporal BEV warping.
        vol = lss_volume[:, 0]  # (B, C, X, Y, Z)
        B, C, X, Y, Z = vol.shape
        bev = vol.permute(0, 1, 4, 2, 3).reshape(B, C * Z, X, Y)
        bev = self.volume_to_bev(bev)
        bev = self.bev_neck(self.bev_backbone(bev))
        return {"det_preds": self.det_head(bev), "bev_feat": bev}

    def det_loss(
        self,
        preds: Dict[str, torch.Tensor],
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.det_head.loss(preds, gt_boxes, gt_labels)

    @torch.no_grad()
    def det_decode(self, preds: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        return self.det_head.decode(preds)


class Stage1DetOriginalModel(_Stage1DetBaseModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, use_lidar_fusion=False, **kwargs)


class Stage1DetPostFusionOnlyModel(_Stage1DetBaseModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, use_lidar_fusion=True, **kwargs)


__all__ = [
    "SimpleCenterHead",
    "Stage1DetOriginalModel",
    "Stage1DetPostFusionOnlyModel",
]
