"""Local BEVDet 3D encoder/neck/head blocks used by the KITTI OCC branch.

This file is an mmcv-free PyTorch port of the BEVDet OCC 3D pieces referenced
by ``bevdet-occ-r50-4d-stereo-24e.py``:
``CustomResNet3D``, ``LSSFPN3D``, ``final_conv`` and ``predicter``.
Keeping the code in this repository avoids any runtime dependency on the
separate BEVDet checkout or its registry stack.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def conv_bn_relu_3d(
    c_in: int,
    c_out: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(
            int(c_in),
            int(c_out),
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm3d(int(c_out)),
        nn.ReLU(inplace=True),
    )


class BasicBlock3D(nn.Module):
    """PyTorch port of BEVDet's BasicBlock3D."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = conv_bn_relu_3d(c_in, c_out, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv3d(int(c_out), int(c_out), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(int(c_out)),
        )
        self.downsample = (
            nn.Sequential(
                nn.Conv3d(
                    int(c_in),
                    int(c_out),
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm3d(int(c_out)),
            )
            if stride != 1 or int(c_in) != int(c_out)
            else None
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return self.relu(out + identity)


class CustomResNet3D(nn.Module):
    """BEVDet CustomResNet3D topology without the mmcv dependency."""

    def __init__(
        self,
        numC_input: int,
        num_layer: Tuple[int, ...] = (1, 2, 4),
        num_channels: Tuple[int, ...] = (64, 128, 256),
        stride: Tuple[int, ...] = (1, 2, 2),
        backbone_output_ids: Tuple[int, ...] = (0, 1, 2),
        with_cp: bool = False,
    ) -> None:
        super().__init__()
        if not (len(num_layer) == len(num_channels) == len(stride)):
            raise ValueError("num_layer, num_channels, and stride must have the same length.")
        self.backbone_output_ids = tuple(int(i) for i in backbone_output_ids)
        self.with_cp = bool(with_cp)

        layers: List[nn.Module] = []
        c_cur = int(numC_input)
        for n_blocks, c_out, s in zip(num_layer, num_channels, stride):
            blocks: List[nn.Module] = [BasicBlock3D(c_cur, int(c_out), stride=int(s))]
            c_cur = int(c_out)
            for _ in range(int(n_blocks) - 1):
                blocks.append(BasicBlock3D(c_cur, c_cur, stride=1))
            layers.append(nn.Sequential(*blocks))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        out = x
        for idx, layer in enumerate(self.layers):
            if self.with_cp and out.requires_grad:
                out = checkpoint(layer, out, use_reentrant=False)
            else:
                out = layer(out)
            if idx in self.backbone_output_ids:
                feats.append(out)
        return feats


class LSSFPN3D(nn.Module):
    """BEVDet LSSFPN3D topology without the mmcv dependency."""

    def __init__(self, in_channels: int, out_channels: int, with_cp: bool = False) -> None:
        super().__init__()
        self.with_cp = bool(with_cp)
        self.conv = conv_bn_relu_3d(
            int(in_channels), int(out_channels), kernel_size=1, stride=1, padding=0
        )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        if len(feats) != 3:
            raise RuntimeError(f"LSSFPN3D expects 3 feature maps, got {len(feats)}.")
        x_8, x_16, x_32 = feats
        x_16 = F.interpolate(x_16, scale_factor=2, mode="trilinear", align_corners=True)
        x_32 = F.interpolate(x_32, scale_factor=4, mode="trilinear", align_corners=True)
        x = torch.cat([x_8, x_16, x_32], dim=1)
        if self.with_cp and x.requires_grad:
            return checkpoint(self.conv, x, use_reentrant=False)
        return self.conv(x)


class BEVDetOcc3DHead(nn.Module):
    """BEVDet-OCC 3D encoder/neck/final head adapted to KITTI grid layout."""

    def __init__(
        self,
        num_classes: int = 20,
        in_channels: int = 64,
        neck_channels: int = 32,
        full_grid: Tuple[int, int, int] = (256, 256, 32),
        with_cp: bool = False,
    ) -> None:
        super().__init__()
        self.full_grid = tuple(int(v) for v in full_grid)
        self.backbone = CustomResNet3D(
            numC_input=int(in_channels),
            num_layer=(1, 2, 4),
            num_channels=(64, 128, 256),
            stride=(1, 2, 2),
            backbone_output_ids=(0, 1, 2),
            with_cp=with_cp,
        )
        self.neck = LSSFPN3D(
            in_channels=64 + 128 + 256,
            out_channels=int(neck_channels),
            with_cp=with_cp,
        )
        self.full_upsample = nn.Sequential(
            nn.Conv3d(
                int(neck_channels),
                int(neck_channels),
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(int(neck_channels)),
            nn.ReLU(inplace=True),
        )
        self.final_conv = conv_bn_relu_3d(
            int(neck_channels), int(neck_channels), kernel_size=3, stride=1, padding=1
        )
        self.predicter = nn.Sequential(
            nn.Linear(int(neck_channels), int(neck_channels) * 2),
            nn.Softplus(),
            nn.Linear(int(neck_channels) * 2, int(num_classes)),
        )

    def forward(
        self,
        x: torch.Tensor,
        full_grid: Tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        target_grid = self.full_grid if full_grid is None else tuple(int(v) for v in full_grid)
        x = self.neck(self.backbone(x))
        x = F.interpolate(
            x.to(dtype=torch.float32),
            size=target_grid,
            mode="trilinear",
            align_corners=False,
        ).to(dtype=x.dtype)
        x = self.full_upsample(x)
        x = self.final_conv(x)
        logits = self.predicter(x.permute(0, 2, 3, 4, 1).contiguous())
        return logits.permute(0, 4, 1, 2, 3).contiguous()


__all__ = [
    "BEVDetOcc3DHead",
    "BasicBlock3D",
    "CustomResNet3D",
    "LSSFPN3D",
    "conv_bn_relu_3d",
]
