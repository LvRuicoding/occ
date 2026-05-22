"""Vendored MonoScene 3D U-Net head (KITTI variant).

Copied verbatim from https://github.com/astra-vision/MonoScene
(monoscene/models/{unet3d_kitti,modules,CRP3D,DDR}.py) with the package
imports rewritten to be local. Only `Process`, `Downsample`, `Upsample`,
`SegmentationHead`, `ASPP`, `CPMegaVoxels`, `Bottleneck3D`, and `UNet3D`
are needed for the kitti head with `context_prior=True`.
"""

from .unet3d_kitti import UNet3D

__all__ = ["UNet3D"]
