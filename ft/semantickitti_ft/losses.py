"""SSC loss: weighted CE + Lovasz-Softmax.

Inspired by MonoScene's SSC loss but kept self-contained. The Lovasz
implementation here follows Berman et al. 2017 (the soft IoU surrogate).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _semantic_kitti_class_frequencies() -> np.ndarray:
    """Frequencies (training-set) for the 20 KITTI SSC classes, used for CE
    weighting. Numbers from MonoScene's published precomputed stats."""
    return np.array(
        [
            5.41773033e09,
            1.57835390e07,
            1.25136000e05,
            1.18809000e05,
            6.46799000e05,
            8.21951000e05,
            2.62978000e05,
            2.83696000e05,
            2.04750000e05,
            6.16887030e07,
            4.50296100e06,
            4.48836500e07,
            2.26992300e06,
            5.68402180e07,
            1.57196520e07,
            1.58442623e08,
            2.06162300e06,
            3.69705220e07,
            1.15198800e06,
            3.34146000e05,
        ],
        dtype=np.float64,
    )


def class_weights_from_frequencies(eps: float = 1e-3) -> torch.Tensor:
    freqs = _semantic_kitti_class_frequencies()
    weights = 1.0 / np.log(freqs + eps)
    weights = weights / weights.mean()
    return torch.from_numpy(weights).float()


def lovasz_softmax_3d(
    probas: torch.Tensor,
    labels: torch.Tensor,
    ignore: int = 255,
    classes: str = "present",
) -> torch.Tensor:
    """Multi-class Lovasz loss.

    Args:
      probas: (B, C, X, Y, Z) softmax probabilities.
      labels: (B, X, Y, Z) int64 labels.
    """
    B, C = probas.shape[:2]
    flat_probas = probas.permute(0, 2, 3, 4, 1).reshape(-1, C)
    flat_labels = labels.reshape(-1)

    valid = flat_labels != ignore
    flat_probas = flat_probas[valid]
    flat_labels = flat_labels[valid]
    if flat_labels.numel() == 0:
        return probas.sum() * 0.0

    losses = []
    cls_iter = range(C) if classes == "all" else _present_classes(flat_labels, C)
    for c in cls_iter:
        fg = (flat_labels == c).float()
        if classes == "present" and fg.sum() == 0:
            continue
        class_pred = flat_probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        fg_sorted = fg[perm]
        grad = _lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    if not losses:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


def _present_classes(labels: torch.Tensor, n_classes: int):
    return torch.unique(labels).tolist()


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[:-1]
    return jaccard


class SSCLoss(nn.Module):
    """CE (with class weights) + Lovasz-Softmax."""

    def __init__(
        self,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        lovasz_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.ce_weight = float(ce_weight)
        self.lovasz_weight = float(lovasz_weight)
        if class_weights is None:
            class_weights = class_weights_from_frequencies()
        self.register_buffer("class_weights", class_weights, persistent=False)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """logits: (B, C, X, Y, Z), target: (B, X, Y, Z)."""
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights.to(logits.dtype),
            ignore_index=self.ignore_index,
        )
        probas = F.softmax(logits, dim=1)
        lov = lovasz_softmax_3d(probas, target, ignore=self.ignore_index)
        loss = self.ce_weight * ce + self.lovasz_weight * lov
        details = dict(ce=float(ce.detach()), lovasz=float(lov.detach()))
        return loss, details
