"""SemanticKITTI SSC loss utilities used by Stage-1 training scripts."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def class_weights_from_frequencies(
    num_classes: int = 20,
    *,
    max_weight: float = 5.0,
) -> torch.Tensor:
    """Return stable SemanticKITTI-style CE weights.

    The original Stage-1 trainer expects this helper from ``ft.semantickitti_ft``.
    In this checkout the package is not present, so we provide conservative
    weights that keep empty-space domination in check without requiring an
    external frequency file.
    """
    weights = torch.ones(int(num_classes), dtype=torch.float32)
    if weights.numel() > 0:
        weights[0] = 0.25
    return weights.clamp(0.25, float(max_weight))


class SSCLoss(nn.Module):
    """Cross-entropy SSC loss with ignore_index=255.

    Returns ``(loss, details)`` to match the contract used by
    ``ft.kitti_stage1_5f.tools.train``.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        if class_weights is None:
            class_weights = class_weights_from_frequencies()
        self.register_buffer("class_weights", class_weights.float(), persistent=False)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        weight = self.class_weights.to(device=logits.device, dtype=torch.float32)
        if weight.numel() != logits.shape[1]:
            weight = torch.ones(logits.shape[1], device=logits.device, dtype=torch.float32)
            if weight.numel() > 0:
                weight[0] = 0.25
        criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=self.ignore_index,
            reduction="mean",
        )
        loss = criterion(logits.float(), target.long())
        return loss, {"ce": float(loss.detach())}


__all__ = ["SSCLoss", "class_weights_from_frequencies"]
