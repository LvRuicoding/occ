"""Shared interfaces for SemanticKITTI fine-tuning modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class LiftedFeatures:
    """Inputs consumed by occupancy heads.

    A lifting module is responsible for building these fields from the raw
    batch/backbone state. A head module should only depend on this contract.
    """

    aggregated_tokens_list: List[torch.Tensor]
    images: torch.Tensor
    intrinsics: torch.Tensor
    camera_to_world: torch.Tensor
    lidar_to_world: torch.Tensor

