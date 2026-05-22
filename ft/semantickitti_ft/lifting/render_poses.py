"""Novel-view pose generation utilities."""
from __future__ import annotations

from typing import List

import numpy as np
import torch


def generate_render_poses(
    anchor_c2w: torch.Tensor,
    n_views: int = 4,
    lateral: float = 0.6,
    vertical: float = -1.0,
    forward: float = 1.5,
    pitch_deg: float = 8.0,
) -> torch.Tensor:
    """Build novel-view c2w poses by perturbing the anchor camera."""
    B = anchor_c2w.shape[0]
    device = anchor_c2w.device
    dtype = anchor_c2w.dtype

    deltas: List[torch.Tensor] = []
    base = torch.eye(4, device=device, dtype=dtype)

    if n_views >= 1:
        d = base.clone()
        d[0, 3] = -lateral
        deltas.append(d)
    if n_views >= 2:
        d = base.clone()
        d[0, 3] = lateral
        deltas.append(d)
    if n_views >= 3:
        c = float(np.cos(np.deg2rad(pitch_deg)))
        s = float(np.sin(np.deg2rad(pitch_deg)))
        d = base.clone()
        d[1, 1] = c
        d[1, 2] = -s
        d[2, 1] = s
        d[2, 2] = c
        d[1, 3] = vertical
        deltas.append(d)
    if n_views >= 4:
        d = base.clone()
        d[2, 3] = forward
        deltas.append(d)
    deltas = deltas[:n_views]

    delta_stack = torch.stack(deltas, dim=0)
    delta_stack = delta_stack.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
    return torch.einsum("bij,bkjl->bkil", anchor_c2w, delta_stack)

