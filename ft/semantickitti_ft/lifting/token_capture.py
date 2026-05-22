"""Decoder token capture helpers."""
from __future__ import annotations

from typing import List

import torch

from occany.model.model_must3r import Must3rDecoder


class DecoderTokenCapturer:
    """Capture token outputs from the last `n_layers` decoder blocks."""

    def __init__(self, n_layers: int) -> None:
        self.n_layers = int(n_layers)
        self._captures: List[torch.Tensor] = []
        self._handles: List = []

    def _hook(self, module, inputs, output):
        self._captures.append(output)

    def attach(self, decoder: Must3rDecoder) -> None:
        if self._handles:
            raise RuntimeError("capturer already attached; detach() first")
        self._captures = []
        blocks = list(decoder.blocks_dec)
        for blk in blocks[-self.n_layers:]:
            self._handles.append(blk.register_forward_hook(self._hook))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def pop(self) -> List[torch.Tensor]:
        result = self._captures
        self._captures = []
        return result


def drop_pose_token(t_bxn_n_d: torch.Tensor, B: int, nimgs: int) -> torch.Tensor:
    """(B*nimgs, N, D) with first token = pose token -> (B, nimgs, N-1, D)."""
    if t_bxn_n_d.dim() != 3:
        raise ValueError(f"Expected (B*nimgs, N, D), got {tuple(t_bxn_n_d.shape)}")
    bxn, N, D = t_bxn_n_d.shape
    if bxn != B * nimgs:
        raise ValueError(f"shape mismatch: {bxn} != {B}*{nimgs}")
    no_pose = t_bxn_n_d[:, 1:, :]
    return no_pose.view(B, nimgs, N - 1, D).contiguous()

