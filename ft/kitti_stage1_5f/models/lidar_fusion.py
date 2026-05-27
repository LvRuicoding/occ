"""LiDAR-image fusion module that updates OccAny's reconstruction tokens.

Pipeline (per (sample, frame) pair, vectorized across the batch):
  1) Voxelize raw LiDAR points (in their velo frame) into a per-frame cam-frame
     grid, run a PointPillars-style per-point MLP + per-voxel max-pool → sparse
     voxel features.
  2) Project each non-empty voxel center via the per-frame K (already matched to
     the resized image) → (h_t, w_t) patch coordinate, drop voxels with z_cam<=0
     or that fall outside the (H_t, W_t) patch grid.
  3) Apply two Swin-style windowed fusion layers (window=(4,4), shift=(0,0)
     then shift=(2,2)). The default Stage-1 LiDAR model uses self-attention over
     [image tokens, projected voxel tokens] with modality embeddings; the
     original image-query / voxel-KV cross-attention path is still available.
  4) Output is the input image feature with a residual update applied to image
     tokens only; voxel tokens are not forwarded to lifting.

In the current Stage-1 pipeline this module operates on ``t_rec`` (the
decoder-side patch tokens at D=768), inserted between OccAny's decoder and the
lifting module. The OccAny backbone stays fully frozen; gradients flow through
the fusion params (VFE / attention / FFN) and into downstream lifting + head.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# VFE (per-frame, in cam coords)
# ============================================================================


class VoxelFeatureEncoder(nn.Module):
    """PointPillars-style PointNet on cam-frame voxels.

    Inputs are raw points in the velodyne frame of the same timestep;
    a ``T_cam_from_velo`` (4x4) transforms them into cam coords prior to
    voxelization. We voxelize at the cam-frame grid defined by
    ``(vox_origin, vox_size, vox_grid)`` (all in cam coords:
    x→right, y→down, z→forward).

    Output:
      - ``voxel_feat``: (V, d_out) features for the non-empty voxels.
      - ``voxel_center_cam``: (V, 3) cam-frame center coordinate per voxel.
    """

    def __init__(
        self,
        vox_origin: Tuple[float, float, float] = (-25.6, -2.0, 0.0),
        vox_size: Tuple[float, float, float] = (0.4, 0.4, 0.4),
        vox_grid: Tuple[int, int, int] = (128, 16, 128),
        d_voxel: int = 128,
        d_out: int = 768,
        hidden: int = 64,
        pe_num_freqs: int = 8,
        point_mlp: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "vox_origin", torch.tensor(vox_origin, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "vox_size", torch.tensor(vox_size, dtype=torch.float32), persistent=False
        )
        self.vox_grid: Tuple[int, int, int] = tuple(int(v) for v in vox_grid)

        # Per-point input features: (x, y, z, intensity, dx_c, dy_c, dz_c) = 7.
        # If a shared module is provided it must output d_voxel-dim features.
        if point_mlp is not None:
            self.point_mlp = point_mlp
        else:
            self.point_mlp = nn.Sequential(
                nn.Linear(7, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, d_voxel),
            )
        self.voxel_norm = nn.LayerNorm(d_voxel)
        self.voxel_proj = nn.Linear(d_voxel, d_out)

        # Sinusoidal 3D PE on voxel center (cam coords), projected to d_out and added.
        self.pe_num_freqs = int(pe_num_freqs)
        pe_dim = 3 * 2 * self.pe_num_freqs
        self.pe_proj = nn.Linear(pe_dim, d_out)

    @staticmethod
    def _sinusoidal_pe_3d(coords: torch.Tensor, num_freqs: int) -> torch.Tensor:
        """coords: (..., 3) → (..., 3*2*num_freqs)."""
        device = coords.device
        dtype = coords.dtype
        # Use geometric base-2 freqs, scaled by pi.
        freqs = (2.0 ** torch.arange(num_freqs, device=device, dtype=dtype)) * math.pi
        # Normalize coordinate magnitudes loosely so high freqs don't alias too
        # fast — divide by an "effective range" of 50 m.
        scaled = coords / 50.0
        x = scaled.unsqueeze(-1) * freqs  # (..., 3, F)
        sin_part = x.sin()
        cos_part = x.cos()
        return torch.cat([sin_part, cos_part], dim=-1).flatten(-2)  # (..., 6F)

    def forward(
        self,
        points_velo: torch.Tensor,       # (P, 4) (x, y, z, intensity), float32
        T_cam_from_velo: torch.Tensor,   # (4, 4), float32
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns (voxel_feat, voxel_center_cam) or (None, None) if no voxel survives.

        Computed in input dtype (float32 by default); caller can cast as needed.
        """
        if points_velo.shape[0] == 0:
            return None, None

        # Transform velo→cam: (P, 3) using the rigid transform.
        # Up-cast points to float for the geometry to avoid precision loss under bf16.
        T = T_cam_from_velo.to(dtype=torch.float32)
        p_velo = points_velo[:, :3].to(dtype=torch.float32)
        intensity = points_velo[:, 3:4].to(dtype=torch.float32)
        R = T[:3, :3]
        t = T[:3, 3]
        p_cam = p_velo @ R.T + t  # (P, 3)

        vox_origin = self.vox_origin
        vox_size = self.vox_size
        Gx, Gy, Gz = self.vox_grid

        idx_f = (p_cam - vox_origin) / vox_size
        idx = idx_f.floor().long()  # (P, 3)

        valid = (
            (idx[:, 0] >= 0) & (idx[:, 0] < Gx)
            & (idx[:, 1] >= 0) & (idx[:, 1] < Gy)
            & (idx[:, 2] >= 0) & (idx[:, 2] < Gz)
        )
        if int(valid.sum().item()) == 0:
            return None, None

        idx = idx[valid]
        p_cam = p_cam[valid]
        intensity = intensity[valid]

        voxel_center = (idx.to(p_cam.dtype) + 0.5) * vox_size + vox_origin  # (P_valid, 3)
        rel = p_cam - voxel_center

        point_feat = torch.cat([p_cam, intensity, rel], dim=-1)  # (P_valid, 7)

        # Linear voxel index for grouping.
        lin = (idx[:, 0] * Gy + idx[:, 1]) * Gz + idx[:, 2]
        # Compact to dense [0..V) ids via unique.
        uniq_lin, inverse = torch.unique(lin, return_inverse=True)
        V = int(uniq_lin.shape[0])

        # Per-point MLP (cast to model dtype via autocast if active).
        h = self.point_mlp(point_feat)  # (P_valid, d_voxel)
        d_voxel = h.shape[-1]

        # Per-voxel max pool via scatter_reduce.
        neg_inf = torch.finfo(h.dtype).min
        voxel_feat = torch.full(
            (V, d_voxel), neg_inf, dtype=h.dtype, device=h.device
        )
        voxel_feat.scatter_reduce_(
            0,
            inverse.unsqueeze(-1).expand(-1, d_voxel),
            h,
            reduce="amax",
            include_self=False,
        )
        # Any voxel with no contributing point (shouldn't happen due to inverse) → zero.
        voxel_feat = voxel_feat.masked_fill(voxel_feat == neg_inf, 0.0)

        # Voxel centers (decompose uniq_lin).
        u_x = uniq_lin // (Gy * Gz)
        u_y = (uniq_lin // Gz) % Gy
        u_z = uniq_lin % Gz
        uniq_idx = torch.stack([u_x, u_y, u_z], dim=-1).to(p_cam.dtype)
        voxel_center_cam = (uniq_idx + 0.5) * vox_size + vox_origin  # (V, 3)

        # Project to d_out + 3D positional encoding.
        voxel_feat = self.voxel_norm(voxel_feat)
        voxel_proj = self.voxel_proj(voxel_feat)
        pe = self._sinusoidal_pe_3d(voxel_center_cam, self.pe_num_freqs).to(voxel_proj.dtype)
        voxel_proj = voxel_proj + self.pe_proj(pe)

        return voxel_proj, voxel_center_cam


# ============================================================================
# Windowed Attention
# ============================================================================


def _build_window_layout(
    H_t: int, W_t: int, window: int, shift: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Precompute the static (frame-relative) window <-> patch layout.

    Args:
        H_t, W_t: patch grid size.
        window:   window size (assumed square).
        shift:    spatial shift applied to the grid origin (0 for W-MSA,
                  window//2 for SW-MSA).

    Returns:
        win_h_grid, win_w_grid: (n_win, M_Q) int64 — per-window list of (h_t, w_t)
                                slot coords. Padded with 0 where invalid.
        win_q_mask:             (n_win, M_Q) bool — True where the slot is a
                                real in-bounds patch.
        n_win:                  total window count.
    """
    M_Q = window * window
    n_h = math.ceil((H_t + shift) / window)
    n_w = math.ceil((W_t + shift) / window)
    # Window i along H covers h_t in [i*window - shift, (i+1)*window - shift - 1].
    win_h_grid = torch.zeros((n_h * n_w, M_Q), dtype=torch.long)
    win_w_grid = torch.zeros((n_h * n_w, M_Q), dtype=torch.long)
    win_q_mask = torch.zeros((n_h * n_w, M_Q), dtype=torch.bool)
    for wh in range(n_h):
        h_start = wh * window - shift
        h_end = h_start + window
        for ww in range(n_w):
            w_start = ww * window - shift
            w_end = w_start + window
            win_id = wh * n_w + ww
            slot = 0
            for h in range(h_start, h_end):
                for w in range(w_start, w_end):
                    if 0 <= h < H_t and 0 <= w < W_t:
                        win_h_grid[win_id, slot] = h
                        win_w_grid[win_id, slot] = w
                        win_q_mask[win_id, slot] = True
                    slot += 1
    return win_h_grid, win_w_grid, win_q_mask, n_h * n_w


class WindowedCrossAttnLayer(nn.Module):
    """One layer: cross-attn (Q=image patch tokens, KV=voxel features) + FFN."""

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 8,
        window: int = 4,
        shift: int = 0,
        H_t: int = 10,
        W_t: int = 32,
        ffn_ratio: float = 2.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by num_heads {num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window = window
        self.shift = shift
        self.H_t = H_t
        self.W_t = W_t
        self.attn_dropout = attn_dropout

        wh_grid, ww_grid, q_mask, n_win = _build_window_layout(H_t, W_t, window, shift)
        self.register_buffer("win_h_grid", wh_grid, persistent=False)  # (n_win, M_Q)
        self.register_buffer("win_w_grid", ww_grid, persistent=False)
        self.register_buffer("win_q_mask", q_mask, persistent=False)
        self.n_win = int(n_win)
        self.M_Q = window * window
        # number of windows along width — needed to bucket voxels by window.
        self.n_w = int(math.ceil((W_t + shift) / window))

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm_ffn = nn.LayerNorm(d_model)
        hidden = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def _voxel_window_id(
        self, voxel_h_t: torch.Tensor, voxel_w_t: torch.Tensor
    ) -> torch.Tensor:
        """Map per-voxel (h_t, w_t) to a frame-relative flat window id."""
        wh = (voxel_h_t + self.shift) // self.window
        ww = (voxel_w_t + self.shift) // self.window
        return wh * self.n_w + ww

    def forward(
        self,
        image_feat: torch.Tensor,        # (F, H_t, W_t, D)
        voxel_feat: torch.Tensor,        # (V_total, D), already projected to D
        voxel_frame_idx: torch.Tensor,   # (V_total,) int64, 0..F-1
        voxel_h_t: torch.Tensor,         # (V_total,) int64
        voxel_w_t: torch.Tensor,         # (V_total,) int64
        voxel_valid: torch.Tensor,       # (V_total,) bool
    ) -> torch.Tensor:
        F_n, H_t, W_t, D = image_feat.shape
        device = image_feat.device

        # If nothing valid → identity.
        if voxel_valid.sum().item() == 0:
            return image_feat

        # Restrict to valid voxels.
        vf_idx = voxel_frame_idx[voxel_valid]
        vh_t = voxel_h_t[voxel_valid]
        vw_t = voxel_w_t[voxel_valid]
        v_feat = voxel_feat[voxel_valid]                          # (V, D)
        v_win_local = self._voxel_window_id(vh_t, vw_t)            # (V,)
        v_global = vf_idx * self.n_win + v_win_local              # (V,)

        # Sort voxels by global window id so we can compute per-window slot offsets.
        order = torch.argsort(v_global)
        v_global = v_global[order]
        v_feat = v_feat[order]

        # Active windows = unique global window ids that received voxels.
        active, inv, counts = torch.unique_consecutive(
            v_global, return_inverse=True, return_counts=True
        )
        n_active = int(active.shape[0])
        M_KV = int(counts.max().item())

        # Slot index within each active window via cumulative offset within group.
        # offsets per group: start = cumsum(counts) shifted.
        starts = torch.zeros_like(counts)
        starts[1:] = counts.cumsum(0)[:-1]
        slot = torch.arange(v_global.shape[0], device=device) - starts[inv]

        # Pack KV.
        kv_pad = torch.zeros((n_active, M_KV, D), dtype=v_feat.dtype, device=device)
        kv_mask = torch.zeros((n_active, M_KV), dtype=torch.bool, device=device)
        kv_pad[inv, slot] = v_feat
        kv_mask[inv, slot] = True

        # Pack Q from image_feat. For each active window, gather its M_Q patch tokens.
        active_frame = active // self.n_win
        active_local = active % self.n_win

        wh_grid_sel = self.win_h_grid[active_local]  # (n_active, M_Q)
        ww_grid_sel = self.win_w_grid[active_local]
        q_mask = self.win_q_mask[active_local]       # (n_active, M_Q) bool
        # Replace invalid slot coords with 0 (safe gather; we'll mask them).
        wh_safe = wh_grid_sel
        ww_safe = ww_grid_sel
        q_pad = image_feat[active_frame.unsqueeze(-1).expand_as(wh_safe), wh_safe, ww_safe]
        # q_pad: (n_active, M_Q, D)

        # Cross-attention.
        q_norm = self.norm_q(q_pad)
        kv_norm = self.norm_kv(kv_pad)
        q = self.q_proj(q_norm)
        k = self.k_proj(kv_norm)
        v = self.v_proj(kv_norm)

        # Reshape for multi-head.
        Hh = self.num_heads
        Dh = self.head_dim
        q = q.view(n_active, self.M_Q, Hh, Dh).transpose(1, 2)  # (n_active, Hh, M_Q, Dh)
        k = k.view(n_active, M_KV, Hh, Dh).transpose(1, 2)
        v = v.view(n_active, M_KV, Hh, Dh).transpose(1, 2)

        # attn_mask: True = allow attending. Broadcast over heads.
        # Shape (n_active, 1, M_Q, M_KV) bool. KV positions invalid → False.
        attn_mask = kv_mask.view(n_active, 1, 1, M_KV).expand(n_active, 1, self.M_Q, M_KV)
        # Note: if an entire row of the mask is False, SDPA returns NaN — but we
        # only build active windows that have >=1 valid voxel by construction.
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_dropout if self.training else 0.0
        )  # (n_active, Hh, M_Q, Dh)
        attn_out = attn_out.transpose(1, 2).contiguous().view(n_active, self.M_Q, D)
        attn_out = self.out_proj(attn_out)  # zero-init at start → 0

        # Mask out invalid Q slots before scattering.
        attn_out = attn_out * q_mask.unsqueeze(-1).to(attn_out.dtype)

        # FFN on the post-attention features (apply only to valid Q positions).
        # Build a "current Q" representation = q_pad + attn_out (residual 1).
        q_res = q_pad + attn_out
        ffn_in = self.norm_ffn(q_res)
        ffn_out = self.ffn(ffn_in)  # zero-init at start → 0
        ffn_out = ffn_out * q_mask.unsqueeze(-1).to(ffn_out.dtype)

        # Scatter the residual updates back to image_feat. For each valid (active,
        # slot), add (attn_out + ffn_out) to image_feat[frame, h, w].
        update = (attn_out + ffn_out)  # (n_active, M_Q, D)

        # Gather destination indices.
        valid_pos = q_mask  # (n_active, M_Q)
        # Output image (we modify a copy to avoid in-place issues with autograd).
        out = image_feat.clone()
        # Use vectorized fancy indexing.
        frame_idx_exp = active_frame.unsqueeze(-1).expand_as(wh_safe)[valid_pos]
        h_idx = wh_safe[valid_pos]
        w_idx = ww_safe[valid_pos]
        upd = update[valid_pos]
        out[frame_idx_exp, h_idx, w_idx] = out[frame_idx_exp, h_idx, w_idx] + upd
        return out


class WindowedSelfAttnLayer(nn.Module):
    """One layer: windowed self-attn over image tokens + projected voxel tokens.

    For every image window, the sequence is:
      [valid/padded image patch slots, voxel tokens projected into this window]

    Modality embeddings are added before attention, but only image-token residual
    updates are scattered back to the image feature map. Voxel tokens provide
    context for the window and are not propagated to downstream lifting.
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 8,
        window: int = 4,
        shift: int = 0,
        H_t: int = 10,
        W_t: int = 32,
        ffn_ratio: float = 2.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by num_heads {num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window = window
        self.shift = shift
        self.H_t = H_t
        self.W_t = W_t
        self.attn_dropout = attn_dropout

        wh_grid, ww_grid, q_mask, n_win = _build_window_layout(H_t, W_t, window, shift)
        self.register_buffer("win_h_grid", wh_grid, persistent=False)
        self.register_buffer("win_w_grid", ww_grid, persistent=False)
        self.register_buffer("win_q_mask", q_mask, persistent=False)
        self.n_win = int(n_win)
        self.M_Q = window * window
        self.n_w = int(math.ceil((W_t + shift) / window))

        self.modality_embed = nn.Embedding(2, d_model)
        self.norm_attn = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm_ffn = nn.LayerNorm(d_model)
        hidden = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def _voxel_window_id(
        self, voxel_h_t: torch.Tensor, voxel_w_t: torch.Tensor
    ) -> torch.Tensor:
        """Map per-voxel (h_t, w_t) to a frame-relative flat window id."""
        wh = (voxel_h_t + self.shift) // self.window
        ww = (voxel_w_t + self.shift) // self.window
        return wh * self.n_w + ww

    def forward(
        self,
        image_feat: torch.Tensor,        # (F, H_t, W_t, D)
        voxel_feat: torch.Tensor,        # (V_total, D), already projected to D
        voxel_frame_idx: torch.Tensor,   # (V_total,) int64, 0..F-1
        voxel_h_t: torch.Tensor,         # (V_total,) int64
        voxel_w_t: torch.Tensor,         # (V_total,) int64
        voxel_valid: torch.Tensor,       # (V_total,) bool
    ) -> torch.Tensor:
        F_n, H_t, W_t, D = image_feat.shape
        device = image_feat.device
        dtype = image_feat.dtype
        if (H_t, W_t, D) != (self.H_t, self.W_t, self.d_model):
            raise RuntimeError(
                f"image_feat shape ({H_t},{W_t},{D}) != layer shape "
                f"({self.H_t},{self.W_t},{self.d_model})."
            )

        valid_voxel_by_window = {}
        if voxel_valid.numel() > 0 and bool(voxel_valid.any().item()):
            vf_idx = voxel_frame_idx[voxel_valid]
            vh_t = voxel_h_t[voxel_valid]
            vw_t = voxel_w_t[voxel_valid]
            v_feat = voxel_feat[voxel_valid].to(dtype=dtype)
            v_win_local = self._voxel_window_id(vh_t, vw_t)
            v_global = vf_idx * self.n_win + v_win_local

            order = torch.argsort(v_global)
            v_global = v_global[order]
            v_feat = v_feat[order]
            active, counts = torch.unique_consecutive(v_global, return_counts=True)
            starts = torch.zeros_like(counts)
            starts[1:] = counts.cumsum(0)[:-1]
            for i in range(int(active.shape[0])):
                start = int(starts[i].item())
                end = start + int(counts[i].item())
                valid_voxel_by_window[int(active[i].item())] = v_feat[start:end]

        out = image_feat.clone()
        image_mod = self.modality_embed.weight[0].to(dtype=dtype)
        voxel_mod = self.modality_embed.weight[1].to(dtype=dtype)

        Hh = self.num_heads
        Dh = self.head_dim
        for global_win in range(F_n * self.n_win):
            frame_idx = global_win // self.n_win
            local_win = global_win % self.n_win

            h_grid = self.win_h_grid[local_win]
            w_grid = self.win_w_grid[local_win]
            q_mask = self.win_q_mask[local_win]
            img_tokens = image_feat[frame_idx, h_grid, w_grid]  # (M_Q, D)

            vox_tokens = valid_voxel_by_window.get(global_win)
            if vox_tokens is None:
                base_tokens = img_tokens
                token_valid = q_mask
                tokens = base_tokens + image_mod
            else:
                base_tokens = torch.cat([img_tokens, vox_tokens], dim=0)
                voxel_valid_mask = torch.ones(
                    (vox_tokens.shape[0],), dtype=torch.bool, device=device
                )
                token_valid = torch.cat([q_mask, voxel_valid_mask], dim=0)
                img_with_mod = img_tokens + image_mod
                vox_with_mod = vox_tokens + voxel_mod
                tokens = torch.cat([img_with_mod, vox_with_mod], dim=0)

            L = int(tokens.shape[0])
            attn_in = self.norm_attn(tokens)
            q = self.q_proj(attn_in).view(1, L, Hh, Dh).transpose(1, 2)
            k = self.k_proj(attn_in).view(1, L, Hh, Dh).transpose(1, 2)
            v = self.v_proj(attn_in).view(1, L, Hh, Dh).transpose(1, 2)

            attn_mask = token_valid.view(1, 1, 1, L).expand(1, 1, L, L)
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(L, D)
            attn_out = self.out_proj(attn_out)

            res_tokens = base_tokens + attn_out
            ffn_out = self.ffn(self.norm_ffn(res_tokens))
            update = attn_out + ffn_out

            img_update = update[: self.M_Q] * q_mask.unsqueeze(-1).to(dtype)
            valid_h = h_grid[q_mask]
            valid_w = w_grid[q_mask]
            out[frame_idx, valid_h, valid_w] = (
                out[frame_idx, valid_h, valid_w] + img_update[q_mask]
            )

        return out


class Sorted3DTokenFusionLayer(nn.Module):
    """Self-attention over 3D-sorted image and voxel token chunks.

    For each flattened frame independently:
      1) compute one local-camera 3D coordinate per image patch from the local
         pointmap using confidence-weighted pooling;
      2) quantize image and voxel coordinates into the same VFE grid;
      3) sort tokens by z -> y -> x serpentine order and split into fixed-length
         chunks;
      4) run self-attention only on chunks containing both image and voxel tokens;
      5) scatter updates back to image tokens only through a gated residual.
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 8,
        seq_len: int = 80,
        patch_size: int = 16,
        vox_origin: Tuple[float, float, float] = (-25.6, -2.0, 0.0),
        vox_size: Tuple[float, float, float] = (0.4, 0.4, 0.4),
        vox_grid: Tuple[int, int, int] = (128, 16, 128),
        ffn_ratio: float = 2.0,
        conf_clamp_max: float = 50.0,
        alpha_init: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by num_heads {num_heads}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = int(d_model) // int(num_heads)
        self.seq_len = int(seq_len)
        self.patch_size = int(patch_size)
        self.vox_grid: Tuple[int, int, int] = tuple(int(v) for v in vox_grid)
        self.conf_clamp_max = float(conf_clamp_max)
        self.attn_dropout = float(attn_dropout)

        self.register_buffer(
            "vox_origin", torch.tensor(vox_origin, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "vox_size", torch.tensor(vox_size, dtype=torch.float32), persistent=False
        )

        self.type_embed = nn.Embedding(2, d_model)
        self.norm_attn = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        hidden = int(d_model * ffn_ratio)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

    def _patch_points_from_pointmap(
        self,
        p_rec_local: torch.Tensor,  # (B, N, H_p, W_p, 3)
        c_rec: torch.Tensor,        # (B, N, H_p, W_p)
        H_t: int,
        W_t: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, H_p, W_p, _ = p_rec_local.shape
        if H_p != H_t * self.patch_size or W_p != W_t * self.patch_size:
            raise RuntimeError(
                f"local pointmap shape ({H_p},{W_p}) is incompatible with "
                f"patch grid ({H_t},{W_t}) and patch_size={self.patch_size}."
            )

        pts = p_rec_local.reshape(B, N, H_t, self.patch_size, W_t, self.patch_size, 3)
        conf = c_rec.reshape(B, N, H_t, self.patch_size, W_t, self.patch_size)
        finite = torch.isfinite(pts).all(dim=-1) & torch.isfinite(conf) & (conf > 0)
        weights = conf.clamp(max=self.conf_clamp_max) * finite.to(conf.dtype)
        sum_w = weights.sum(dim=(3, 5))
        sum_wp = (pts * weights.unsqueeze(-1)).sum(dim=(3, 5))
        patch_points = sum_wp / sum_w.clamp_min(1e-6).unsqueeze(-1)
        valid = sum_w > 0
        return patch_points, valid

    def _coord_to_grid(
        self,
        coords: torch.Tensor,  # (T, 3), local camera coords
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        coords_f = coords.to(dtype=torch.float32)
        idx = ((coords_f - self.vox_origin) / self.vox_size).floor().long()
        Gx, Gy, Gz = self.vox_grid
        valid = (
            torch.isfinite(coords_f).all(dim=-1)
            & (idx[:, 0] >= 0) & (idx[:, 0] < Gx)
            & (idx[:, 1] >= 0) & (idx[:, 1] < Gy)
            & (idx[:, 2] >= 0) & (idx[:, 2] < Gz)
        )
        return idx, valid

    def _serpentine_key(self, idx: torch.Tensor) -> torch.Tensor:
        """Return z-major, y-minor, x-minor serpentine scan keys."""
        Gx, Gy, _Gz = self.vox_grid
        ix = idx[:, 0]
        iy = idx[:, 1]
        iz = idx[:, 2]

        # Alternate y direction across z layers, then alternate x direction
        # across the scanned rows to preserve local continuity.
        y_scan = torch.where((iz % 2) == 0, iy, (Gy - 1) - iy)
        x_scan = torch.where(((iz + y_scan) % 2) == 0, ix, (Gx - 1) - ix)
        return (iz * Gy + y_scan) * Gx + x_scan

    def forward(
        self,
        image_feat: torch.Tensor,         # (B*N, H_t, W_t, D)
        p_rec_local: torch.Tensor,        # (B, N, H_p, W_p, 3)
        c_rec: torch.Tensor,              # (B, N, H_p, W_p)
        voxel_feat: torch.Tensor,         # (V_total, D)
        voxel_center_cam: torch.Tensor,   # (V_total, 3)
        voxel_frame_idx: torch.Tensor,    # (V_total,) int64, 0..B*N-1
    ) -> torch.Tensor:
        F_n, H_t, W_t, D = image_feat.shape
        B, N = p_rec_local.shape[:2]
        if B * N != F_n:
            raise RuntimeError(
                f"image frames ({F_n}) != local pointmap batch*frames ({B}*{N})."
            )
        if D != self.d_model:
            raise RuntimeError(f"image feature dim {D} != d_model {self.d_model}.")

        device = image_feat.device
        dtype = image_feat.dtype
        patch_points, patch_valid = self._patch_points_from_pointmap(
            p_rec_local.to(device=device),
            c_rec.to(device=device),
            H_t,
            W_t,
        )
        patch_points = patch_points.reshape(F_n, H_t * W_t, 3)
        patch_valid = patch_valid.reshape(F_n, H_t * W_t)

        out = image_feat.clone()
        img_feat_flat = image_feat.reshape(F_n, H_t * W_t, D)
        voxel_feat = voxel_feat.to(device=device, dtype=dtype)
        voxel_center_cam = voxel_center_cam.to(device=device)
        voxel_frame_idx = voxel_frame_idx.to(device=device)

        Hh = self.num_heads
        Dh = self.head_dim
        S = self.seq_len
        alpha = self.alpha.to(dtype=dtype)

        for frame_idx in range(F_n):
            img_idx = torch.nonzero(patch_valid[frame_idx], as_tuple=False).flatten()
            if img_idx.numel() == 0:
                continue

            img_coords = patch_points[frame_idx, img_idx]
            img_grid_idx, img_grid_valid = self._coord_to_grid(img_coords)
            if not bool(img_grid_valid.any().item()):
                continue
            img_idx = img_idx[img_grid_valid]
            img_grid_idx = img_grid_idx[img_grid_valid]
            img_tokens = img_feat_flat[frame_idx, img_idx]

            vox_mask = voxel_frame_idx == frame_idx
            if not bool(vox_mask.any().item()):
                continue
            vox_tokens = voxel_feat[vox_mask]
            vox_coords = voxel_center_cam[vox_mask]
            vox_grid_idx, vox_grid_valid = self._coord_to_grid(vox_coords)
            if not bool(vox_grid_valid.any().item()):
                continue
            vox_tokens = vox_tokens[vox_grid_valid]
            vox_grid_idx = vox_grid_idx[vox_grid_valid]

            tokens = torch.cat([img_tokens, vox_tokens], dim=0)
            grid_idx = torch.cat([img_grid_idx, vox_grid_idx], dim=0)
            type_ids = torch.cat(
                [
                    torch.zeros((img_tokens.shape[0],), dtype=torch.long, device=device),
                    torch.ones((vox_tokens.shape[0],), dtype=torch.long, device=device),
                ],
                dim=0,
            )
            image_dst = torch.cat(
                [
                    img_idx.to(device=device),
                    torch.full(
                        (vox_tokens.shape[0],),
                        -1,
                        dtype=torch.long,
                        device=device,
                    ),
                ],
                dim=0,
            )

            order = torch.argsort(self._serpentine_key(grid_idx))
            tokens = tokens[order]
            type_ids = type_ids[order]
            image_dst = image_dst[order]

            L = int(tokens.shape[0])
            n_chunks = (L + S - 1) // S
            padded = n_chunks * S
            tokens_pad = tokens.new_zeros((padded, D))
            valid_pad = torch.zeros((padded,), dtype=torch.bool, device=device)
            type_pad = torch.zeros((padded,), dtype=torch.long, device=device)
            image_dst_pad = torch.full((padded,), -1, dtype=torch.long, device=device)
            tokens_pad[:L] = tokens
            valid_pad[:L] = True
            type_pad[:L] = type_ids
            image_dst_pad[:L] = image_dst

            valid_chunks = valid_pad.reshape(n_chunks, S)
            type_chunks = type_pad.reshape(n_chunks, S)
            active = (
                (valid_chunks & (type_chunks == 0)).any(dim=1)
                & (valid_chunks & (type_chunks == 1)).any(dim=1)
            )
            if not bool(active.any().item()):
                continue

            token_chunks = tokens_pad.reshape(n_chunks, S, D)[active]
            valid_active = valid_chunks[active]
            type_active = type_chunks[active]
            image_dst_active = image_dst_pad.reshape(n_chunks, S)[active]

            token_chunks = token_chunks + (
                self.type_embed(type_active).to(dtype=dtype)
                * valid_active.unsqueeze(-1).to(dtype=dtype)
            )

            attn_in = self.norm_attn(token_chunks)
            n_active = int(attn_in.shape[0])
            q = self.q_proj(attn_in).view(n_active, S, Hh, Dh).transpose(1, 2)
            k = self.k_proj(attn_in).view(n_active, S, Hh, Dh).transpose(1, 2)
            v = self.v_proj(attn_in).view(n_active, S, Hh, Dh).transpose(1, 2)
            attn_mask = valid_active.view(n_active, 1, 1, S).expand(n_active, 1, S, S)
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(n_active, S, D)
            attn_tokens = token_chunks + self.out_proj(attn_out)
            update = self.mlp(self.norm_mlp(attn_tokens))

            image_token = valid_active & (type_active == 0) & (image_dst_active >= 0)
            if not bool(image_token.any().item()):
                continue
            dst = image_dst_active[image_token]
            upd = update[image_token] * alpha
            h_idx = dst // W_t
            w_idx = dst % W_t
            out[frame_idx, h_idx, w_idx] = out[frame_idx, h_idx, w_idx] + upd

        return out


# ============================================================================
# Dense target-grid VFE for post-lift 3D fusion
# ============================================================================


class TargetGridLidarFeatureEncoder(nn.Module):
    """Aggregate multi-frame LiDAR into the target/velo voxel grid.

    All sweeps are transformed into the target frame's velodyne system, a
    per-point frame embedding is added after the shared point MLP, and a single
    per-voxel max-pool is run across the union of all points. The output is the
    learned feature volume plus three diagnostic channels per voxel:
      - ``mask``      : 1 where any sweep deposited a point.
      - ``count``     : ``log1p`` saturating point-density in [0,1].
      - ``diversity`` : (number of distinct contributing frames) / N in [0,1].

    ``diversity`` separates stable multi-sweep occupancy from single-frame
    ghosting caused by dynamic objects.
    """

    def __init__(
        self,
        vox_origin: Tuple[float, float, float] = (0.0, -25.6, -2.0),
        base_vox_size: Tuple[float, float, float] = (0.4, 0.4, 0.4),
        base_grid: Tuple[int, int, int] = (128, 128, 16),
        full_grid: Tuple[int, int, int] = (256, 256, 32),
        d_voxel: int = 128,
        d_out: int = 32,
        hidden: int = 64,
        pe_num_freqs: int = 8,
        num_frames: int = 5,
        point_mlp: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "vox_origin", torch.tensor(vox_origin, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "base_vox_size", torch.tensor(base_vox_size, dtype=torch.float32), persistent=False
        )
        self.base_grid: Tuple[int, int, int] = tuple(int(v) for v in base_grid)
        self.full_grid: Tuple[int, int, int] = tuple(int(v) for v in full_grid)
        self.d_out = int(d_out)
        self.d_voxel = int(d_voxel)
        if self.d_out <= 0:
            raise ValueError(f"d_out must be positive, got {d_out}")
        if int(num_frames) <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        self.num_frames = int(num_frames)

        if point_mlp is not None:
            self.point_mlp = point_mlp
        else:
            self.point_mlp = nn.Sequential(
                nn.Linear(7, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, d_voxel),
            )
        # Per-point frame embedding added post-MLP so the shared point_mlp can
        # stay temporally agnostic (used as-is by the cam-frame VFE).
        self.time_embed = nn.Embedding(self.num_frames, self.d_voxel)
        nn.init.zeros_(self.time_embed.weight)

        self.voxel_norm = nn.LayerNorm(d_voxel)
        self.voxel_proj = nn.Linear(d_voxel, d_out)

        self.pe_num_freqs = int(pe_num_freqs)
        pe_dim = 3 * 2 * self.pe_num_freqs
        self.pe_proj = nn.Linear(pe_dim, d_out)

    def _transform_to_target(
        self,
        points_velo: torch.Tensor,
        T_target_from_frame_velo: torch.Tensor,
    ) -> torch.Tensor:
        T = T_target_from_frame_velo.to(dtype=torch.float32)
        p_velo = points_velo[:, :3].to(dtype=torch.float32)
        R = T[:3, :3]
        t = T[:3, 3]
        return p_velo @ R.T + t

    def _upsample_to_full(
        self,
        feat: torch.Tensor,
        mask: torch.Tensor,
        count: torch.Tensor,
        diversity: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.base_grid == self.full_grid:
            return feat, mask, count, diversity
        # Trilinear is fp32-only on some backends; cast then back to be safe.
        feat_f32 = feat.to(dtype=torch.float32)
        feat_f32 = F.interpolate(
            feat_f32, size=self.full_grid, mode="trilinear", align_corners=False
        )
        feat = feat_f32.to(dtype=feat.dtype)
        mask = F.interpolate(mask, size=self.full_grid, mode="nearest")
        count = F.interpolate(count, size=self.full_grid, mode="nearest")
        diversity = F.interpolate(diversity, size=self.full_grid, mode="nearest")
        return feat, mask, count, diversity

    def forward(
        self,
        points_per_frame: List[List[torch.Tensor]],   # [B][N] (P, 4), frame-velo coords
        T_cam_from_velo: torch.Tensor,                # (B, 4, 4)
        T_target_from_refcam: torch.Tensor,           # (B, 4, 4), target-velo <- target-cam
        cam2world_per_frame: torch.Tensor,            # (B, N, 4, 4), world <- cam_f
        output_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = len(points_per_frame)
        if B == 0:
            raise RuntimeError("points_per_frame must contain at least one sample.")
        N = len(points_per_frame[0])
        if cam2world_per_frame.shape[:2] != (B, N):
            raise RuntimeError(
                f"cam2world shape {tuple(cam2world_per_frame.shape[:2])} "
                f"does not match points_per_frame ({B}, {N})."
            )
        if N > self.num_frames:
            raise RuntimeError(
                f"input has {N} frames but encoder was built with "
                f"num_frames={self.num_frames}."
            )

        device = T_cam_from_velo.device
        out_dtype = output_dtype if output_dtype is not None else T_cam_from_velo.dtype
        Gx, Gy, Gz = self.base_grid
        n_base_voxels = B * Gx * Gy * Gz

        vox_origin = self.vox_origin.to(device=device, dtype=torch.float32)
        vox_size = self.base_vox_size.to(device=device, dtype=torch.float32)
        cam2world = cam2world_per_frame.to(device=device, dtype=torch.float32)

        # Accumulate per-point features across all (b, f) before a single
        # batch-wide max-pool. Each entry holds post-MLP + time-embedded
        # features so the pool keeps "winning point" semantics rather than
        # being diluted by an outer mean across sweeps.
        all_h: List[torch.Tensor] = []
        all_lin: List[torch.Tensor] = []
        all_frame: List[torch.Tensor] = []

        for b in range(B):
            T_cv = T_cam_from_velo[b].to(device=device, dtype=torch.float32)
            T_target_from_cam = T_target_from_refcam[b].to(device=device, dtype=torch.float32)
            T_cam_target_from_world = torch.linalg.inv(cam2world[b, 0])

            for f in range(N):
                pts = points_per_frame[b][f]
                if pts.shape[0] == 0:
                    continue
                pts = pts.to(device=device, non_blocking=True)

                T_target_from_frame_velo = (
                    T_target_from_cam
                    @ T_cam_target_from_world
                    @ cam2world[b, f]
                    @ T_cv
                )
                p_target = self._transform_to_target(pts, T_target_from_frame_velo)
                intensity = pts[:, 3:4].to(dtype=torch.float32)

                idx_f = (p_target - vox_origin) / vox_size
                idx = idx_f.floor().long()
                valid = (
                    torch.isfinite(p_target).all(dim=-1)
                    & torch.isfinite(intensity).flatten()
                    & (idx[:, 0] >= 0) & (idx[:, 0] < Gx)
                    & (idx[:, 1] >= 0) & (idx[:, 1] < Gy)
                    & (idx[:, 2] >= 0) & (idx[:, 2] < Gz)
                )
                if not bool(valid.any().item()):
                    continue

                idx = idx[valid]
                p_valid = p_target[valid]
                i_valid = intensity[valid]
                voxel_center = (idx.to(p_valid.dtype) + 0.5) * vox_size + vox_origin
                rel = p_valid - voxel_center
                point_feat = torch.cat([p_valid, i_valid, rel], dim=-1)  # (P, 7)

                h = self.point_mlp(point_feat)  # (P, d_voxel)
                P_valid = h.shape[0]
                frame_ids = torch.full(
                    (P_valid,), int(f), dtype=torch.long, device=device
                )
                h = h + self.time_embed(frame_ids).to(dtype=h.dtype)

                lin = (((b * Gx + idx[:, 0]) * Gy + idx[:, 1]) * Gz + idx[:, 2])
                all_h.append(h)
                all_lin.append(lin)
                all_frame.append(frame_ids)

        if len(all_h) == 0:
            feat = torch.zeros((B, self.d_out, Gx, Gy, Gz), device=device, dtype=out_dtype)
            mask = torch.zeros((B, 1, Gx, Gy, Gz), device=device, dtype=out_dtype)
            count = torch.zeros((B, 1, Gx, Gy, Gz), device=device, dtype=out_dtype)
            diversity = torch.zeros((B, 1, Gx, Gy, Gz), device=device, dtype=out_dtype)
            return self._upsample_to_full(feat, mask, count, diversity)

        h_cat = torch.cat(all_h, dim=0)
        lin_cat = torch.cat(all_lin, dim=0)
        frame_cat = torch.cat(all_frame, dim=0)

        # Single per-voxel max-pool across the union of all points.
        uniq_lin, inverse = torch.unique(lin_cat, return_inverse=True)
        U = int(uniq_lin.shape[0])
        d_voxel = h_cat.shape[-1]
        neg_inf = torch.finfo(h_cat.dtype).min
        voxel_feat = torch.full((U, d_voxel), neg_inf, dtype=h_cat.dtype, device=device)
        voxel_feat.scatter_reduce_(
            0,
            inverse.unsqueeze(-1).expand(-1, d_voxel),
            h_cat,
            reduce="amax",
            include_self=False,
        )
        voxel_feat = voxel_feat.masked_fill(voxel_feat == neg_inf, 0.0)

        voxel_feat = self.voxel_norm(voxel_feat)
        voxel_proj = self.voxel_proj(voxel_feat)

        local_lin = uniq_lin % (Gx * Gy * Gz)
        u_x = local_lin // (Gy * Gz)
        u_y = (local_lin // Gz) % Gy
        u_z = local_lin % Gz
        uniq_idx = torch.stack([u_x, u_y, u_z], dim=-1).to(dtype=torch.float32)
        voxel_center = (uniq_idx + 0.5) * vox_size + vox_origin
        pe = VoxelFeatureEncoder._sinusoidal_pe_3d(
            voxel_center, self.pe_num_freqs
        ).to(voxel_proj.dtype)
        voxel_proj = voxel_proj + self.pe_proj(pe)

        # Point density per voxel (log1p saturates so a few-hundred dense
        # voxels don't dwarf single-hit ones).
        point_count = torch.zeros((U,), dtype=torch.float32, device=device)
        point_count.index_add_(
            0, inverse, torch.ones_like(inverse, dtype=torch.float32)
        )
        count_norm = (torch.log1p(point_count) / math.log1p(64.0)).clamp_(0.0, 1.0)

        # Frame diversity = distinct contributing frames per voxel. Encode
        # (voxel, frame) pairs as a single int and unique them — the resulting
        # count per voxel is the diversity.
        pair = lin_cat * self.num_frames + frame_cat
        uniq_pair = torch.unique(pair)
        pair_lin = uniq_pair // self.num_frames
        # uniq_lin is sorted, so searchsorted yields the per-pair voxel id.
        pair_vox_idx = torch.searchsorted(uniq_lin, pair_lin)
        diversity = torch.zeros((U,), dtype=torch.float32, device=device)
        diversity.index_add_(
            0,
            pair_vox_idx,
            torch.ones_like(pair_vox_idx, dtype=torch.float32),
        )
        diversity_norm = (diversity / float(max(N, 1))).clamp_(0.0, 1.0)

        dense_feat = torch.zeros(
            (n_base_voxels, self.d_out), dtype=voxel_proj.dtype, device=device
        )
        dense_feat[uniq_lin] = voxel_proj
        mask_flat = torch.zeros((n_base_voxels, 1), dtype=voxel_proj.dtype, device=device)
        mask_flat[uniq_lin] = 1.0
        count_flat = torch.zeros((n_base_voxels, 1), dtype=voxel_proj.dtype, device=device)
        count_flat[uniq_lin] = count_norm.unsqueeze(-1).to(dtype=voxel_proj.dtype)
        div_flat = torch.zeros((n_base_voxels, 1), dtype=voxel_proj.dtype, device=device)
        div_flat[uniq_lin] = diversity_norm.unsqueeze(-1).to(dtype=voxel_proj.dtype)

        feat = dense_feat.view(B, Gx, Gy, Gz, self.d_out).permute(0, 4, 1, 2, 3).contiguous()
        mask = mask_flat.view(B, Gx, Gy, Gz, 1).permute(0, 4, 1, 2, 3).contiguous()
        count = count_flat.view(B, Gx, Gy, Gz, 1).permute(0, 4, 1, 2, 3).contiguous()
        diversity = div_flat.view(B, Gx, Gy, Gz, 1).permute(0, 4, 1, 2, 3).contiguous()

        feat, mask, count, diversity = self._upsample_to_full(
            feat, mask, count, diversity
        )
        return (
            feat.to(dtype=out_dtype),
            mask.to(dtype=out_dtype),
            count.to(dtype=out_dtype),
            diversity.to(dtype=out_dtype),
        )


# ============================================================================
# Top-level fusion module
# ============================================================================


class LidarImageFusionModule(nn.Module):
    """VFE + 2D windowed fusion, optionally followed by 3D sorted fusion.

    Input/output ``t_rec`` has shape (B, N, H_t, W_t, D). The optional 3D
    stage reuses the same VFE voxel features and local-camera voxel centers.
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 8,
        H_t: int = 10,
        W_t: int = 32,
        patch_size: int = 16,
        window: int = 4,
        vox_origin: Tuple[float, float, float] = (-25.6, -2.0, 0.0),
        vox_size: Tuple[float, float, float] = (0.4, 0.4, 0.4),
        vox_grid: Tuple[int, int, int] = (128, 16, 128),
        vfe_d_voxel: int = 128,
        vfe_hidden: int = 64,
        pe_num_freqs: int = 8,
        ffn_ratio: float = 2.0,
        attn_type: str = "cross",
        fusion3d_enabled: bool = False,
        fusion3d_seq_len: int = 80,
        fusion3d_num_heads: Optional[int] = None,
        fusion3d_ffn_ratio: float = 2.0,
        fusion3d_alpha_init: float = 0.0,
        fusion3d_conf_clamp_max: float = 50.0,
        vfe_point_mlp: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if attn_type not in ("cross", "self"):
            raise ValueError(f"attn_type must be 'cross' or 'self'; got {attn_type!r}")
        self.d_model = d_model
        self.H_t = H_t
        self.W_t = W_t
        self.patch_size = int(patch_size)
        self.attn_type = attn_type

        self.vfe = VoxelFeatureEncoder(
            vox_origin=vox_origin,
            vox_size=vox_size,
            vox_grid=vox_grid,
            d_voxel=vfe_d_voxel,
            d_out=d_model,
            hidden=vfe_hidden,
            pe_num_freqs=pe_num_freqs,
            point_mlp=vfe_point_mlp,
        )
        layer_cls = WindowedSelfAttnLayer if attn_type == "self" else WindowedCrossAttnLayer
        self.layer_w = layer_cls(
            d_model=d_model, num_heads=num_heads, window=window, shift=0,
            H_t=H_t, W_t=W_t, ffn_ratio=ffn_ratio,
        )
        self.layer_sw = layer_cls(
            d_model=d_model, num_heads=num_heads, window=window, shift=window // 2,
            H_t=H_t, W_t=W_t, ffn_ratio=ffn_ratio,
        )
        self.fusion3d = (
            Sorted3DTokenFusionLayer(
                d_model=d_model,
                num_heads=num_heads if fusion3d_num_heads is None else fusion3d_num_heads,
                seq_len=fusion3d_seq_len,
                patch_size=patch_size,
                vox_origin=vox_origin,
                vox_size=vox_size,
                vox_grid=vox_grid,
                ffn_ratio=fusion3d_ffn_ratio,
                conf_clamp_max=fusion3d_conf_clamp_max,
                alpha_init=fusion3d_alpha_init,
            )
            if fusion3d_enabled
            else None
        )

    def _project_voxels_to_patches(
        self,
        voxel_center_cam: torch.Tensor,  # (V, 3) cam-frame center
        K: torch.Tensor,                  # (3, 3)
        img_H: int,
        img_W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project voxel centers to (h_t, w_t) patch indices.

        Returns:
            h_t, w_t: (V,) int64 patch indices (clamped, but mask says which valid).
            valid:    (V,) bool.
        """
        K = K.to(dtype=torch.float32)
        center = voxel_center_cam.to(dtype=torch.float32)
        z = center[:, 2]
        eps = 1e-6
        valid_z = z > 0.1
        # Avoid div-by-zero for invalid points.
        z_safe = torch.where(valid_z, z, torch.ones_like(z))
        uv = (K @ center.T).T  # (V, 3); uv[:, :2] = K @ (X*z, Y*z, z)
        # Actually K @ [X, Y, Z] → [fx*X + cx*Z, fy*Y + cy*Z, Z]; divide by Z:
        u = uv[:, 0] / z_safe
        v = uv[:, 1] / z_safe
        valid = (
            valid_z
            & (u >= 0) & (u < img_W)
            & (v >= 0) & (v < img_H)
        )
        h_t = (v / self.patch_size).floor().long()
        w_t = (u / self.patch_size).floor().long()
        # Clamp safely; valid mask gates use.
        h_t = h_t.clamp(0, self.H_t - 1)
        w_t = w_t.clamp(0, self.W_t - 1)
        return h_t, w_t, valid

    def forward(
        self,
        t_rec: torch.Tensor,                          # (B, N, H_t, W_t, D)
        points_per_frame: List[List[torch.Tensor]],   # [B][N] (P_bn, 4) velo-frame
        T_cam_from_velo: torch.Tensor,                # (B, 4, 4)
        K_per_frame: torch.Tensor,                    # (B, N, 3, 3)
        image_hw: torch.Tensor,                       # (B, 2) (H, W)
        p_rec_local: Optional[torch.Tensor] = None,   # (B, N, H_p, W_p, 3)
        c_rec: Optional[torch.Tensor] = None,         # (B, N, H_p, W_p)
    ) -> torch.Tensor:
        B, N, H_t, W_t, D = t_rec.shape
        if (H_t, W_t) != (self.H_t, self.W_t):
            raise RuntimeError(
                f"t_rec patch grid ({H_t},{W_t}) != fusion grid "
                f"({self.H_t},{self.W_t}); check patch_size / image resolution."
            )
        # (B*N, H_t, W_t, D) layout used by the attention layers.
        image_feat = t_rec.reshape(B * N, self.H_t, self.W_t, D).contiguous()

        # Build voxel features for every (sample, frame) pair, then concat into
        # one large set with a per-voxel "global frame index" (flat 0..B*N-1).
        all_voxel_feats: List[torch.Tensor] = []
        all_voxel_centers: List[torch.Tensor] = []
        all_voxel_frame_idx: List[torch.Tensor] = []
        all_voxel_h_t: List[torch.Tensor] = []
        all_voxel_w_t: List[torch.Tensor] = []
        all_voxel_valid: List[torch.Tensor] = []

        device = t_rec.device
        for b in range(B):
            img_H = int(image_hw[b, 0].item())
            img_W = int(image_hw[b, 1].item())
            T_cv = T_cam_from_velo[b]  # (4, 4)
            for f in range(N):
                pts = points_per_frame[b][f]
                pts = pts.to(device=device, non_blocking=True)
                feat, center = self.vfe(pts, T_cv)
                if feat is None or feat.shape[0] == 0:
                    continue
                K = K_per_frame[b, f]  # (3, 3)
                h_t, w_t, valid = self._project_voxels_to_patches(center, K, img_H, img_W)
                global_frame = b * N + f
                frame_idx = torch.full((feat.shape[0],), global_frame, dtype=torch.long, device=device)
                all_voxel_feats.append(feat)
                all_voxel_centers.append(center)
                all_voxel_frame_idx.append(frame_idx)
                all_voxel_h_t.append(h_t)
                all_voxel_w_t.append(w_t)
                all_voxel_valid.append(valid)

        if len(all_voxel_feats) == 0:
            if self.attn_type == "cross":
                # No voxels at all in the entire batch — original cross-attn
                # behavior is identity.
                return t_rec
            voxel_feat_cat = image_feat.new_zeros((0, D))
            voxel_center_cat = image_feat.new_zeros((0, 3))
            voxel_frame_idx_cat = torch.zeros((0,), dtype=torch.long, device=device)
            voxel_h_t_cat = torch.zeros((0,), dtype=torch.long, device=device)
            voxel_w_t_cat = torch.zeros((0,), dtype=torch.long, device=device)
            voxel_valid_cat = torch.zeros((0,), dtype=torch.bool, device=device)
        else:
            voxel_feat_cat = torch.cat(all_voxel_feats, dim=0)
            voxel_center_cat = torch.cat(all_voxel_centers, dim=0)
            voxel_frame_idx_cat = torch.cat(all_voxel_frame_idx, dim=0)
            voxel_h_t_cat = torch.cat(all_voxel_h_t, dim=0)
            voxel_w_t_cat = torch.cat(all_voxel_w_t, dim=0)
            voxel_valid_cat = torch.cat(all_voxel_valid, dim=0)

        # Match dtype to the image features (autocast may have it in bf16).
        voxel_feat_cat = voxel_feat_cat.to(dtype=image_feat.dtype)

        # Apply two windowed cross-attention layers in series.
        image_feat = self.layer_w(
            image_feat,
            voxel_feat_cat,
            voxel_frame_idx_cat,
            voxel_h_t_cat,
            voxel_w_t_cat,
            voxel_valid_cat,
        )
        image_feat = self.layer_sw(
            image_feat,
            voxel_feat_cat,
            voxel_frame_idx_cat,
            voxel_h_t_cat,
            voxel_w_t_cat,
            voxel_valid_cat,
        )

        if self.fusion3d is not None:
            if p_rec_local is None or c_rec is None:
                raise RuntimeError(
                    "3D fusion is enabled, but local pointmap/confidence was not provided."
                )
            image_feat = self.fusion3d(
                image_feat=image_feat,
                p_rec_local=p_rec_local,
                c_rec=c_rec,
                voxel_feat=voxel_feat_cat,
                voxel_center_cam=voxel_center_cat,
                voxel_frame_idx=voxel_frame_idx_cat,
            )

        return image_feat.view(B, N, self.H_t, self.W_t, D)


__all__ = [
    "VoxelFeatureEncoder",
    "TargetGridLidarFeatureEncoder",
    "WindowedCrossAttnLayer",
    "WindowedSelfAttnLayer",
    "Sorted3DTokenFusionLayer",
    "LidarImageFusionModule",
]
