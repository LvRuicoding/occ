"""DA3/OccAny+ reconstruction backbone adapter for KITTI Stage-1."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .. import _paths  # noqa: F401  (must come first)


_HF_TO_LOCAL_CONFIG = {
    "depth-anything/da3-giant-1.1": "da3-giant",
    "depth-anything/da3-large-1.1": "da3-large",
    "depth-anything/da3-base-1.1": "da3-base",
    "depth-anything/da3-small-1.1": "da3-small",
}


def _normalize_da3_model_name(name: object) -> str:
    value = str(name or "da3-giant")
    key = value.lower()
    if key in _HF_TO_LOCAL_CONFIG:
        return _HF_TO_LOCAL_CONFIG[key]
    if key.startswith("depth-anything/"):
        short = key.rsplit("/", 1)[-1]
        if short.endswith("-1.1"):
            short = short[: -len("-1.1")]
        return short
    if key.startswith("da3"):
        return key
    return value


def _replace_xformers_swiglu(module: nn.Module) -> int:
    """Replace xFormers fused SwiGLU with the vendored PyTorch fallback.

    The current occany env has xFormers installed, so DA3-GIANT constructs
    SwiGLUFFNFused modules, but its C++ operator may be unavailable at runtime.
    SwiGLUFFN uses the same w12/w3 state_dict layout and avoids that operator.
    """
    from depth_anything_3.model.dinov2.layers.swiglu_ffn import SwiGLUFFN, SwiGLUFFNFused

    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, SwiGLUFFNFused) and not isinstance(child, SwiGLUFFN):
            if getattr(child, "w12", None) is None:
                raise RuntimeError("Expected DA3 SwiGLUFFNFused to use packed w12 weights.")
            fallback = SwiGLUFFN(
                in_features=child.w12.in_features,
                hidden_features=child.w3.in_features,
                out_features=child.w3.out_features,
                bias=child.w12.bias is not None,
            )
            fallback.load_state_dict(child.state_dict(), strict=True)
            setattr(module, name, fallback)
            replaced += 1
        else:
            replaced += _replace_xformers_swiglu(child)
    return replaced


class OccAnyDA3Recon5FrameBackbone(nn.Module):
    """DA3 wrapper with the same Stage-1 output contract as OccAnyRecon5FrameBackbone."""

    def __init__(
        self,
        img_size: Tuple[int, int] = (168, 518),
        embed_dim: int = 3072,
        patch_size: int = 14,
        backbone_dtype: torch.dtype = torch.bfloat16,
        freeze: bool = True,
        model_input_size: int = 518,
    ) -> None:
        super().__init__()
        self.img_size = tuple(int(v) for v in img_size)
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.backbone_dtype = backbone_dtype
        self.freeze = bool(freeze)
        self.model_input_size = int(model_input_size)

        from occany.model.model_da3 import DA3Wrapper

        self.model = DA3Wrapper(
            model_name="da3-giant",
            img_size=self.model_input_size,
            projection_features="pts3d_local,pts3d,rgb,conf",
        )
        replaced = _replace_xformers_swiglu(self.model)
        if replaced:
            print(f"[OccAnyDA3Recon5FrameBackbone] replaced {replaced} xFormers SwiGLU modules.")
        self.register_buffer(
            "_imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.set_frozen(self.freeze)

    def load_checkpoint(self, ckpt_path: str) -> None:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        args = ckpt.get("args", None)
        da3_model_name = _normalize_da3_model_name(getattr(args, "da3_model_name", "da3-giant"))
        projection_features = getattr(args, "projection_features", "pts3d_local,pts3d,rgb,conf")
        model_input_size = int(getattr(args, "img_size", self.model_input_size))
        if da3_model_name != self.model.model_name:
            raise RuntimeError(
                f"DA3 checkpoint uses {da3_model_name!r}, but the Stage-1 adapter was "
                f"constructed as {self.model.model_name!r}."
            )
        self.model.img_size = model_input_size
        self.model.projection_features = projection_features
        status = self.model.load_state_dict(ckpt.get("model", {}), strict=False)
        print(
            f"[OccAnyDA3Recon5FrameBackbone] model={da3_model_name} "
            f"load: missing={len(status.missing_keys)} unexpected={len(status.unexpected_keys)}"
        )
        del ckpt

        self.model_input_size = model_input_size
        self.set_frozen(self.freeze)

    def set_frozen(self, freeze: bool = True) -> None:
        self.freeze = bool(freeze)
        for p in self.model.parameters():
            p.requires_grad = not self.freeze
        if self.freeze:
            self.model.eval()
        else:
            self.model.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def _stack_da3_images(self, views: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        images = torch.stack([v["img"] for v in views], dim=1).float()
        images = (images + 1.0) * 0.5
        return (images - self._imagenet_mean.to(images)) / self._imagenet_std.to(images)

    def forward(
        self,
        views: List[Dict[str, torch.Tensor]],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        if device is None:
            device = views[0]["img"].device
        model = self.model
        images = self._stack_da3_images(views).to(device=device)
        B, nimgs, _C, H, W = images.shape
        H_t = H // self.patch_size
        W_t = W // self.patch_size

        grad_context = torch.no_grad() if self.freeze else nullcontext()
        with grad_context, torch.autocast(device_type=device.type, dtype=self.backbone_dtype):
            out_layers = list(model.get_backbone_metadata()["out_layers"])
            feats, _aux_feats = model.model.backbone(
                images,
                cam_token=None,
                export_feat_layers=out_layers,
                ref_view_strategy="first",
            )
            depth_out = model._process_depth_output(
                feats=feats,
                h=H,
                w=W,
                device_type=device.type,
                pose_from_depth_ray=False,
                pose_from_cam_dec=False,
                point_from_depth_and_pose=False,
                images=images,
            )

        tokens = feats[-1][0]
        if tokens.shape[:3] != (B, nimgs, H_t * W_t):
            raise RuntimeError(
                "DA3 token grid mismatch: "
                f"got {tuple(tokens.shape)}, expected (B={B}, N={nimgs}, {H_t * W_t}, C)."
            )
        if tokens.shape[-1] != self.embed_dim:
            raise RuntimeError(
                f"DA3 token dim mismatch: got {tokens.shape[-1]}, expected {self.embed_dim}. "
                "Set --token_dim to match the DA3 config/checkpoint."
            )

        t_rec = tokens.float().reshape(B, nimgs, H_t, W_t, self.embed_dim).contiguous()
        p_rec_global = depth_out["pointmap"].float()
        p_rec_local = p_rec_global
        c_rec = depth_out["depth_conf"].float()

        if self.freeze:
            t_rec = t_rec.detach()
            p_rec_global = p_rec_global.detach()
            c_rec = c_rec.detach()

        return {
            "t_rec": t_rec,
            "p_rec_global": p_rec_global,
            "p_rec_local": p_rec_local,
            "c_rec": c_rec,
        }
