"""Occupancy head registry."""
from .registry import available_heads, build_head, register_head

# Import modules for registration side effects.
from .monoscene import MonoSceneSSCHead  # noqa: F401

__all__ = [
    "MonoSceneSSCHead",
    "available_heads",
    "build_head",
    "register_head",
]

