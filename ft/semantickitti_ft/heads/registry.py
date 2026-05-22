"""Small registry for swappable SSC heads."""
from __future__ import annotations

from typing import Callable, Dict, Type

import torch.nn as nn


_HEADS: Dict[str, Type[nn.Module]] = {}


def register_head(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    def _decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _HEADS:
            raise KeyError(f"Duplicate head registration: {name}")
        _HEADS[name] = cls
        return cls

    return _decorator


def build_head(name: str, **kwargs) -> nn.Module:
    if name not in _HEADS:
        raise KeyError(f"Unknown head={name!r}. Available: {available_heads()}")
    return _HEADS[name](**kwargs)


def available_heads() -> tuple[str, ...]:
    return tuple(sorted(_HEADS))

