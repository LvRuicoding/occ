"""Small registry for swappable lifting modules."""
from __future__ import annotations

from typing import Callable, Dict, Type

import torch.nn as nn


_LIFTERS: Dict[str, Type[nn.Module]] = {}


def register_lifter(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    def _decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _LIFTERS:
            raise KeyError(f"Duplicate lifter registration: {name}")
        _LIFTERS[name] = cls
        return cls

    return _decorator


def build_lifter(name: str, **kwargs) -> nn.Module:
    if name not in _LIFTERS:
        raise KeyError(f"Unknown lift={name!r}. Available: {available_lifts()}")
    return _LIFTERS[name](**kwargs)


def available_lifts() -> tuple[str, ...]:
    return tuple(sorted(_LIFTERS))

