"""Indicator registry with auto-discovery.

Adding an indicator is a single new file — no existing file is edited:

    # app/indicators/rsi.py
    from .base import Indicator, IndicatorResult, RenderSpec, SeriesSpec, PANE_OWN
    from . import register

    @register
    class Rsi(Indicator):
        key, name, order = "rsi", "RSI", 40
        render = RenderSpec(pane=PANE_OWN, series=(SeriesSpec("rsi", "RSI"),))
        def compute(self, ctx): ...

Restart the server and it appears in the API payload, gets its own chart pane
and its stats show up in the metrics strip automatically.
"""

from __future__ import annotations

import sys

from ..core.plugins import discover
from .base import (
    PANE_OWN,
    PANE_PRICE,
    Indicator,
    IndicatorContext,
    IndicatorResult,
    Marker,
    RenderSpec,
    SeriesSpec,
    Stat,
)

_REGISTRY: dict[str, Indicator] = {}
_loaded = False


def register(cls: type[Indicator]) -> type[Indicator]:
    if not cls.key:
        raise ValueError(f"{cls.__name__} must define a non-empty `key`")
    if cls.key in _REGISTRY:
        raise ValueError(f"indicator {cls.key!r} already registered")
    _REGISTRY[cls.key] = cls()
    return cls


def _ensure_loaded() -> None:
    global _loaded
    if not _loaded:
        _loaded = True
        discover(sys.modules[__name__])


def all_indicators() -> list[Indicator]:
    _ensure_loaded()
    return sorted(_REGISTRY.values(), key=lambda i: (i.order, i.key))


def get(key: str) -> Indicator:
    _ensure_loaded()
    return _REGISTRY[key]


__all__ = [
    "Indicator",
    "IndicatorContext",
    "IndicatorResult",
    "Marker",
    "RenderSpec",
    "SeriesSpec",
    "Stat",
    "PANE_OWN",
    "PANE_PRICE",
    "register",
    "all_indicators",
    "get",
]
