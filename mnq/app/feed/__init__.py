"""Feed registry.

Adding a data source = adding one file to this package:

    # app/feed/mybroker.py
    from .base import PriceFeed
    from . import register

    @register
    class MyBrokerFeed(PriceFeed):
        key = "mybroker"
        name = "My Broker"
        async def fetch_intraday(self): ...
        async def fetch_daily(self): ...

Then run with ``MNQ_FEED=mybroker``.  No existing file is modified.
"""

from __future__ import annotations

import sys

from ..config import Settings
from ..core.plugins import discover
from .base import PriceFeed

_REGISTRY: dict[str, type[PriceFeed]] = {}


def register(cls: type[PriceFeed]) -> type[PriceFeed]:
    if not cls.key:
        raise ValueError(f"{cls.__name__} must define a non-empty `key`")
    _REGISTRY[cls.key] = cls
    return cls


def available() -> list[str]:
    _ensure_loaded()
    return sorted(_REGISTRY)


def create(settings: Settings) -> PriceFeed:
    _ensure_loaded()
    try:
        cls = _REGISTRY[settings.feed]
    except KeyError:
        raise KeyError(
            f"unknown feed {settings.feed!r}; available: {', '.join(sorted(_REGISTRY))}"
        ) from None
    return cls(settings)


_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if not _loaded:
        _loaded = True
        discover(sys.modules[__name__])


__all__ = ["PriceFeed", "register", "create", "available"]
