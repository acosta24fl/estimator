"""Price-feed interface.

A feed's only job is to return normalised :class:`Bar` objects.  Everything
downstream (logging, aggregation, indicators) is feed-agnostic, so swapping or
adding a data source touches nothing outside this package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import Settings
from ..models import FeedSnapshot


class PriceFeed(ABC):
    #: Value accepted by ``MNQ_FEED`` to select this implementation.
    key: str = ""
    #: Human-readable name shown in the dashboard status strip.
    name: str = ""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    async def fetch_intraday(self) -> FeedSnapshot:
        """Return recent 1-minute bars, newest last, plus the latest quote."""

    @abstractmethod
    async def fetch_daily(self) -> FeedSnapshot:
        """Return daily bars, newest last."""

    async def aclose(self) -> None:
        """Release any network resources."""
