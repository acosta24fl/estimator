"""Plain data structures shared by every layer.

These are deliberately dependency-free so feeds, aggregation, indicators and
the API all speak the same vocabulary without importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Bar:
    """One OHLCV candle.

    ``ts`` is the bar's *start* time as a UTC epoch second, which is also the
    key used everywhere (store, aggregation, chart).  ``count`` records how
    many 1-minute source bars were folded in, which makes aggregation
    auditable.  ``complete`` is False while the bar is still forming.
    """

    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    count: int = 1
    complete: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "count": self.count,
            "complete": self.complete,
        }

    def as_chart_dict(self) -> dict[str, Any]:
        """Shape expected by lightweight-charts (``time`` in epoch seconds)."""
        return {
            "time": self.ts,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "complete": self.complete,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Bar":
        return cls(
            ts=int(raw["ts"]),
            open=float(raw["open"]),
            high=float(raw["high"]),
            low=float(raw["low"]),
            close=float(raw["close"]),
            volume=float(raw.get("volume", 0.0)),
            count=int(raw.get("count", 1)),
            complete=bool(raw.get("complete", False)),
        )

    def copy(self) -> "Bar":
        return Bar(
            ts=self.ts,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            count=self.count,
            complete=self.complete,
        )

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def typical(self) -> float:
        return (self.high + self.low + self.close) / 3.0


@dataclass
class Quote:
    """Latest tradable price plus the day context the feed hands us."""

    price: float
    ts: int
    prev_close: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    currency: str = "USD"
    exchange: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "price": self.price,
            "ts": self.ts,
            "prev_close": self.prev_close,
            "day_high": self.day_high,
            "day_low": self.day_low,
            "currency": self.currency,
            "exchange": self.exchange,
        }


@dataclass
class FeedSnapshot:
    """What a single feed poll returns."""

    bars: list[Bar] = field(default_factory=list)
    quote: Quote | None = None
