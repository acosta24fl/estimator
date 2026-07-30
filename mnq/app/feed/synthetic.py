"""Deterministic offline feed — no network required.

Run with ``MNQ_FEED=synthetic`` to develop the UI, exercise aggregation and
indicators, or demo the dashboard when the market is closed or Yahoo is
unreachable.  The walk is seeded off each bar's own timestamp, so the series is
reproducible across restarts and simply extends as wall-clock time advances.
"""

from __future__ import annotations

import math
import random
import re
import time

from ..models import Bar, FeedSnapshot, Quote
from . import register
from .base import PriceFeed

_RANGE_RE = re.compile(r"^(\d+)\s*([mhdwy])$", re.IGNORECASE)
_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 604800, "y": 31_536_000}


def _range_seconds(spec: str, default: int) -> int:
    match = _RANGE_RE.match(spec.strip())
    if not match:
        return default
    return int(match.group(1)) * _UNIT_SECONDS[match.group(2).lower()]


@register
class SyntheticFeed(PriceFeed):
    key = "synthetic"
    name = "Synthetic (offline)"

    base_price = 20_000.0
    seed = 20_240_101

    async def fetch_intraday(self) -> FeedSnapshot:
        span = min(_range_seconds(self.settings.intraday_range, 5 * 86400), 7 * 86400)
        bars = self._walk(step=60, span=span, vol=0.00035, volume_base=180)
        quote = (
            Quote(price=bars[-1].close, ts=int(time.time()), prev_close=bars[0].open)
            if bars
            else None
        )
        return FeedSnapshot(bars=bars, quote=quote)

    async def fetch_daily(self) -> FeedSnapshot:
        span = _range_seconds(self.settings.daily_range, 365 * 86400)
        bars = self._walk(step=86400, span=span, vol=0.011, volume_base=90_000)
        return FeedSnapshot(bars=bars, quote=None)

    # -- generation --------------------------------------------------------

    # Superposed cycles: (period in seconds, relative amplitude, phase).
    _WAVES = (
        (30 * 86400, 0.060, 0.0),
        (3 * 86400, 0.020, 1.1),
        (6 * 3600, 0.006, 2.3),
        (90 * 60, 0.002, 0.7),
    )

    def _level(self, ts: int) -> float:
        """Price path as a pure function of time.

        Deriving the level from ``ts`` alone — rather than accumulating a walk
        from the start of the window — means a given minute always produces the
        same bar.  Closed bars therefore never get revised as the window slides,
        which is what the real feed does and what the append-only log expects.
        """
        level = 1.0
        for period, amplitude, phase in self._WAVES:
            level += amplitude * math.sin(2 * math.pi * ts / period + phase)
        jitter = random.Random(ts ^ self.seed).gauss(0.0, 0.0004)
        return self.base_price * (level + jitter)

    def _walk(self, step: int, span: int, vol: float, volume_base: int) -> list[Bar]:
        now = time.time()
        last_start = int(now // step) * step
        first_start = last_start - span
        bars: list[Bar] = []

        for ts in range(first_start, last_start + step, step):
            rng = random.Random(ts ^ self.seed ^ step)
            # Sharing the level function across adjacent bars keeps the series
            # continuous: this bar's close is the next bar's open.
            open_ = self._level(ts)
            close = self._level(ts + step)
            wick = abs(close - open_) * rng.uniform(0.3, 1.6) + open_ * vol * 0.3
            high = max(open_, close) + wick * rng.uniform(0.0, 1.0)
            low = min(open_, close) - wick * rng.uniform(0.0, 1.0)
            # MNQ trades in 0.25 increments.
            open_, high, low, close = (_tick(x) for x in (open_, high, low, close))
            high = max(high, open_, close)
            low = min(low, open_, close)

            session_weight = 1.0 + 0.8 * math.sin((ts % 86400) / 86400 * 2 * math.pi)
            bars.append(
                Bar(
                    ts=ts,
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=round(volume_base * session_weight * rng.uniform(0.4, 1.9)),
                    count=1,
                    complete=(ts + step) <= now,
                )
            )

        return bars


def _tick(price: float) -> float:
    return round(price * 4.0) / 4.0
