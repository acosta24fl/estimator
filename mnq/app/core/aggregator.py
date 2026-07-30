"""Fold 1-minute bars up into any registered timeframe.

Aggregation is a pure function of the 1-minute series, which is what keeps the
system easy to reason about: there is no incremental state to drift out of
sync, and a bad higher-timeframe bar can always be traced back to the minutes
that produced it.  Recomputation is cheap (a few tens of thousands of bars at
most), and :class:`AggregationCache` skips the work when nothing has changed.
"""

from __future__ import annotations

import time
from typing import Sequence

from ..models import Bar
from .timeframes import Timeframe


def aggregate(bars: Sequence[Bar], tf: Timeframe, now: float | None = None) -> list[Bar]:
    """Group ``bars`` (1-minute, ascending) into ``tf`` buckets.

    A bucket is marked complete once wall-clock time has passed its close *and*
    every source bar inside it is complete, so a partially-filled bucket is
    never presented as final.
    """
    if not bars:
        return []

    now = time.time() if now is None else now
    out: list[Bar] = []
    current: Bar | None = None
    current_end = 0
    all_sources_complete = True

    for src in bars:
        start = tf.bucketer.start(src.ts)
        if current is None or start != current.ts:
            if current is not None:
                current.complete = all_sources_complete and now >= current_end
                out.append(current)
            current = Bar(
                ts=start,
                open=src.open,
                high=src.high,
                low=src.low,
                close=src.close,
                volume=src.volume,
                count=src.count,
            )
            current_end = tf.bucketer.end(start)
            all_sources_complete = src.complete
        else:
            current.high = max(current.high, src.high)
            current.low = min(current.low, src.low)
            current.close = src.close
            current.volume += src.volume
            current.count += src.count
            all_sources_complete = all_sources_complete and src.complete

    if current is not None:
        current.complete = all_sources_complete and now >= current_end
        out.append(current)

    return out


class AggregationCache:
    """Memoises aggregation per timeframe, keyed on the 1-minute series state.

    The key includes the last bar's close so an update to the still-forming bar
    invalidates the cache, while an unchanged series serves instantly.
    """

    def __init__(self) -> None:
        self._cache: dict[str, tuple[tuple, list[Bar]]] = {}

    @staticmethod
    def _key(bars: Sequence[Bar]) -> tuple:
        if not bars:
            return (0,)
        last = bars[-1]
        return (len(bars), bars[0].ts, last.ts, last.close, last.high, last.low,
                last.volume, last.complete)

    def get(self, bars: Sequence[Bar], tf: Timeframe, now: float | None = None) -> list[Bar]:
        key = self._key(bars)
        hit = self._cache.get(tf.key)
        if hit is not None and hit[0] == key:
            return hit[1]
        result = aggregate(bars, tf, now=now)
        self._cache[tf.key] = (key, result)
        return result

    def clear(self) -> None:
        self._cache.clear()
