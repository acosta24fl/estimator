"""Live bar store.

TradingView pushes one alert per minute. This class accumulates those into 1m
bars and resamples them into the 5m/15m/4h frames the feature builder expects,
so the live path and the backtest path consume identically-shaped data.

Thread safety matters: the webhook writes from a request handler while the
scheduler reads from a background task. Every public method takes the lock.

Only *closed* bars are ever exposed. The in-progress minute is held back, since
a partially formed bar has a meaningless high, low and volume, and feeding it to
the model would produce a signal from data that has not happened yet.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .yahoo import OHLCV, resample_ohlcv

log = logging.getLogger(__name__)


class BarStore:
    """Append-only store of 1-minute bars with on-demand resampling."""

    def __init__(self, path: str | Path | None = None, max_bars: int = 200_000):
        self.path = Path(path) if path else None
        self.max_bars = max_bars
        self._lock = threading.RLock()
        self._bars: dict[datetime, dict[str, float]] = {}
        self._dirty = False
        if self.path and self.path.exists():
            self._load()

    # ------------------------------------------------------------- ingest

    def add_bar(
        self,
        timestamp: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0,
    ) -> datetime:
        """Insert or replace a complete 1m bar. Returns the normalised stamp."""
        ts = _floor_minute(timestamp)
        with self._lock:
            self._bars[ts] = {
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(volume),
            }
            self._dirty = True
            self._trim()
        return ts

    def add_price(
        self, timestamp: datetime, price: float, volume: float = 0.0
    ) -> datetime:
        """Fold a single price print into the current minute's bar.

        For TradingView alerts that only carry ``{{close}}``. The first print of
        a minute opens the bar; later prints extend its range.
        """
        ts = _floor_minute(timestamp)
        price = float(price)
        with self._lock:
            bar = self._bars.get(ts)
            if bar is None:
                self._bars[ts] = {
                    "open": price, "high": price, "low": price,
                    "close": price, "volume": float(volume),
                }
            else:
                bar["high"] = max(bar["high"], price)
                bar["low"] = min(bar["low"], price)
                bar["close"] = price
                bar["volume"] += float(volume)
            self._dirty = True
            self._trim()
        return ts

    def seed(self, minute_bars: pd.DataFrame) -> int:
        """Bulk-load history, e.g. Yahoo bars at startup.

        Without a seed the store needs ~17 hours of live alerts before the 4h
        EMA-200 is even defined, so the system would sit silent for days.
        """
        with self._lock:
            for ts, row in minute_bars.iterrows():
                self._bars[_floor_minute(ts)] = {
                    k: float(row[k]) for k in OHLCV
                }
            self._dirty = True
            self._trim()
            return len(self._bars)

    def _trim(self) -> None:
        if len(self._bars) <= self.max_bars:
            return
        for ts in sorted(self._bars)[: len(self._bars) - self.max_bars]:
            del self._bars[ts]

    # -------------------------------------------------------------- access

    def minute_frame(self, include_current: bool = False) -> pd.DataFrame:
        """All 1m bars, most recent incomplete minute excluded by default."""
        with self._lock:
            if not self._bars:
                return pd.DataFrame(columns=OHLCV)
            df = pd.DataFrame.from_dict(self._bars, orient="index")[OHLCV]
        df.index = pd.DatetimeIndex(df.index, name="timestamp")
        df = df.sort_index()
        if not include_current and len(df):
            current = _floor_minute(datetime.now(timezone.utc))
            df = df[df.index < current]
        return df

    def frames(self) -> dict[str, pd.DataFrame]:
        """The 5m/15m/4h views the feature builder consumes.

        Trailing partial buckets are dropped: a 4h bar stamped 12:00 is only
        complete at 16:00, and including it early would feed the model a
        truncated candle.
        """
        minutes = self.minute_frame()
        if minutes.empty:
            return {"5m": minutes, "15m": minutes, "4h": minutes}

        now = datetime.now(timezone.utc)
        out: dict[str, pd.DataFrame] = {}
        for key, rule, span in (("5m", "5min", 5), ("15m", "15min", 15), ("4h", "4h", 240)):
            res = resample_ohlcv(minutes, rule)
            if len(res):
                complete_before = now - timedelta(minutes=span)
                res = res[res.index <= complete_before]
            out[key] = res
        return out

    def last_price(self) -> float | None:
        with self._lock:
            if not self._bars:
                return None
            return self._bars[max(self._bars)]["close"]

    def last_bar_time(self) -> datetime | None:
        with self._lock:
            return max(self._bars) if self._bars else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._bars)

    # ---------------------------------------------------------- persistence

    def flush(self) -> None:
        """Write to disk so a restart does not lose the accumulated history."""
        if not self.path or not self._dirty:
            return
        with self._lock:
            df = pd.DataFrame.from_dict(self._bars, orient="index")[OHLCV].sort_index()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            # Atomic replace: a crash mid-write must not truncate the history.
            df.to_csv(tmp)
            tmp.replace(self.path)
            self._dirty = False
        log.debug("flushed %d bars -> %s", len(df), self.path)

    def _load(self) -> None:
        try:
            df = pd.read_csv(self.path, index_col=0, parse_dates=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read bar store %s: %s", self.path, exc)
            return
        idx = pd.to_datetime(df.index, utc=True)
        for ts, row in zip(idx, df.to_dict("records")):
            self._bars[_floor_minute(ts)] = {k: float(row[k]) for k in OHLCV}
        log.info("loaded %d bars from %s", len(self._bars), self.path)


def _floor_minute(ts: datetime | pd.Timestamp) -> datetime:
    """Normalise any timestamp to a tz-aware UTC minute boundary.

    Mixing naive and aware stamps silently produces duplicate bars for the same
    minute, so everything is coerced to UTC on the way in.
    """
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.floor("min").to_pydatetime()
