"""Timeframe registry — the single source of truth for every bar interval.

To add a timeframe (e.g. 2 hours) you append one ``register`` call at the
bottom of this file.  Nothing else changes: the aggregator, the API payload and
the UI's timeframe switcher are all driven from this registry.

Bucketing is pluggable so that calendar-aware intervals (the futures trade day)
can live beside simple fixed-width ones without special-casing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol
from zoneinfo import ZoneInfo


class Bucketer(Protocol):
    """Maps a timestamp onto the bar-bucket that contains it."""

    def start(self, ts: int) -> int:
        """Epoch second of the bucket start containing ``ts``."""

    def end(self, start: int) -> int:
        """Epoch second at which the bucket beginning at ``start`` closes."""


@dataclass(frozen=True)
class EpochBucket:
    """Fixed-width buckets aligned to ``anchor`` (default: the UTC epoch).

    All sub-daily intervals here divide evenly into an hour or a day, so UTC
    alignment produces the same boundaries traders expect (:00, :05, :15, ...).
    """

    seconds: int
    anchor: int = 0

    def start(self, ts: int) -> int:
        return ((int(ts) - self.anchor) // self.seconds) * self.seconds + self.anchor

    def end(self, start: int) -> int:
        return start + self.seconds


@dataclass(frozen=True)
class SessionBucket:
    """One bucket per futures trade date.

    The CME trade date rolls at 18:00 America/New_York, so the bar labelled
    "Tuesday" spans Monday 18:00 ET through Tuesday 17:00 ET.  Using zoneinfo
    keeps this correct across daylight-saving transitions.
    """

    tz: str = "America/New_York"
    open_hour: int = 18

    def _zone(self) -> ZoneInfo:
        return ZoneInfo(self.tz)

    def start(self, ts: int) -> int:
        zone = self._zone()
        local = datetime.fromtimestamp(int(ts), tz=zone)
        open_today = local.replace(
            hour=self.open_hour, minute=0, second=0, microsecond=0
        )
        if local < open_today:
            open_today -= timedelta(days=1)
            # Re-resolve against the zone so a DST shift on the previous day
            # does not skew the boundary.
            open_today = datetime(
                open_today.year,
                open_today.month,
                open_today.day,
                self.open_hour,
                tzinfo=zone,
            )
        return int(open_today.astimezone(timezone.utc).timestamp())

    def end(self, start: int) -> int:
        zone = self._zone()
        local = datetime.fromtimestamp(int(start), tz=zone)
        nxt = local + timedelta(days=1)
        nxt = datetime(nxt.year, nxt.month, nxt.day, self.open_hour, tzinfo=zone)
        return int(nxt.astimezone(timezone.utc).timestamp())


@dataclass(frozen=True)
class Timeframe:
    key: str
    label: str
    bucketer: Bucketer
    nominal_seconds: int
    default_limit: int = 400
    order: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "seconds": self.nominal_seconds,
            "default_limit": self.default_limit,
            "order": self.order,
        }


TIMEFRAMES: dict[str, Timeframe] = {}

#: Every higher timeframe is folded up from this one.
BASE_TIMEFRAME_KEY = "1m"

# The session definition used by the daily bucket.  Overridden at startup from
# Settings so the whole app agrees on where the trade day begins.
_SESSION = SessionBucket()


def register(tf: Timeframe) -> Timeframe:
    if tf.key in TIMEFRAMES:
        raise ValueError(f"timeframe {tf.key!r} already registered")
    TIMEFRAMES[tf.key] = tf
    return tf


def get(key: str) -> Timeframe:
    try:
        return TIMEFRAMES[key]
    except KeyError:
        raise KeyError(f"unknown timeframe {key!r}") from None


def ordered() -> list[Timeframe]:
    return sorted(TIMEFRAMES.values(), key=lambda t: (t.order, t.nominal_seconds))


def base() -> Timeframe:
    return TIMEFRAMES[BASE_TIMEFRAME_KEY]


def configure_session(tz: str, open_hour: int) -> None:
    """Point the daily bucket at the configured session, in place."""
    global _SESSION
    _SESSION = SessionBucket(tz=tz, open_hour=open_hour)
    daily = TIMEFRAMES.get("1d")
    if daily is not None:
        TIMEFRAMES["1d"] = Timeframe(
            key=daily.key,
            label=daily.label,
            bucketer=_SESSION,
            nominal_seconds=daily.nominal_seconds,
            default_limit=daily.default_limit,
            order=daily.order,
        )


def session_bucket() -> SessionBucket:
    """The active session definition (used by VWAP and daily structure)."""
    return _SESSION


MINUTE = 60
HOUR = 60 * MINUTE

# ---------------------------------------------------------------------------
# Registered timeframes.  Append here to add more.
# ---------------------------------------------------------------------------
register(Timeframe("1m", "1 Minute", EpochBucket(1 * MINUTE), 1 * MINUTE, 480, 10))
register(Timeframe("5m", "5 Minutes", EpochBucket(5 * MINUTE), 5 * MINUTE, 480, 20))
register(Timeframe("10m", "10 Minutes", EpochBucket(10 * MINUTE), 10 * MINUTE, 400, 30))
register(Timeframe("15m", "15 Minutes", EpochBucket(15 * MINUTE), 15 * MINUTE, 400, 40))
register(Timeframe("30m", "30 Minutes", EpochBucket(30 * MINUTE), 30 * MINUTE, 400, 50))
register(Timeframe("1h", "1 Hour", EpochBucket(1 * HOUR), 1 * HOUR, 400, 60))
register(Timeframe("4h", "4 Hours", EpochBucket(4 * HOUR), 4 * HOUR, 300, 70))
register(Timeframe("1d", "1 Day", _SESSION, 24 * HOUR, 250, 80))
