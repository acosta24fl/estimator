"""Bar storage: in-memory series plus an append-only minute log on disk.

Design notes
------------
* The 1-minute series is the only thing persisted.  Everything else is derived,
  so there is exactly one file format to care about.
* The log is **append-only JSONL, one file per UTC day**.  Nothing is ever
  rewritten in place; a corrected bar is appended again and the loader takes
  the last record for a timestamp.  That makes the log safe to tail, easy to
  diff, and impossible to corrupt with a partial write.
* A bar is written the moment it closes, which is the "log bars every minute"
  behaviour the dashboard reports in its status strip.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from ..models import Bar

log = logging.getLogger(__name__)

_ONE_MINUTE = 60


class BarStore:
    """Holds the 1-minute series and the feed's daily series."""

    def __init__(self, bars_dir: Path, max_1m_bars: int = 20_000) -> None:
        self._dir = Path(bars_dir)
        self._minute_dir = self._dir / "1m"
        self._daily_path = self._dir / "1d.jsonl"
        self._max = max_1m_bars
        self._lock = threading.RLock()

        self._minute: dict[int, Bar] = {}
        self._daily: dict[int, Bar] = {}
        self._logged: dict[int, tuple] = {}  # ts -> fingerprint already on disk
        self.last_logged_ts: int | None = None
        self.logged_count = 0

    # -- lifecycle ---------------------------------------------------------

    def ensure_dirs(self) -> None:
        self._minute_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> int:
        """Replay every JSONL file back into memory.  Last record wins."""
        self.ensure_dirs()
        loaded = 0
        for path in sorted(self._minute_dir.glob("*.jsonl")):
            for bar in self._read_jsonl(path):
                self._minute[bar.ts] = bar
                self._logged[bar.ts] = self._fingerprint(bar)
                loaded += 1
        for bar in self._read_jsonl(self._daily_path):
            self._daily[bar.ts] = bar
        self._trim()
        if self._minute:
            self.last_logged_ts = max(self._minute)
        log.info("loaded %d minute records from %s", loaded, self._minute_dir)
        return loaded

    @staticmethod
    def _read_jsonl(path: Path) -> Iterable[Bar]:
        if not path.exists():
            return []
        out: list[Bar] = []
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(Bar.from_dict(json.loads(line)))
                except (ValueError, KeyError) as exc:
                    log.warning("skipping bad record %s:%d (%s)", path, line_no, exc)
        return out

    # -- writes ------------------------------------------------------------

    def upsert_minute(self, bars: Iterable[Bar]) -> list[Bar]:
        """Merge feed bars into the series; return bars newly written to disk.

        Only *complete* bars are logged.  If a complete bar is later revised by
        the feed, the new version is appended and supersedes the old one.
        """
        newly_logged: list[Bar] = []
        with self._lock:
            for incoming in bars:
                incoming = incoming.copy()
                incoming.ts = (incoming.ts // _ONE_MINUTE) * _ONE_MINUTE
                self._minute[incoming.ts] = incoming
                if not incoming.complete:
                    continue
                fingerprint = self._fingerprint(incoming)
                if self._logged.get(incoming.ts) == fingerprint:
                    continue
                self._append(self._minute_path(incoming.ts), incoming)
                self._logged[incoming.ts] = fingerprint
                self.logged_count += 1
                self.last_logged_ts = max(self.last_logged_ts or 0, incoming.ts)
                newly_logged.append(incoming)
            self._trim()
        return newly_logged

    def upsert_daily(self, bars: Iterable[Bar]) -> list[Bar]:
        """Same append-only contract as the minute log, for daily bars."""
        changed: list[Bar] = []
        with self._lock:
            for incoming in bars:
                existing = self._daily.get(incoming.ts)
                if existing is not None and self._fingerprint(existing) == self._fingerprint(incoming):
                    continue
                self._daily[incoming.ts] = incoming.copy()
                changed.append(self._daily[incoming.ts])
            for bar in changed:
                self._append(self._daily_path, bar)
        return changed

    def _append(self, path: Path, bar: Bar) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(bar.as_dict(), separators=(",", ":")) + "\n")

    def _minute_path(self, ts: int) -> Path:
        day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        return self._minute_dir / f"{day}.jsonl"

    @staticmethod
    def _fingerprint(bar: Bar) -> tuple:
        return (bar.open, bar.high, bar.low, bar.close, bar.volume)

    def _trim(self) -> None:
        if len(self._minute) <= self._max:
            return
        for ts in sorted(self._minute)[: len(self._minute) - self._max]:
            self._minute.pop(ts, None)
            self._logged.pop(ts, None)

    # -- reads -------------------------------------------------------------

    def minute_series(self) -> list[Bar]:
        with self._lock:
            return [self._minute[ts] for ts in sorted(self._minute)]

    def daily_series(self) -> list[Bar]:
        with self._lock:
            return [self._daily[ts] for ts in sorted(self._daily)]

    @property
    def minute_count(self) -> int:
        return len(self._minute)

    @property
    def daily_count(self) -> int:
        return len(self._daily)
