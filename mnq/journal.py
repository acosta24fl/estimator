"""The decision log: every action the system takes, written down.

Until now the system's reasoning lived in three places that could not be
queried after the fact - stdout (gone when the window closes), the paper trade
journal (closed trades only, which is the *end* of a decision) and model
artifacts (rebuilt on every train). Nothing recorded the far more common event:
the system looked, and decided not to act.

That silence is the expensive one. "Why did it not take that move?" is the
question that improves a system, and it can only be answered from a record of
the rejections.

So every observable step writes one line here:

* **poll**    - market data fetched, how many bars were new, or why it failed
* **score**   - the model's probabilities for the current bar
* **gate**    - a setup considered and accepted or rejected, with the reason
* **signal**  - a paper position opened, with its levels
* **exit**    - a position closed, with the outcome
* **retrain** - a refit started or finished
* **system**  - start, stop, mode changes

Format is JSON Lines: one self-describing object per line, appended, never
rewritten. It is greppable with the eye, loadable with one pandas call, and a
truncated final line (from a kill during a write) costs one record rather than
the file.

Every record carries a ``summary`` written in plain English. The dashboard
shows that string directly, so the activity feed is readable by someone who
does not know what ``p_long`` means.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

#: Rotate once the file passes this. A minute-cadence poll writes roughly
#: 100 KB/day, so this holds about two months before the first rotation.
MAX_BYTES = 8 * 1024 * 1024

#: How many bytes to read from the end when tailing. Comfortably more than the
#: few hundred records the dashboard asks for.
TAIL_BYTES = 512 * 1024


class DecisionLog:
    """Append-only JSONL record of what the system did and why.

    Writing must never break the caller. A trading loop that dies because a
    disk is full has turned an observability feature into an outage, so every
    public method swallows I/O errors and logs them instead.
    """

    def __init__(self, path: str | Path, max_bytes: int = MAX_BYTES):
        self.path = Path(path)
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._count = 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning("decision log directory unavailable: %s", exc)

    # ------------------------------------------------------------- writing

    def record(self, kind: str, summary: str, **fields: Any) -> dict[str, Any]:
        """Append one event. Returns the record (useful in tests)."""
        entry = {
            "t": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kind": kind,
            "summary": summary,
            **{k: _plain(v) for k, v in fields.items()},
        }
        line = json.dumps(entry, default=str)
        with self._lock:
            try:
                self._rotate_if_needed()
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                self._count += 1
            except OSError as exc:  # noqa: PERF203 - logging must not raise
                log.warning("could not write to the decision log: %s", exc)
        return entry

    def _rotate_if_needed(self) -> None:
        try:
            if self.path.exists() and self.path.stat().st_size > self.max_bytes:
                previous = self.path.with_suffix(self.path.suffix + ".1")
                os.replace(self.path, previous)
                log.info("decision log rotated to %s", previous.name)
        except OSError as exc:
            log.warning("could not rotate the decision log: %s", exc)

    # ------------------------------------------------------------- reading

    def tail(self, n: int = 60, kinds: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """The most recent ``n`` records, newest first.

        Reads only the tail of the file: the dashboard polls this every few
        seconds and the file grows without bound.
        """
        wanted = set(kinds) if kinds else None
        try:
            size = self.path.stat().st_size
        except OSError:
            return []

        try:
            with self.path.open("rb") as fh:
                if size > TAIL_BYTES:
                    fh.seek(size - TAIL_BYTES)
                    fh.readline()          # discard a partial first line
                raw = fh.read().decode("utf-8", errors="replace")
        except OSError as exc:
            log.warning("could not read the decision log: %s", exc)
            return []

        out: list[dict[str, Any]] = []
        for line in reversed(raw.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                # A record truncated by a kill mid-write. Skip it; the rest of
                # the file is still valid.
                continue
            if wanted and entry.get("kind") not in wanted:
                continue
            out.append(entry)
            if len(out) >= n:
                break
        return out

    def counts(self) -> dict[str, int]:
        """How many of each kind are in the readable tail, plus the file size."""
        tally: dict[str, int] = {}
        for entry in self.tail(n=100_000):
            kind = str(entry.get("kind", "?"))
            tally[kind] = tally.get(kind, 0) + 1
        return tally

    def stats(self) -> dict[str, Any]:
        """Summary for the dashboard: size, totals, and the newest event."""
        newest = self.tail(1)
        try:
            size = self.path.stat().st_size if self.path.exists() else 0
        except OSError:
            size = 0
        return {
            "path": str(self.path),
            "bytes": int(size),
            "written_this_session": self._count,
            "counts": self.counts(),
            "latest": newest[0] if newest else None,
        }

    def load(self):
        """The whole log as a DataFrame, for offline analysis."""
        import pandas as pd

        if not self.path.exists():
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        frame = pd.DataFrame(rows)
        if not frame.empty and "t" in frame:
            frame["t"] = pd.to_datetime(frame["t"], utc=True, errors="coerce")
            frame = frame.set_index("t").sort_index()
        return frame


def _plain(value: Any) -> Any:
    """Coerce numpy/pandas scalars so ``json.dumps`` cannot emit bare NaN.

    A NaN written into JSON parses in Python and fails in every browser, which
    would take the dashboard down for a value nobody was reading.
    """
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, (np.floating, float)):
            v = float(value)
            return None if v != v or v in (float("inf"), float("-inf")) else round(v, 6)
        if isinstance(value, np.bool_):
            return bool(value)
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


class NullLog(DecisionLog):
    """A log that discards everything.

    Lets every caller write ``self.journal.record(...)`` unconditionally rather
    than guarding each call site with ``if self.journal is not None``. The guard
    is the thing that eventually gets forgotten at one call site.
    """

    def __init__(self):  # noqa: D107 - deliberately does not touch the disk
        self.path = Path(os.devnull)
        self.max_bytes = MAX_BYTES
        self._lock = threading.Lock()
        self._count = 0

    def record(self, kind: str, summary: str, **fields: Any) -> dict[str, Any]:
        return {"t": "", "kind": kind, "summary": summary, **fields}

    def tail(self, n: int = 60, kinds: Iterable[str] | None = None) -> list[dict[str, Any]]:
        return []

    def counts(self) -> dict[str, int]:
        return {}
