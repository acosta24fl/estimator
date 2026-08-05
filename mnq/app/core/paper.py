"""Live paper trading off the dashboard's directional call.

**Simulated only.** Nothing here connects to a broker or places an order; it
records what would have happened if the call had been traded.

The rule is deliberately the same one ``app.tradetest`` replays, so live results
and backtested results are comparable:

* when a new signal-horizon bar opens and the call is bullish or bearish, enter
  at that bar's open,
* exit at that bar's close — a fixed hold, no stop, no target,
* charge the same round-turn cost.

Every trade is appended to ``data/trades.jsonl`` as it opens and again as it
closes, following the same append-only, last-record-wins convention as the bar
and prediction logs.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..models import Bar
from .trading import DEFAULT_COST_POINTS, DOLLARS_PER_POINT, TradeStats, summarise

log = logging.getLogger(__name__)

LONG = 1
SHORT = -1


@dataclass
class PaperTrade:
    trade_id: str  # the bar timestamp it was opened on, as a string
    direction: int
    opened_ts: int
    entry: float
    cost_points: float
    horizon_seconds: int
    confidence: str = ""
    reason: str = ""
    closed_ts: int | None = None
    exit: float | None = None
    gross_points: float | None = None
    net_points: float | None = None

    @property
    def is_open(self) -> bool:
        return self.closed_ts is None

    @property
    def side(self) -> str:
        return "long" if self.direction == LONG else "short"

    def unrealised(self, price: float) -> float:
        """Mark-to-market in points, costs included."""
        return self.direction * (price - self.entry) - self.cost_points

    def close(self, price: float, ts: int) -> None:
        self.exit = price
        self.closed_ts = ts
        self.gross_points = self.direction * (price - self.entry)
        self.net_points = self.gross_points - self.cost_points

    def as_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "direction": self.direction,
            "side": self.side,
            "opened_ts": self.opened_ts,
            "entry": round(self.entry, 2),
            "cost_points": self.cost_points,
            "horizon_seconds": self.horizon_seconds,
            "confidence": self.confidence,
            "reason": self.reason,
            "closed_ts": self.closed_ts,
            "exit": None if self.exit is None else round(self.exit, 2),
            "gross_points": None if self.gross_points is None else round(self.gross_points, 2),
            "net_points": None if self.net_points is None else round(self.net_points, 2),
            "net_dollars": (
                None if self.net_points is None
                else round(self.net_points * DOLLARS_PER_POINT, 2)
            ),
            "open": self.is_open,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PaperTrade":
        trade = cls(
            trade_id=str(raw["trade_id"]),
            direction=int(raw["direction"]),
            opened_ts=int(raw["opened_ts"]),
            entry=float(raw["entry"]),
            cost_points=float(raw.get("cost_points", DEFAULT_COST_POINTS)),
            horizon_seconds=int(raw.get("horizon_seconds", 600)),
            confidence=str(raw.get("confidence", "")),
            reason=str(raw.get("reason", "")),
        )
        if raw.get("closed_ts") is not None:
            trade.closed_ts = int(raw["closed_ts"])
            trade.exit = float(raw["exit"])
            trade.gross_points = float(raw["gross_points"])
            trade.net_points = float(raw["net_points"])
        return trade


class PaperTrader:
    """Opens and closes simulated positions from the directional call."""

    def __init__(
        self,
        path: Path,
        cost_points: float = DEFAULT_COST_POINTS,
        max_entry_age: float = 90.0,
    ) -> None:
        self._path = Path(path)
        self.cost_points = cost_points
        #: Refuse to enter on a bar that opened long ago. On startup the current
        #: bar may already be well advanced, and filling at its open would be a
        #: price nobody could have got.
        self.max_entry_age = max_entry_age
        self._lock = threading.RLock()
        self._trades: dict[str, PaperTrade] = {}

    # -- persistence -------------------------------------------------------

    def load(self) -> int:
        if not self._path.exists():
            return 0
        loaded = 0
        with self._path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    trade = PaperTrade.from_dict(json.loads(line))
                except (ValueError, KeyError) as exc:
                    log.warning("skipping bad trade %s:%d (%s)", self._path, line_no, exc)
                    continue
                self._trades[trade.trade_id] = trade  # last record wins
                loaded += 1
        return loaded

    def _append(self, trade: PaperTrade) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(trade.as_dict(), separators=(",", ":")) + "\n")

    # -- state -------------------------------------------------------------

    @property
    def open_trade(self) -> PaperTrade | None:
        for trade in self._trades.values():
            if trade.is_open:
                return trade
        return None

    def trades(self) -> list[PaperTrade]:
        return sorted(self._trades.values(), key=lambda t: t.opened_ts)

    def closed(self) -> list[PaperTrade]:
        return [t for t in self.trades() if not t.is_open]

    # -- the rule ----------------------------------------------------------

    def update(self, bars: Sequence[Bar], outlook, now: float) -> list[PaperTrade]:
        """Advance the simulation one poll. Returns trades that changed."""
        changed: list[PaperTrade] = []
        if not bars:
            return changed

        by_ts = {b.ts: b for b in bars}
        with self._lock:
            # Close first: a position is held for exactly one bar, so once the
            # bar it opened on has completed the exit price is that bar's close.
            position = self.open_trade
            if position is not None:
                bar = by_ts.get(position.opened_ts)
                if bar is not None and bar.complete:
                    position.close(bar.close, position.opened_ts + position.horizon_seconds)
                    self._append(position)
                    changed.append(position)
                    position = None

            if position is not None:
                return changed  # still holding

            forming = bars[-1]
            if forming.complete:
                return changed  # no live bar to enter on
            if str(forming.ts) in self._trades:
                return changed  # already traded this bar
            if outlook is None or outlook.direction not in ("bullish", "bearish"):
                return changed
            if (now - forming.ts) > self.max_entry_age:
                return changed  # bar already too far along to fill at its open

            trade = PaperTrade(
                trade_id=str(forming.ts),
                direction=LONG if outlook.direction == "bullish" else SHORT,
                opened_ts=forming.ts,
                entry=forming.open,
                cost_points=self.cost_points,
                horizon_seconds=outlook.horizon_minutes * 60,
                confidence=outlook.confidence,
                reason=outlook.reason,
            )
            self._trades[trade.trade_id] = trade
            self._append(trade)
            changed.append(trade)
        return changed

    # -- reporting ---------------------------------------------------------

    def stats(self) -> TradeStats:
        return summarise([t.net_points or 0.0 for t in self.closed()], self.cost_points)

    def summary(self, price: float | None = None) -> dict[str, Any]:
        stats = self.stats()
        payload = stats.as_dict()
        position = self.open_trade
        payload.update(
            {
                "cost_points": self.cost_points,
                "open_position": None if position is None else position.as_dict(),
                "open_unrealised": (
                    None if (position is None or price is None)
                    else round(position.unrealised(price), 2)
                ),
            }
        )
        return payload
