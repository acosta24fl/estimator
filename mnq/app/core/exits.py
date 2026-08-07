"""Bracket exits: hold until a stop or a target is hit.

The paper trader holds for exactly one bar and exits at its close, which needs
no assumption about the path price took inside the bar. A stop and a target do:
from a bar's OHLC alone you cannot tell whether the high or the low came first,
and guessing decides the result.

This system stores 1-minute bars as the single source of truth for every
timeframe, so the path is available — a 10-minute bar is ten ordered minute
bars — and the walk below resolves stop and target in real order against them.

Three conservatisms, each of which costs the simulation money rather than
granting it any:

* when a stop and a target both lie inside the *same* minute's range, the
  **stop** is taken; 1-minute granularity cannot say which came first, and the
  pessimistic reading is the one that will not flatter a strategy,
* a minute that **opens beyond** a level fills at that open, not at the level —
  price gapped past the order, which is what actually happens and is strictly
  worse than the level for a stop,
* a position still open at the horizon is closed at the market, counted as a
  timeout, and charged the full cost.

The arithmetic to keep in mind
------------------------------
For a driftless price, a bracket has **exactly zero expectancy before costs**,
whatever the stop and target are. By optional stopping, the probability of
reaching the target first is ``stop / (stop + target)``, so

    E = target * stop/(stop+target) - stop * target/(stop+target) = 0

A 5-point stop with a 25-point target hits its target about 1 time in 6 and
pays 5:1, which is break-even by construction — and after costs, a loss. A
bracket cannot manufacture an edge; it can only reshape the distribution of an
edge that is already there. :func:`break_even_win_rate` prints the bar any real
result has to clear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..models import Bar
from .trading import DEFAULT_COST_POINTS, DOLLARS_PER_POINT, TradeStats, summarise

LONG = 1
SHORT = -1

STOP = "stop"
TARGET = "target"
TIMEOUT = "timeout"


@dataclass
class BracketTrade:
    """One trade held to a stop, a target, or the horizon."""

    ts: int
    direction: int
    entry: float
    stop_points: float
    target_points: float
    cost_points: float
    exit: float = 0.0
    exit_ts: int = 0
    exit_reason: str = TIMEOUT
    minutes_held: int = 0
    #: Best and worst the position ever got, in points, before costs.
    mfe: float = 0.0
    mae: float = 0.0

    @property
    def gross_points(self) -> float:
        return self.direction * (self.exit - self.entry)

    @property
    def net_points(self) -> float:
        return self.gross_points - self.cost_points

    @property
    def side(self) -> str:
        return "long" if self.direction == LONG else "short"

    def as_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "side": self.side,
            "entry": round(self.entry, 2),
            "exit": round(self.exit, 2),
            "exit_ts": self.exit_ts,
            "exit_reason": self.exit_reason,
            "minutes_held": self.minutes_held,
            "mfe": round(self.mfe, 2),
            "mae": round(self.mae, 2),
            "gross_points": round(self.gross_points, 2),
            "net_points": round(self.net_points, 2),
            "net_dollars": round(self.net_points * DOLLARS_PER_POINT, 2),
        }


def break_even_win_rate(stop_points: float, target_points: float,
                        cost_points: float = 0.0) -> float | None:
    """Win rate a stop/target pair needs just to break even.

    ``stop / (stop + target)`` once costs are folded into both legs. This is the
    number to compare a measured win rate against — a 17% win rate on a 1:5
    bracket is not a bad strategy, it is exactly a coin flip.
    """
    win = target_points - cost_points
    loss = stop_points + cost_points
    if win <= 0:
        return None
    return loss / (win + loss)


def theoretical_win_rate(stop_points: float, target_points: float) -> float | None:
    """P(target before stop) for a driftless price, by optional stopping."""
    total = stop_points + target_points
    return (stop_points / total) if total > 0 else None


def resolve(
    minutes: Sequence[Bar],
    direction: int,
    entry: float,
    stop_points: float,
    target_points: float,
    cost_points: float = DEFAULT_COST_POINTS,
    ts: int = 0,
) -> BracketTrade:
    """Walk minute bars from the entry until a level is hit or they run out.

    ``minutes`` must start at or after the entry minute and be in order. An
    empty sequence returns a trade that never moved, which keeps callers from
    having to special-case a gap in the data.
    """
    trade = BracketTrade(
        ts=ts or (minutes[0].ts if minutes else 0),
        direction=direction,
        entry=entry,
        stop_points=stop_points,
        target_points=target_points,
        cost_points=cost_points,
    )
    if not minutes:
        trade.exit = entry
        return trade

    stop = entry - direction * stop_points if stop_points > 0 else None
    target = entry + direction * target_points if target_points > 0 else None

    for index, minute in enumerate(minutes):
        trade.minutes_held = index + 1
        trade.exit_ts = minute.ts
        # Track how far the position ever ran, both ways, for MFE/MAE.
        best = direction * ((minute.high if direction == LONG else minute.low) - entry)
        worst = direction * ((minute.low if direction == LONG else minute.high) - entry)
        trade.mfe = max(trade.mfe, best)
        trade.mae = min(trade.mae, worst)

        # Stop first, deliberately — see the module docstring.
        if stop is not None and _through(minute, direction, stop, adverse=True):
            # A minute that opened past the stop fills at that open, which is
            # worse than the stop. Not a detail: a 5-point stop is 20 ticks, and
            # gapping through it is common in a fast move.
            gapped = (minute.open <= stop) if direction == LONG else (minute.open >= stop)
            trade.exit = minute.open if gapped else stop
            trade.exit_reason = STOP
            return trade

        if target is not None and _through(minute, direction, target, adverse=False):
            gapped = (
                (minute.open >= target) if direction == LONG else (minute.open <= target)
            )
            trade.exit = minute.open if gapped else target
            trade.exit_reason = TARGET
            return trade

    trade.exit = minutes[-1].close
    trade.exit_reason = TIMEOUT
    return trade


def _through(minute: Bar, direction: int, level: float, adverse: bool) -> bool:
    against = (direction == LONG) == adverse
    return minute.low <= level if against else minute.high >= level


# ---------------------------------------------------------------------------
# Running a whole signal series
# ---------------------------------------------------------------------------


@dataclass
class BracketRun:
    trades: list[BracketTrade] = field(default_factory=list)
    stats: TradeStats | None = None
    skipped_overlap: int = 0

    def by_reason(self) -> dict[str, int]:
        out = {STOP: 0, TARGET: 0, TIMEOUT: 0}
        for t in self.trades:
            out[t.exit_reason] = out.get(t.exit_reason, 0) + 1
        return out

    def mean(self, attr: str) -> float:
        if not self.trades:
            return 0.0
        return sum(getattr(t, attr) for t in self.trades) / len(self.trades)


def run(
    signals: Sequence[tuple[int, int, float]],
    minutes: Sequence[Bar],
    stop_points: float,
    target_points: float,
    cost_points: float = DEFAULT_COST_POINTS,
    max_hold_minutes: int = 120,
    allow_overlap: bool = False,
) -> BracketRun:
    """Trade every signal with the same bracket.

    ``signals`` is ``(entry_ts, direction, entry_price)`` in time order.

    By default a new signal arriving while a position is still open is
    **skipped**, because that is what the live trader does — one position at a
    time. Counting them all would quietly assume unlimited capital and would
    also double-count the same market move across overlapping trades, which
    inflates both the win rate and the drawdown.
    """
    out = BracketRun()
    if not minutes:
        return out

    times = [m.ts for m in minutes]
    busy_until = -1

    for entry_ts, direction, entry_price in signals:
        if not allow_overlap and entry_ts < busy_until:
            out.skipped_overlap += 1
            continue
        start = _index_at(times, entry_ts)
        if start is None:
            continue
        window = minutes[start : start + max_hold_minutes]
        trade = resolve(
            window, direction, entry_price, stop_points, target_points,
            cost_points, ts=entry_ts,
        )
        out.trades.append(trade)
        busy_until = trade.exit_ts

    out.stats = summarise([t.net_points for t in out.trades], cost_points)
    return out


def _index_at(times: Sequence[int], ts: int) -> int | None:
    """First minute at or after ``ts``. Binary search; the series is sorted."""
    lo, hi = 0, len(times)
    while lo < hi:
        mid = (lo + hi) // 2
        if times[mid] < ts:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < len(times) else None
