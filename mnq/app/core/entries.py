"""Where inside a bar to enter, and what it costs to wait.

"Optimal entry" is an execution question, and it separates cleanly from the
question of whether the signal is any good. That separation is the whole point
of this module: signal quality varies and is hard to establish, but *where the
fills are* is a stable, directly measurable property of the price series.

The rule family
---------------
Given a directional call at a bar's open, place a limit ``k`` volatility units
better than the open and hold to the bar's close::

    long   entry = open - k*sigma,  filled iff low  <= entry
    short  entry = open + k*sigma,  filled iff high >= entry

``k = 0`` is a market order at the open — what the current paper trader does.

Why this is exactly computable
------------------------------
Working in the standardised units of :mod:`app.core.candle`, where ``u`` is the
bar's up excursion, ``d`` its down excursion and ``c`` its net move, all
divided by the volatility known before the bar opened:

* a long at depth ``k`` **fills** iff ``d >= k``,
* its result is ``(close - entry)/sigma = c + k``,
* its worst point after entry is ``-(d - k)``  (adverse excursion),
* its best point after entry is ``u + k``      (favourable excursion).

So fill rate, price improvement, stop distance and target distance all fall out
of the same three numbers per bar. No simulation required, no path assumption.

The trap this is built to expose
--------------------------------
Waiting for a better price looks free. It is not, and the cost has a name:
**adverse selection**. A limit ``k`` below the open only fills on bars that
first traded down through it — and if the series trends at all, those are
disproportionately the bars that kept going down. The naive backtest reports
"average profit per filled trade", which improves with depth because the entry
is better, while quietly discarding the bars that ran away in your favour and
never filled. Those were the winners.

:func:`fill_profile` therefore reports three things side by side:

* ``price_improvement`` — ``k``, the points you save, which is real,
* ``adverse_selection`` — ``E[c | filled] - E[c]``, the drift you give up by
  only trading bars that first went against you, which is the hidden cost,
* ``per_attempt`` — expectancy over **every signal**, filled or not, which is
  the only number that can be compared across depths.

Read ``per_attempt``. ``per_fill`` will nearly always look better with depth
and nearly always be a mirage.

The second trap, which is worse
------------------------------
"Filled because the bar's low touched my limit" assumes a single printed tick
is a fill you could have had, in size, with your order already resting there.
On real MNQ data that assumption *is the entire result*:

===============================  ==================  ====
fill model (5m, depth 0.5 sigma)  per attempt         PF
===============================  ==================  ====
low touched the limit             **+4.90 pts**       4.21
a 1-minute bar *closed* through   **-1.10 pts**       0.71
===============================  ==================  ====

Same bars, same rule, opposite conclusion. And a train/test split does **not**
catch it — both halves of the data agree, because the bias lives in the fill
assumption rather than in overfitting. Out-of-sample testing protects against
choosing the wrong parameters; nothing but a second fill model protects against
choosing the wrong physics.

So both are computed, always, and :func:`recommend` uses the conservative one
whenever minute bars are available to build it. See :data:`TOUCH` and
:data:`SUSTAINED`. If the two disagree in sign, the honest reading is that
there is no demonstrated edge — only a fill assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..models import Bar
from .candle import EWMA_LAMBDA, Shape, shapes
from .trading import DEFAULT_COST_POINTS, DOLLARS_PER_POINT

#: Entry depths swept by default, in volatility units below/above the open.
DEFAULT_OFFSETS = (0.0, 0.1, 0.25, 0.4, 0.5, 0.75, 1.0, 1.5)
#: Fills needed before a depth's statistics are worth reporting.
MIN_FILLS = 30

LONG = 1
SHORT = -1

#: Resolution of the source series. Timeframes at or below this have no finer
#: path stored, so the conservative fill model cannot be built for them.
MINUTE_SECONDS = 60

#: Optimistic: the bar's wick reaching the limit counts as a fill.
TOUCH = "touch"
#: Conservative: a 1-minute bar must *close* through the limit. Requires the
#: minute series, which this system always has because 1m is its source of
#: truth for every timeframe.
SUSTAINED = "sustained"


@dataclass
class OffsetStats:
    """What entering ``offset`` volatility units better than the open produces."""

    offset: float
    direction: int
    attempts: int = 0
    fills: int = 0
    #: All of the following are in volatility units unless the name says points.
    price_improvement: float = 0.0
    unconditional_move: float = 0.0  # E[c] over every bar, in the trade's sign
    conditional_move: float = 0.0  # E[c | filled], same sign
    adverse_selection: float = 0.0  # conditional - unconditional; negative is bad
    per_fill: float = 0.0  # expectancy given a fill, costs included
    per_attempt: float = 0.0  # expectancy per signal, unfilled counted as zero
    win_rate: float = 0.0
    profit_factor: float | None = None
    #: Excursions measured *from the entry*, which is what a stop and a target
    #: have to be sized against.
    median_adverse: float = 0.0
    median_favourable: float = 0.0
    p90_adverse: float = 0.0
    #: Points-denominated versions, using the current volatility.
    sigma: float = 0.0
    #: Which fill assumption produced these numbers. See TOUCH / SUSTAINED.
    fill_model: str = TOUCH

    @property
    def fill_rate(self) -> float:
        return self.fills / self.attempts if self.attempts else 0.0

    def points(self, value: float) -> float:
        return value * self.sigma

    def as_dict(self) -> dict[str, Any]:
        return {
            "offset": self.offset,
            "direction": "long" if self.direction == LONG else "short",
            "attempts": self.attempts,
            "fills": self.fills,
            "fill_rate": round(self.fill_rate, 4),
            "price_improvement": round(self.price_improvement, 4),
            "unconditional_move": round(self.unconditional_move, 4),
            "conditional_move": round(self.conditional_move, 4),
            "adverse_selection": round(self.adverse_selection, 4),
            "per_fill": round(self.per_fill, 4),
            "per_attempt": round(self.per_attempt, 4),
            "per_attempt_points": round(self.points(self.per_attempt), 3),
            "per_attempt_dollars": round(
                self.points(self.per_attempt) * DOLLARS_PER_POINT, 2
            ),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": (
                None if self.profit_factor is None else round(self.profit_factor, 3)
            ),
            "median_adverse": round(self.median_adverse, 4),
            "median_favourable": round(self.median_favourable, 4),
            "p90_adverse": round(self.p90_adverse, 4),
            "suggested_stop_points": round(self.points(self.p90_adverse), 2),
            "suggested_target_points": round(self.points(self.median_favourable), 2),
            "sigma": round(self.sigma, 3),
            "fill_model": self.fill_model,
        }


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def offset_stats(
    history: Sequence[Shape],
    offset: float,
    direction: int = LONG,
    cost_points: float = DEFAULT_COST_POINTS,
    sigma: float = 0.0,
) -> OffsetStats:
    """Exact statistics for one entry depth, over every bar in ``history``.

    Every bar counts as one attempt — a signal you would have acted on. Bars
    that never trade through the limit are fills you did not get, and they
    enter ``per_attempt`` as zero, not as missing data.
    """
    stats = OffsetStats(
        offset=offset,
        direction=direction,
        attempts=len(history),
        price_improvement=offset,
        sigma=sigma,
    )
    if not history:
        return stats

    # Sign convention: work in "points in favour of the trade" throughout, so
    # long and short share one code path.
    signed = [direction * s.net for s in history]
    stats.unconditional_move = sum(signed) / len(signed)

    cost = (cost_points / sigma) if sigma > 0 else 0.0

    results: list[float] = []
    adverse: list[float] = []
    favourable: list[float] = []
    for s, move in zip(history, signed):
        # A long fills on the down excursion, a short on the up excursion.
        excursion = s.down if direction == LONG else s.up
        if excursion < offset:
            continue
        results.append(move + offset - cost)
        # After entering ``offset`` better than the open, the remaining travel
        # against you is that much smaller and the travel in your favour that
        # much larger.
        adverse.append(excursion - offset)
        favourable.append((s.up if direction == LONG else s.down) + offset)

    stats.fills = len(results)
    if not results:
        return stats

    stats.conditional_move = sum(m for m in results) / len(results) - offset + cost
    stats.adverse_selection = stats.conditional_move - stats.unconditional_move
    stats.per_fill = sum(results) / len(results)
    stats.per_attempt = sum(results) / stats.attempts
    stats.win_rate = sum(1 for r in results if r > 0) / len(results)

    gross_win = sum(r for r in results if r > 0)
    gross_loss = -sum(r for r in results if r < 0)
    stats.profit_factor = (gross_win / gross_loss) if gross_loss > 0 else None

    stats.median_adverse = _quantile(sorted(adverse), 0.5)
    stats.p90_adverse = _quantile(sorted(adverse), 0.9)
    stats.median_favourable = _quantile(sorted(favourable), 0.5)
    return stats


def sustained_bars(
    bars: Sequence[Bar], minutes: Sequence[Bar], step_seconds: int
) -> list[Bar]:
    """Re-cut each bar's extremes to the most extreme 1-minute *close* inside it.

    A level that a whole minute closed through is one price genuinely traded
    and held at, rather than one tick that printed and vanished. Feeding these
    bars through the same sweep gives the conservative fill model, and the gap
    between the two is the honest measure of how much a result depends on
    getting filled on spikes.

    Open and close are never widened away — a bar's own open and close are
    prices that certainly traded, so they bound the re-cut extremes.
    """
    # A 1-minute bar has no sub-minute path to resolve against: re-cutting it
    # to its own close just deletes both wicks and leaves the body, which is a
    # different instrument rather than a stricter fill model. Anything at or
    # below the minute resolution is returned untouched.
    if not minutes or step_seconds <= MINUTE_SECONDS:
        return list(bars)

    buckets: dict[int, list[Bar]] = {}
    for m in minutes:
        buckets.setdefault(m.ts - (m.ts % step_seconds), []).append(m)

    out: list[Bar] = []
    for bar in bars:
        inner = buckets.get(bar.ts - (bar.ts % step_seconds), ())
        closes = [m.close for m in inner if bar.ts <= m.ts < bar.ts + step_seconds]
        if not closes:
            out.append(bar)
            continue
        out.append(
            Bar(
                ts=bar.ts,
                open=bar.open,
                high=max(max(closes), bar.open, bar.close),
                low=min(min(closes), bar.open, bar.close),
                close=bar.close,
                volume=bar.volume,
                count=bar.count,
                complete=bar.complete,
            )
        )
    return out


def fill_profile(
    bars: Sequence[Bar],
    offsets: Sequence[float] = DEFAULT_OFFSETS,
    direction: int = LONG,
    cost_points: float = DEFAULT_COST_POINTS,
    lam: float = EWMA_LAMBDA,
    fill_model: str = TOUCH,
) -> list[OffsetStats]:
    """Sweep entry depths for one timeframe.

    ``bars`` should already carry the extremes matching ``fill_model`` — pass
    the output of :func:`sustained_bars` for the conservative one. The label is
    recorded on each result so a table can never mix the two silently.
    """
    history = shapes(bars, lam)
    sigma = history[-1].sigma if history else 0.0
    out = []
    for k in offsets:
        stats = offset_stats(history, k, direction, cost_points, sigma)
        stats.fill_model = fill_model
        out.append(stats)
    return out


def profiles(
    bars: Sequence[Bar],
    minutes: Sequence[Bar] = (),
    step_seconds: int = 0,
    offsets: Sequence[float] = DEFAULT_OFFSETS,
    direction: int = LONG,
    cost_points: float = DEFAULT_COST_POINTS,
    lam: float = EWMA_LAMBDA,
) -> dict[str, list[OffsetStats]]:
    """Both fill models side by side, so neither can be read on its own."""
    out = {
        TOUCH: fill_profile(bars, offsets, direction, cost_points, lam, TOUCH)
    }
    if minutes and step_seconds > MINUTE_SECONDS:
        out[SUSTAINED] = fill_profile(
            sustained_bars(bars, minutes, step_seconds),
            offsets, direction, cost_points, lam, SUSTAINED,
        )
    return out


def disagree(profile_map: dict[str, list[OffsetStats]]) -> list[float]:
    """Depths where the two fill models disagree about the sign of the edge.

    Any depth listed here is one where the result is a statement about fills,
    not about the market.
    """
    optimistic = {p.offset: p for p in profile_map.get(TOUCH, ())}
    conservative = {p.offset: p for p in profile_map.get(SUSTAINED, ())}
    return [
        k for k in sorted(optimistic)
        if k in conservative
        and optimistic[k].per_attempt > 0 >= conservative[k].per_attempt
    ]


@dataclass
class Recommendation:
    """The depth that maximises expectancy per attempt, with its context."""

    timeframe: str = ""
    direction: int = LONG
    best: OffsetStats | None = None
    at_market: OffsetStats | None = None
    reason: str = ""
    fill_model: str = TOUCH
    #: Depths where the optimistic and conservative fill models disagree about
    #: the sign of the edge. Non-empty means "this is a fill assumption".
    contested: list[float] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        """Points per signal gained over entering at the open."""
        if self.best is None or self.at_market is None:
            return 0.0
        return (self.best.per_attempt - self.at_market.per_attempt) * self.best.sigma

    @property
    def trustworthy(self) -> bool:
        """Whether the recommendation survives the conservative fill model."""
        return self.fill_model == SUSTAINED and not self.contested

    def as_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "direction": "long" if self.direction == LONG else "short",
            "reason": self.reason,
            "fill_model": self.fill_model,
            "contested": list(self.contested),
            "trustworthy": self.trustworthy,
            "improvement_points": round(self.improvement, 3),
            "best": None if self.best is None else self.best.as_dict(),
            "at_market": None if self.at_market is None else self.at_market.as_dict(),
        }


def recommend(
    bars: Sequence[Bar],
    timeframe: str = "",
    offsets: Sequence[float] = DEFAULT_OFFSETS,
    direction: int = LONG,
    cost_points: float = DEFAULT_COST_POINTS,
    min_fills: int = MIN_FILLS,
    lam: float = EWMA_LAMBDA,
    minutes: Sequence[Bar] = (),
    step_seconds: int = 0,
) -> Recommendation:
    """Pick the entry depth with the best expectancy per attempt.

    Scored on the **conservative** fill model whenever ``minutes`` are supplied,
    because recommending a depth off the optimistic one would be recommending a
    fill nobody is guaranteed to get. Depths where the two models disagree about
    the sign are listed in ``contested`` and the recommendation says so.

    Depths with too few fills are excluded rather than ranked: a deep limit that
    filled four times can post a spectacular per-attempt number that is pure
    small-sample noise, and it is exactly the kind of result this analysis
    exists to avoid endorsing.
    """
    both = profiles(bars, minutes, step_seconds, offsets, direction, cost_points, lam)
    model = SUSTAINED if SUSTAINED in both else TOUCH
    profile = both[model]
    contested = disagree(both)

    if not profile or not profile[0].attempts:
        return Recommendation(timeframe=timeframe, direction=direction,
                              fill_model=model, contested=contested,
                              reason="no standardised bars")

    at_market = next((p for p in profile if p.offset == 0.0), profile[0])
    usable = [p for p in profile if p.fills >= min_fills]
    if not usable:
        return Recommendation(
            timeframe=timeframe, direction=direction, at_market=at_market,
            fill_model=model, contested=contested,
            reason=f"no depth reached {min_fills} fills",
        )

    best = max(usable, key=lambda p: p.per_attempt)

    if best.per_attempt <= 0:
        # Every depth loses. Ranking them would dress "least bad" up as advice,
        # and the fault is upstream: entry placement cannot rescue a call with
        # no edge, it can only change how fast the costs arrive.
        return Recommendation(
            timeframe=timeframe, direction=direction, best=None,
            at_market=at_market, fill_model=model, contested=contested,
            reason=(
                "no entry depth has positive expectancy "
                f"(best is {best.offset:g} sigma at "
                f"{best.per_attempt * best.sigma:+.2f} pts per signal) — "
                "the direction call is what needs fixing, not the entry"
            ),
        )

    if best.per_attempt <= at_market.per_attempt:
        best, reason = at_market, "no depth improved on entering at the open"
    elif best.offset == 0.0:
        reason = "entering at the open beats waiting; the pullback costs more than it saves"
    else:
        reason = (
            f"waiting {best.offset:g} sigma fills {best.fill_rate:.0%} of the time "
            f"and adds {(best.per_attempt - at_market.per_attempt) * best.sigma:+.2f} "
            "pts per signal"
        )
    if contested:
        reason += (
            f" — but depths {', '.join(f'{k:g}' for k in contested)} only profit "
            "if a wick print fills you, so treat this as a fill assumption"
        )
    return Recommendation(
        timeframe=timeframe, direction=direction, best=best,
        at_market=at_market, fill_model=model, contested=contested, reason=reason,
    )


# ---------------------------------------------------------------------------
# Path-resolved simulation
# ---------------------------------------------------------------------------
#
# Everything above holds a position to the bar's close, which needs no path
# assumption. Adding a stop or a target does: from OHLC alone you cannot tell
# whether the high or the low came first, and guessing decides the result.
#
# This system stores 1-minute bars as its single source of truth for every
# timeframe, so the path *is* available — a 30-minute bar is thirty ordered
# minute bars. Resolving against them turns a guess into a measurement.


@dataclass
class TradeOutcome:
    filled: bool = False
    entry: float | None = None
    exit: float | None = None
    exit_reason: str = "unfilled"  # target | stop | close | unfilled
    net_points: float = 0.0
    minutes_held: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "filled": self.filled,
            "entry": None if self.entry is None else round(self.entry, 2),
            "exit": None if self.exit is None else round(self.exit, 2),
            "exit_reason": self.exit_reason,
            "net_points": round(self.net_points, 2),
            "minutes_held": self.minutes_held,
        }


def minutes_within(minutes: Sequence[Bar], start_ts: int, end_ts: int) -> list[Bar]:
    """The 1-minute bars making up one higher-timeframe bar, in order."""
    return [m for m in minutes if start_ts <= m.ts < end_ts]


def simulate_entry(
    parent: Bar,
    path: Sequence[Bar],
    direction: int,
    offset_points: float,
    stop_points: float = 0.0,
    target_points: float = 0.0,
    cost_points: float = DEFAULT_COST_POINTS,
) -> TradeOutcome:
    """Walk one bar minute by minute, filling and exiting in real order.

    Two honest conservatisms, both of which cost the simulation money rather
    than granting it any:

    * within the single minute where both the stop and the target lie inside
      the range, the **stop** is taken — 1-minute granularity cannot say which
      came first, and the pessimistic reading is the one that will not flatter
      a strategy into looking tradeable;
    * a limit fills at the limit price, except when the minute *opens* through
      it, in which case the fill is that better open — which is what actually
      happens, and is not a favour, since it means price gapped past you.

    With no path available the function falls back to the parent bar's OHLC and
    reports honestly, which for a stop/target trade means the conservative
    stop-first reading applies to the whole bar.
    """
    out = TradeOutcome()
    limit = parent.open - direction * offset_points
    sequence = list(path) or [parent]

    entry: float | None = None
    filled_at = 0
    for i, minute in enumerate(sequence):
        if entry is None:
            if offset_points <= 0:
                # Market at the open: only the first minute can fill it.
                entry = parent.open if i == 0 else None
                if entry is None:
                    continue
            elif direction == LONG and minute.low <= limit:
                entry = min(limit, minute.open)
            elif direction == SHORT and minute.high >= limit:
                entry = max(limit, minute.open)
            else:
                continue
            out.filled = True
            out.entry = entry
            filled_at = i

        out.minutes_held = i - filled_at + 1
        stop = entry - direction * stop_points if stop_points > 0 else None
        target = entry + direction * target_points if target_points > 0 else None

        # Stop before target, deliberately. See the docstring.
        if stop is not None and _touched(minute, direction, stop, adverse=True):
            out.exit, out.exit_reason = stop, "stop"
            break
        if target is not None and _touched(minute, direction, target, adverse=False):
            out.exit, out.exit_reason = target, "target"
            break
    else:
        if entry is not None:
            out.exit, out.exit_reason = parent.close, "close"

    if out.entry is None:
        return out
    if out.exit is None:  # filled on the last minute and never resolved
        out.exit, out.exit_reason = parent.close, "close"
    out.net_points = direction * (out.exit - out.entry) - cost_points
    return out


def _touched(minute: Bar, direction: int, level: float, adverse: bool) -> bool:
    """Did this minute trade through ``level`` in the relevant direction?"""
    against = (direction == LONG) == adverse
    return minute.low <= level if against else minute.high >= level
