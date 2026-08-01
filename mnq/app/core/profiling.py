"""Measure the structural properties of a bar series.

Six families of measurement, each reported with sample sizes and confidence
intervals so a number can be told apart from noise:

1. **Volatility** — how far price travels per bar, and whether that clusters.
2. **Acceleration** — whether range expansion persists into the next bar.
3. **Continuation** — whether direction persists, overall and conditionally.
4. **Range scaling** — how bar size grows with timeframe. The exponent is the
   headline: 0.5 is a random walk, above is trending, below is mean-reverting.
5. **Wick / body shape** — and whether wick asymmetry predicts the next bar.
6. **Hour of day** — all of the above, split by session hour.

Nothing here forecasts. It answers "is there anything in this series worth
modelling", which is the question to settle before building a model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence
from zoneinfo import ZoneInfo

from ..models import Bar
from .features import ewma_volatility, garman_klass_variance, volatility_persistence
from .stats import (
    autocorrelation,
    correlation,
    correlation_stderr,
    mean,
    ols_slope,
    significant,
    stdev,
    terciles,
    wilson_interval,
)


def bar_step(bars: Sequence[Bar]) -> int:
    """Modal spacing between consecutive bars, in seconds."""
    if len(bars) < 3:
        return 0
    gaps: dict[int, int] = {}
    for i in range(len(bars) - 1):
        gap = bars[i + 1].ts - bars[i].ts
        gaps[gap] = gaps.get(gap, 0) + 1
    return max(gaps, key=gaps.get)


def contiguous_runs(bars: Sequence[Bar]) -> list[list[Bar]]:
    """Split a series wherever bars are not exactly one step apart.

    Weekends, the daily maintenance break and any missing data leave gaps. A
    close-to-close difference across one of those is not a bar-sized move, and
    treating it as one inflates volatility and invents drift, so every
    measurement here works run by run.
    """
    series = [b for b in bars if b.complete]
    step = bar_step(series)
    if step <= 0:
        return [series] if series else []

    runs: list[list[Bar]] = []
    current: list[Bar] = []
    for bar in series:
        if current and bar.ts - current[-1].ts != step:
            if len(current) > 1:
                runs.append(current)
            current = []
        current.append(bar)
    if len(current) > 1:
        runs.append(current)
    return runs


def returns(bars: Sequence[Bar]) -> list[float]:
    """Close-to-close changes in points, across contiguous bars only."""
    out: list[float] = []
    for run in contiguous_runs(bars):
        out.extend(run[i + 1].close - run[i].close for i in range(len(run) - 1))
    return out


def return_pairs(bars: Sequence[Bar]) -> list[tuple[Bar, float, float]]:
    """(bar, this return, next return) for genuinely consecutive triples."""
    out: list[tuple[Bar, float, float]] = []
    for run in contiguous_runs(bars):
        rets = [run[i + 1].close - run[i].close for i in range(len(run) - 1)]
        for i in range(len(rets) - 1):
            out.append((run[i], rets[i], rets[i + 1]))
    return out


def bar_returns(bars: Sequence[Bar]) -> list[tuple[Bar, float]]:
    """(bar, the move from this bar's close to the next) for contiguous pairs."""
    out: list[tuple[Bar, float]] = []
    for run in contiguous_runs(bars):
        out.extend((run[i], run[i + 1].close - run[i].close) for i in range(len(run) - 1))
    return out


# ---------------------------------------------------------------------------
# 1. Volatility
# ---------------------------------------------------------------------------


@dataclass
class VolatilityProfile:
    bars: int = 0
    mean_range: float = 0.0
    median_range: float = 0.0
    close_to_close_sd: float = 0.0
    garman_klass_sd: float = 0.0
    persistence: float = 0.0
    range_p10: float = 0.0
    range_p90: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def volatility_profile(bars: Sequence[Bar]) -> VolatilityProfile:
    series = [b for b in bars if b.complete]
    if len(series) < 5:
        return VolatilityProfile(bars=len(series))
    ranges = sorted(b.range for b in series)
    gk = [garman_klass_variance(b) for b in series]
    price = mean([b.close for b in series])
    return VolatilityProfile(
        bars=len(series),
        mean_range=mean([b.range for b in series]),
        median_range=ranges[len(ranges) // 2],
        close_to_close_sd=stdev(returns(series)),
        garman_klass_sd=math.sqrt(mean(gk)) * price,
        persistence=volatility_persistence(series, lookback=len(series)),
        range_p10=ranges[int(0.10 * len(ranges))],
        range_p90=ranges[int(0.90 * len(ranges))],
    )


# ---------------------------------------------------------------------------
# 2. Acceleration (range expansion)
# ---------------------------------------------------------------------------


@dataclass
class AccelerationProfile:
    samples: int = 0
    expansion_autocorr: float = 0.0
    expansion_stderr: float = 0.0
    big_then_big_rate: float | None = None
    big_then_big_ci: tuple[float, float] | None = None
    significant: bool = False

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def acceleration_profile(bars: Sequence[Bar], lookback: int = 20) -> AccelerationProfile:
    """Does an expanding bar predict another expanding bar?

    Range is normalised by its own trailing average so the answer is about
    expansion relative to the regime, not absolute size.
    """
    series = [b for b in bars if b.complete]
    if len(series) < lookback + 30:
        return AccelerationProfile(samples=0)

    ratios: list[float] = []
    for i in range(lookback, len(series)):
        window = series[i - lookback : i]
        base = mean([b.range for b in window])
        ratios.append(series[i].range / base if base > 0 else 1.0)

    median = sorted(ratios)[len(ratios) // 2]
    pairs = [(ratios[i], ratios[i + 1]) for i in range(len(ratios) - 1)]
    big = [nxt for cur, nxt in pairs if cur > median]
    hits = sum(1 for nxt in big if nxt > median)
    ci = wilson_interval(hits, len(big)) if big else None

    return AccelerationProfile(
        samples=len(ratios),
        expansion_autocorr=autocorrelation(ratios),
        expansion_stderr=correlation_stderr(len(ratios)),
        big_then_big_rate=(hits / len(big)) if big else None,
        big_then_big_ci=ci,
        significant=significant(ci) if ci else False,
    )


# ---------------------------------------------------------------------------
# 3. Continuation (direction persistence)
# ---------------------------------------------------------------------------


@dataclass
class ContinuationProfile:
    samples: int = 0
    same_direction_rate: float | None = None
    ci: tuple[float, float] | None = None
    significant: bool = False
    return_autocorr: float = 0.0
    autocorr_stderr: float = 0.0
    after_big_rate: float | None = None
    after_big_ci: tuple[float, float] | None = None
    after_small_rate: float | None = None
    after_small_ci: tuple[float, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def continuation_profile(bars: Sequence[Bar]) -> ContinuationProfile:
    """Does this bar's direction carry into the next one?

    50% is a coin flip. Above means trending, below means mean-reverting; the
    interval decides whether either claim is supportable.
    """
    series = [b for b in bars if b.complete]
    rets = returns(series)
    pairs = [(a, b) for _, a, b in return_pairs(series) if a != 0 and b != 0]
    if len(pairs) < 30:
        return ContinuationProfile(samples=len(pairs))

    hits = sum(1 for a, b in pairs if (a > 0) == (b > 0))
    ci = wilson_interval(hits, len(pairs))

    # Split by the size of the first move: big moves and small moves can behave
    # differently (exhaustion versus drift).
    sizes = [abs(a) for a, _ in pairs]
    low_cut, high_cut = terciles(sizes)
    big = [(a, b) for a, b in pairs if abs(a) >= high_cut]
    small = [(a, b) for a, b in pairs if abs(a) <= low_cut]

    def rate(subset):
        if len(subset) < 20:
            return None, None
        h = sum(1 for a, b in subset if (a > 0) == (b > 0))
        return h / len(subset), wilson_interval(h, len(subset))

    big_rate, big_ci = rate(big)
    small_rate, small_ci = rate(small)

    return ContinuationProfile(
        samples=len(pairs),
        same_direction_rate=hits / len(pairs),
        ci=ci,
        significant=significant(ci),
        return_autocorr=autocorrelation(rets),
        autocorr_stderr=correlation_stderr(len(rets)),
        after_big_rate=big_rate,
        after_big_ci=big_ci,
        after_small_rate=small_rate,
        after_small_ci=small_ci,
    )


# ---------------------------------------------------------------------------
# 4. Range scaling across timeframes
# ---------------------------------------------------------------------------


@dataclass
class ScalingProfile:
    points: list[tuple[int, float]] = field(default_factory=list)  # (minutes, mean |move|)
    exponent: float = 0.0
    interpretation: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "points": self.points,
            "exponent": self.exponent,
            "interpretation": self.interpretation,
        }


def scaling_profile(series_by_minutes: dict[int, Sequence[Bar]]) -> ScalingProfile:
    """Fit mean |move| ~ T^b across timeframes.

    For a random walk variance grows linearly with time, so typical absolute
    move grows with the square root: ``b = 0.5``. Above 0.5 means moves compound
    (trending); below means they partly cancel (mean-reverting). This is the
    same idea as a variance-ratio or Hurst test, read off timeframes you
    already compute.
    """
    points: list[tuple[int, float]] = []
    for minutes in sorted(series_by_minutes):
        bars = [b for b in series_by_minutes[minutes] if b.complete]
        rets = [abs(r) for r in returns(bars)]
        if len(rets) >= 30:
            points.append((minutes, mean(rets)))

    if len(points) < 3:
        return ScalingProfile(points=points)

    xs = [math.log(m) for m, _ in points]
    ys = [math.log(v) for _, v in points if v > 0]
    if len(ys) != len(xs):
        return ScalingProfile(points=points)

    b = ols_slope(xs, ys)
    if b > 0.55:
        note = "moves compound faster than a random walk (trending)"
    elif b < 0.45:
        note = "moves partly cancel (mean-reverting)"
    else:
        note = "consistent with a random walk"
    return ScalingProfile(points=points, exponent=b, interpretation=note)


# ---------------------------------------------------------------------------
# 5. Wick / body shape
# ---------------------------------------------------------------------------


@dataclass
class WickProfile:
    bars: int = 0
    body_fraction: float = 0.0
    upper_fraction: float = 0.0
    lower_fraction: float = 0.0
    wick_to_body: float = 0.0
    after_long_lower_up_rate: float | None = None
    after_long_lower_ci: tuple[float, float] | None = None
    after_long_lower_significant: bool = False
    after_long_upper_up_rate: float | None = None
    after_long_upper_ci: tuple[float, float] | None = None
    after_long_upper_significant: bool = False
    lower_samples: int = 0
    upper_samples: int = 0

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def wick_profile(bars: Sequence[Bar], dominance: float = 0.5) -> WickProfile:
    """Shape of the average bar, and whether wick asymmetry predicts anything.

    The folk claim is that a long lower wick is a rejection of lower prices and
    therefore bullish. It is testable: take bars whose lower wick is at least
    ``dominance`` of the range and measure how often the next bar closes up.
    """
    series = [b for b in bars if b.complete and b.range > 0]
    if len(series) < 30:
        return WickProfile(bars=len(series))

    bodies, uppers, lowers = [], [], []
    for bar in series:
        upper = bar.high - max(bar.open, bar.close)
        lower = min(bar.open, bar.close) - bar.low
        bodies.append(bar.body / bar.range)
        uppers.append(upper / bar.range)
        lowers.append(lower / bar.range)

    body_fraction = mean(bodies)
    index = {id(b): i for i, b in enumerate(series)}
    long_lower, long_upper = [], []
    for run in contiguous_runs(series):
        for i in range(len(run) - 1):
            nxt = run[i + 1].close - run[i + 1].open
            if nxt == 0:
                continue
            pos = index.get(id(run[i]))
            if pos is None:
                continue
            if lowers[pos] >= dominance:
                long_lower.append(nxt > 0)
            if uppers[pos] >= dominance:
                long_upper.append(nxt > 0)

    def rate(flags):
        if len(flags) < 20:
            return None, None
        hits = sum(flags)
        return hits / len(flags), wilson_interval(hits, len(flags))

    lower_rate, lower_ci = rate(long_lower)
    upper_rate, upper_ci = rate(long_upper)

    return WickProfile(
        bars=len(series),
        body_fraction=body_fraction,
        upper_fraction=mean(uppers),
        lower_fraction=mean(lowers),
        wick_to_body=((1.0 - body_fraction) / body_fraction) if body_fraction > 0 else 0.0,
        after_long_lower_up_rate=lower_rate,
        after_long_lower_ci=lower_ci,
        after_long_lower_significant=bool(lower_ci and significant(lower_ci)),
        lower_samples=len(long_lower),
        after_long_upper_up_rate=upper_rate,
        after_long_upper_ci=upper_ci,
        after_long_upper_significant=bool(upper_ci and significant(upper_ci)),
        upper_samples=len(long_upper),
    )


# ---------------------------------------------------------------------------
# 6. Hour of day
# ---------------------------------------------------------------------------


@dataclass
class HourStats:
    hour: int = 0
    bars: int = 0
    mean_range: float = 0.0
    mean_abs_move: float = 0.0
    mean_move: float = 0.0
    move_stderr: float = 0.0
    continuation_rate: float | None = None
    continuation_ci: tuple[float, float] | None = None
    body_fraction: float = 0.0
    mean_volume: float = 0.0

    @property
    def drift_significant(self) -> bool:
        """Whether the average move differs from zero by more than 2 SE."""
        return self.move_stderr > 0 and abs(self.mean_move) > 2 * self.move_stderr

    def as_dict(self) -> dict[str, Any]:
        out = self.__dict__.copy()
        out["drift_significant"] = self.drift_significant
        return out


def hourly_profile(bars: Sequence[Bar], tz: str) -> list[HourStats]:
    """Split the series by exchange-local hour and profile each bucket."""
    zone = ZoneInfo(tz)
    series = [b for b in bars if b.complete]

    def hour_of(bar: Bar) -> int:
        return datetime.fromtimestamp(bar.ts, tz=zone).hour

    # Bucket the bars themselves for shape statistics...
    shape: dict[int, list[Bar]] = {}
    for bar in series:
        shape.setdefault(hour_of(bar), []).append(bar)

    # ...but bucket *moves* by the hour of the bar they start from, using only
    # genuinely consecutive bars. Diffing inside an hour bucket would subtract
    # yesterday's close from today's and report it as a bar-sized move.
    moves: dict[int, list[float]] = {}
    for bar, move in bar_returns(series):
        moves.setdefault(hour_of(bar), []).append(move)

    pairs_by_hour: dict[int, list[tuple[float, float]]] = {}
    for bar, first, second in return_pairs(series):
        if first != 0 and second != 0:
            pairs_by_hour.setdefault(hour_of(bar), []).append((first, second))

    out: list[HourStats] = []
    for hour in sorted(shape):
        group = shape[hour]
        rets = moves.get(hour, [])
        if len(group) < 10 or len(rets) < 5:
            continue

        pairs = pairs_by_hour.get(hour, [])
        cont_rate = cont_ci = None
        if len(pairs) >= 20:
            hits = sum(1 for a, b in pairs if (a > 0) == (b > 0))
            cont_rate, cont_ci = hits / len(pairs), wilson_interval(hits, len(pairs))

        bodies = [b.body / b.range for b in group if b.range > 0]
        out.append(
            HourStats(
                hour=hour,
                bars=len(group),
                mean_range=mean([b.range for b in group]),
                mean_abs_move=mean([abs(r) for r in rets]),
                mean_move=mean(rets),
                move_stderr=(stdev(rets) / math.sqrt(len(rets))) if len(rets) > 1 else 0.0,
                continuation_rate=cont_rate,
                continuation_ci=cont_ci,
                body_fraction=mean(bodies) if bodies else 0.0,
                mean_volume=mean([b.volume for b in group]),
            )
        )
    return out
