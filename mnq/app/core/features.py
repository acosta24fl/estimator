"""Shared series maths.

These are the primitives the indicators draw and the forecast reasons about.
They live here, outside ``indicators/``, so the forecast can reuse the exact
same calculation the chart is showing — a MACD divergence between "what you
see" and "what the model used" would make the projection impossible to trust.
"""

from __future__ import annotations

import math
from typing import Sequence

from ..models import Bar


def ema(values: Sequence[float], period: int) -> list[float | None]:
    """Exponential moving average, seeded with an SMA of the first ``period``.

    Returns a list aligned with ``values``; entries before the seed are None.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    alpha = 2.0 / (period + 1.0)
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = (values[i] - prev) * alpha + prev
        out[i] = prev
    return out


def macd(
    closes: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Return (macd line, signal line, histogram), each aligned to ``closes``."""
    fast_ema = ema(closes, fast)
    slow_ema = ema(closes, slow)

    macd_line: list[float | None] = [
        (f - s) if (f is not None and s is not None) else None
        for f, s in zip(fast_ema, slow_ema)
    ]

    # The signal EMA runs over the MACD line's defined region only, otherwise
    # the leading Nones would shift its seed.
    defined = [i for i, v in enumerate(macd_line) if v is not None]
    signal_line: list[float | None] = [None] * len(macd_line)
    if len(defined) >= signal:
        dense = [macd_line[i] for i in defined]
        seeded = ema(dense, signal)  # type: ignore[arg-type]
        for pos, i in enumerate(defined):
            signal_line[i] = seeded[pos]

    hist: list[float | None] = [
        (m - s) if (m is not None and s is not None) else None
        for m, s in zip(macd_line, signal_line)
    ]
    return macd_line, signal_line, hist


def session_vwap(bars: Sequence[Bar], session) -> list[float]:
    """Volume-weighted average price, restarting at each session boundary.

    Falls back to an unweighted mean of typical prices while the feed reports
    no volume, so the series stays continuous instead of dropping out.
    """
    out: list[float] = []
    current: int | None = None
    cum_pv = cum_vol = cum_typ = 0.0
    count = 0

    for bar in bars:
        start = session.start(bar.ts)
        if start != current:
            current = start
            cum_pv = cum_vol = cum_typ = 0.0
            count = 0
        cum_typ += bar.typical
        count += 1
        cum_pv += bar.typical * bar.volume
        cum_vol += bar.volume
        out.append((cum_pv / cum_vol) if cum_vol > 0 else (cum_typ / count))
    return out


def average_range(bars: Sequence[Bar], period: int) -> float:
    """Mean high-low range over the last ``period`` bars."""
    if not bars:
        return 0.0
    window = list(bars)[-period:]
    return sum(b.range for b in window) / len(window)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile. ``q`` in [0, 1]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def close_to_close_moves(bars: Sequence[Bar]) -> list[float]:
    """Absolute change from each bar's close to the next."""
    closes = [b.close for b in bars]
    return [abs(closes[i + 1] - closes[i]) for i in range(len(closes) - 1)]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def garman_klass_variance(bar: Bar) -> float:
    """Per-bar variance estimate from the full OHLC, in squared log-returns.

        0.5*ln(H/L)^2 - (2*ln2 - 1)*ln(C/O)^2

    Close-to-close uses one number per bar and throws the path away; this uses
    all four and is roughly 7x more statistically efficient, so the volatility
    estimate is far steadier for the same amount of data. It assumes no gap
    between bars, which is why it is clamped at zero rather than trusted blindly.
    """
    if bar.low <= 0 or bar.open <= 0 or bar.high <= 0 or bar.close <= 0:
        return 0.0
    hl = math.log(bar.high / bar.low)
    co = math.log(bar.close / bar.open)
    return max(0.5 * hl * hl - (2.0 * math.log(2.0) - 1.0) * co * co, 0.0)


def ewma_volatility(bars: Sequence[Bar], lam: float = 0.94) -> list[float]:
    """Conditional volatility in price points, one value per bar.

        sigma^2_t = lam * sigma^2_{t-1} + (1 - lam) * gk_t

    Entry ``i`` is the estimate formed *after* observing bar ``i``, i.e. the
    forecast of how far bar ``i+1`` will travel — so using it to size the band
    for the next bar involves no lookahead.

    Volatility clustering is the strongest regularity in price series: returns
    are close to unpredictable, but their magnitude is strongly autocorrelated.
    A fixed-window percentile ignores that and reacts slowly; this tracks it.
    """
    if not bars:
        return []
    out: list[float] = []
    var = garman_klass_variance(bars[0])
    for bar in bars:
        var = lam * var + (1.0 - lam) * garman_klass_variance(bar)
        out.append(math.sqrt(var) * bar.close)  # log-return sigma -> points
    return out


def volatility_persistence(bars: Sequence[Bar], lookback: int = 300) -> float:
    """How predictable this series' volatility actually is, in [0, 1].

    Lag-1 autocorrelation of per-bar Garman-Klass variance, clipped to [0, 1].

    An adaptive volatility estimate only helps when volatility clusters. Real
    futures data clusters strongly (typically 0.2-0.4); a series without
    clustering returns ~0, and reacting to recent volatility there is just
    tracking noise — it widens the band after a shock that carries no
    information about the next bar. Measuring this instead of assuming it lets
    the band blend toward a flat estimate exactly when adaptivity cannot help.
    """
    window = list(bars)[-lookback:]
    if len(window) < 20:
        return 0.0
    v = [garman_klass_variance(b) for b in window]
    mean = sum(v) / len(v)
    if mean <= 0:
        return 0.0
    den = sum((x - mean) ** 2 for x in v)
    # A constant series has zero spread in exact arithmetic but not in floating
    # point: rounding in `mean` leaves deviations around 1e-19, and their ratio
    # is meaningless noise that would otherwise read as near-perfect
    # persistence. Compare the spread to the level before trusting it.
    if math.sqrt(den / len(v)) <= 1e-6 * mean:
        return 0.0
    num = sum((v[i] - mean) * (v[i + 1] - mean) for i in range(len(v) - 1))
    return clamp(num / den, 0.0, 1.0)
