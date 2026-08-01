"""Daily swing structure — pivots, HH/LH/HL/LL labels, and the trend verdict.

This lives in ``core`` rather than inside the indicator because the forecast
needs the *history* of the structure read, not just its current value: to fit a
coefficient for the structure factor you need what the structure was at every
past bar, using only information available then.

:func:`as_of_labels` is the piece that makes that safe. A fractal pivot at bar
``j`` is not knowable until ``j + lookback`` bars have printed, so each bar is
given only the labels whose pivots had already been confirmed by that point.
"""

from __future__ import annotations

from typing import Sequence

from ..models import Bar

#: Structure verdict -> directional bias in [-1, 1].
STRUCTURE_BIAS = {
    "Uptrend": 1.0,
    "Range (bullish tilt)": 0.5,
    "Range": 0.0,
    "Range (bearish tilt)": -0.5,
    "Downtrend": -1.0,
}


def find_pivots(bars: Sequence[Bar], lookback: int) -> tuple[list[int], list[int]]:
    """Return (indices of pivot highs, indices of pivot lows)."""
    highs: list[int] = []
    lows: list[int] = []
    n = len(bars)
    for i in range(lookback, n - lookback):
        window = range(i - lookback, i + lookback + 1)
        if all(bars[i].high > bars[j].high for j in window if j != i):
            highs.append(i)
        if all(bars[i].low < bars[j].low for j in window if j != i):
            lows.append(i)
    return highs, lows


def label_swings(
    bars: Sequence[Bar], lookback: int
) -> tuple[list[tuple[int, float, str]], list[tuple[int, float, str]]]:
    """Label each swing against the previous swing of the same kind.

    Returns (highs, lows) as ``(index, price, label)``; labels are HH/LH for
    highs and HL/LL for lows, with H/L for the first of each.
    """
    high_idx, low_idx = find_pivots(bars, lookback)

    highs: list[tuple[int, float, str]] = []
    prev = None
    for i in high_idx:
        price = bars[i].high
        label = "HH" if prev is not None and price > prev else ("LH" if prev is not None else "H")
        highs.append((i, price, label))
        prev = price

    lows: list[tuple[int, float, str]] = []
    prev = None
    for i in low_idx:
        price = bars[i].low
        label = "HL" if prev is not None and price > prev else ("LL" if prev is not None else "L")
        lows.append((i, price, label))
        prev = price

    return highs, lows


def classify(high_label: str | None, low_label: str | None) -> str:
    """Combine the latest swing-high and swing-low labels into a verdict."""
    if not high_label or not low_label:
        return "Range"
    if high_label == "HH" and low_label == "HL":
        return "Uptrend"
    if high_label == "LH" and low_label == "LL":
        return "Downtrend"
    if high_label == "HH" or low_label == "HL":
        return "Range (bullish tilt)"
    if high_label == "LH" or low_label == "LL":
        return "Range (bearish tilt)"
    return "Range"


def as_of_labels(bars: Sequence[Bar], lookback: int) -> list[str]:
    """Structure verdict as it was knowable at each bar — no lookahead.

    A pivot at index ``j`` becomes visible only at ``j + lookback``, so bar
    ``i`` sees only swings with ``j + lookback <= i``.
    """
    n = len(bars)
    if n == 0:
        return []
    highs, lows = label_swings(bars, lookback)

    out: list[str] = []
    hi = li = 0
    last_high = last_low = None
    for i in range(n):
        while hi < len(highs) and highs[hi][0] + lookback <= i:
            last_high = highs[hi][2]
            hi += 1
        while li < len(lows) and lows[li][0] + lookback <= i:
            last_low = lows[li][2]
            li += 1
        out.append(classify(last_high, last_low))
    return out


def bias(label: str) -> float:
    return STRUCTURE_BIAS.get(label, 0.0)
