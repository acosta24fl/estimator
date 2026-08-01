"""Small statistics helpers shared by scoring and profiling.

Every rate this project reports comes with an interval, because at the sample
sizes involved a bare percentage is usually indistinguishable from noise.
"""

from __future__ import annotations

import math
from typing import Sequence


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion.

    Wilson rather than the normal approximation because it stays inside [0, 1]
    and behaves sensibly near 0, near 1, and at small n.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = hits / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def significant(interval: tuple[float, float], reference: float = 0.5) -> bool:
    """Whether an interval excludes the no-effect value."""
    return interval[0] > reference or interval[1] < reference


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def correlation(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a, b = a[:n], b[:n]
    ma, mb = mean(a), mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da > 0 and db > 0 else 0.0


def autocorrelation(values: Sequence[float], lag: int = 1) -> float:
    if len(values) <= lag + 2:
        return 0.0
    return correlation(values[:-lag], values[lag:])


def correlation_stderr(n: int) -> float:
    """Approximate standard error of a correlation under the null of zero."""
    return 1.0 / math.sqrt(n) if n > 3 else float("inf")


def ols_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Least-squares slope of ys on xs."""
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    xs, ys = xs[:n], ys[:n]
    mx, my = mean(xs), mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def terciles(values: Sequence[float]) -> tuple[float, float]:
    """The two cut points splitting ``values`` into thirds."""
    if not values:
        return (0.0, 0.0)
    ordered = sorted(values)
    return (
        ordered[len(ordered) // 3],
        ordered[2 * len(ordered) // 3],
    )
