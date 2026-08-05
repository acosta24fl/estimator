"""Predicting the *next candle* on any timeframe.

The existing projection answers "where will the close be", which is the one
part of a candle that is close to unpredictable. A candle has four numbers, and
they are not equally hard:

===========  ===================================================================
open         Essentially known. The next bar opens where this one closed, up to
             a gap that is zero for contiguous intraday bars and only matters
             across a session break. Measured, not assumed.
high / low   **Genuinely predictable.** Not their direction — their *size*.
             Excursion scales with volatility, and volatility clusters hard
             (lag-1 autocorrelation of variance is typically 0.2-0.4 on real
             futures). This is the part worth modelling.
close        The hard one. Direction at short horizons is near-random; all the
             drift model can honestly say is "somewhere in this range".
===========  ===================================================================

Method: standardised excursion quantiles
----------------------------------------
For every completed bar, measure how far it travelled *in units of the
volatility that was already knowable before it opened*::

    u_i = (high_i  - open_i) / sigma_{i-1}      upward excursion
    d_i = (open_i  - low_i)  / sigma_{i-1}      downward excursion
    c_i = (close_i - open_i) / sigma_{i-1}      net move

``sigma`` is the EWMA of Garman-Klass variance, whose entry ``i-1`` is by
construction the forecast for bar ``i``, so no future information enters.

Those standardised shapes are stable across volatility regimes — a quiet bar
and a violent bar look alike once divided by their own sigma — so their
empirical quantiles transfer. The prediction for the next bar is then::

    open_hat  = last close + gap
    high_hat  = open_hat + Q(u, q) * sigma_now
    low_hat   = open_hat - Q(d, q) * sigma_now
    close_hat = open_hat + drift

Quantiles come from the empirical distribution rather than a normal
assumption, because excursions are strongly right-skewed and a Gaussian band
under-covers exactly in the tails that matter.

Why quantiles and not a mean
----------------------------
"The high will be X" is a claim no model can keep. "The high stays below X
about 80% of the time" is one that can be checked, and :func:`evaluate` checks
it walk-forward: fit on everything before bar ``i``, predict bar ``i``, score.
If the reported coverage is not near the requested quantile the model is
miscalibrated and says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..models import Bar
from .features import ewma_volatility, percentile

#: Default band. 0.80 means "the high should stay below this ~80% of the time".
DEFAULT_QUANTILE = 0.80
#: Rolling window of standardised bars the quantiles are read from. Bounded so
#: the estimate tracks the current regime instead of averaging over all history.
DEFAULT_LOOKBACK = 500
#: Standardised bars needed before any prediction is offered.
MIN_SAMPLES = 60
EWMA_LAMBDA = 0.94


@dataclass
class Shape:
    """One bar's travel, in units of the volatility known before it opened."""

    ts: int
    up: float  # (high - open) / sigma
    down: float  # (open - low) / sigma
    net: float  # (close - open) / sigma
    sigma: float


@dataclass
class CandleForecast:
    """Predicted OHLC envelope for the next bar of some timeframe."""

    valid: bool = False
    reason: str = ""
    timeframe: str = ""
    ts: int = 0  # start time of the bar being predicted
    quantile: float = DEFAULT_QUANTILE
    samples: int = 0
    sigma: float = 0.0
    gap: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    #: Median (not quantile) excursions — the typical bar, for reference.
    typical_high: float = 0.0
    typical_low: float = 0.0
    up_multiple: float = 0.0
    down_multiple: float = 0.0
    drift: float = 0.0

    @property
    def predicted_range(self) -> float:
        return self.high - self.low

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "timeframe": self.timeframe,
            "ts": self.ts,
            "quantile": self.quantile,
            "samples": self.samples,
            "sigma": round(self.sigma, 2),
            "gap": round(self.gap, 2),
            "open": round(self.open, 2),
            "high": round(self.high, 2),
            "low": round(self.low, 2),
            "close": round(self.close, 2),
            "typical_high": round(self.typical_high, 2),
            "typical_low": round(self.typical_low, 2),
            "range": round(self.predicted_range, 2),
            "up_multiple": round(self.up_multiple, 3),
            "down_multiple": round(self.down_multiple, 3),
            "drift": round(self.drift, 2),
        }


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------


def shapes(bars: Sequence[Bar], lam: float = EWMA_LAMBDA) -> list[Shape]:
    """Standardise each completed bar by the volatility known before it.

    The first bar has no prior sigma and is dropped rather than standardised
    against itself, which would leak its own range into its own scale.
    """
    completed = [b for b in bars if b.complete]
    if len(completed) < 2:
        return []
    sigma = ewma_volatility(completed, lam)

    out: list[Shape] = []
    for i in range(1, len(completed)):
        prior = sigma[i - 1]  # formed after bar i-1, i.e. the forecast for bar i
        if prior <= 0:
            continue
        bar = completed[i]
        out.append(
            Shape(
                ts=bar.ts,
                up=(bar.high - bar.open) / prior,
                down=(bar.open - bar.low) / prior,
                net=(bar.close - bar.open) / prior,
                sigma=prior,
            )
        )
    return out


def gap_estimate(bars: Sequence[Bar]) -> float:
    """Median ``open - previous close``, in points.

    Zero for contiguous intraday bars; non-zero across session breaks, which is
    where the daily timeframe lives. Median rather than mean so one limit-up
    session does not move it.
    """
    completed = [b for b in bars if b.complete]
    if len(completed) < 2:
        return 0.0
    gaps = [completed[i].open - completed[i - 1].close for i in range(1, len(completed))]
    return percentile(sorted(gaps), 0.5)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict(
    bars: Sequence[Bar],
    drift: float = 0.0,
    quantile: float = DEFAULT_QUANTILE,
    lookback: int = DEFAULT_LOOKBACK,
    timeframe: str = "",
    step_seconds: int = 0,
    lam: float = EWMA_LAMBDA,
) -> CandleForecast:
    """Predict the next bar's OHLC envelope.

    ``drift`` is the projected close-minus-open in points, normally supplied by
    the ridge model. It defaults to zero — "no view" — because at these horizons
    that is the honest prior and the envelope is the part carrying information.
    """
    history = shapes(bars, lam)
    if len(history) < MIN_SAMPLES:
        return CandleForecast(
            timeframe=timeframe,
            reason=f"needs {MIN_SAMPLES} standardised bars, has {len(history)}",
            samples=len(history),
        )

    completed = [b for b in bars if b.complete]
    window = history[-lookback:]
    sigma_now = ewma_volatility(completed, lam)[-1]
    if sigma_now <= 0:
        return CandleForecast(timeframe=timeframe, reason="no measurable volatility")

    ups = sorted(s.up for s in window)
    downs = sorted(s.down for s in window)

    gap = gap_estimate(completed)
    open_hat = completed[-1].close + gap
    up_multiple = percentile(ups, quantile)
    down_multiple = percentile(downs, quantile)

    high = open_hat + up_multiple * sigma_now
    low = open_hat - down_multiple * sigma_now
    close = open_hat + drift

    return CandleForecast(
        valid=True,
        timeframe=timeframe,
        ts=completed[-1].ts + (step_seconds or _infer_step(completed)),
        quantile=quantile,
        samples=len(window),
        sigma=sigma_now,
        gap=gap,
        open=open_hat,
        high=high,
        # A drift larger than the envelope would draw a close outside its own
        # band. Clamp rather than widen: the band is the measured, calibrated
        # quantity and the drift is the speculative one.
        low=low,
        close=min(max(close, low), high),
        typical_high=open_hat + percentile(ups, 0.5) * sigma_now,
        typical_low=open_hat - percentile(downs, 0.5) * sigma_now,
        up_multiple=up_multiple,
        down_multiple=down_multiple,
        drift=drift,
    )


def _infer_step(bars: Sequence[Bar]) -> int:
    """Most common spacing between bar starts, for timing the predicted bar."""
    if len(bars) < 2:
        return 0
    diffs = [bars[i].ts - bars[i - 1].ts for i in range(1, len(bars))]
    return min(diffs) if diffs else 0


# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------


@dataclass
class CandleAccuracy:
    """Out-of-sample scorecard for the envelope."""

    bars: int = 0
    quantile: float = DEFAULT_QUANTILE
    high_coverage: float = 0.0  # fraction of bars whose high stayed below high_hat
    low_coverage: float = 0.0
    contained: float = 0.0  # fraction where both held
    range_mae: float = 0.0  # mean |predicted range - actual range|, points
    range_mae_naive: float = 0.0  # same, predicting "range of the last bar"
    range_mae_flat: float = 0.0  # same, predicting the average range
    range_skill: float = 0.0  # 1 - mae/mae_flat; >0 means it beats the average
    naive_skill: float = 0.0  # 1 - mae/mae_naive
    close_mae: float = 0.0
    close_mae_baseline: float = 0.0  # predicting "closes where it opened"
    close_skill: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "bars": self.bars,
            "quantile": self.quantile,
            "high_coverage": round(self.high_coverage, 4),
            "low_coverage": round(self.low_coverage, 4),
            "contained": round(self.contained, 4),
            "range_mae": round(self.range_mae, 3),
            "range_mae_naive": round(self.range_mae_naive, 3),
            "range_mae_flat": round(self.range_mae_flat, 3),
            "range_skill": round(self.range_skill, 4),
            "naive_skill": round(self.naive_skill, 4),
            "close_mae": round(self.close_mae, 3),
            "close_mae_baseline": round(self.close_mae_baseline, 3),
            "close_skill": round(self.close_skill, 4),
        }


def evaluate(
    bars: Sequence[Bar],
    quantile: float = DEFAULT_QUANTILE,
    lookback: int = DEFAULT_LOOKBACK,
    min_train: int = MIN_SAMPLES,
    lam: float = EWMA_LAMBDA,
) -> CandleAccuracy:
    """Score the envelope walk-forward: fit on the past, predict, compare.

    At each step the quantiles come only from bars that had already closed, so
    this is genuine out-of-sample coverage and not a description of the fit.

    Coverage is the number to read first. A band claiming 0.80 that contains
    0.55 of actual highs is not conservative, it is wrong, and every stop
    placed off it would be too tight.
    """
    completed = [b for b in bars if b.complete]
    history = shapes(completed, lam)
    if len(history) <= min_train:
        return CandleAccuracy(quantile=quantile)

    # shapes() drops the first bar, so shape j describes completed[j + 1].
    offset = len(completed) - len(history)

    highs_ok = lows_ok = both_ok = 0
    range_err = naive_err = flat_err = close_err = close_base = 0.0
    scored = 0

    for j in range(min_train, len(history)):
        actual = completed[j + offset]
        prior = completed[j + offset - 1]
        window = history[max(0, j - lookback) : j]  # strictly before this bar
        if not window:
            continue

        sigma = history[j].sigma  # the estimate formed before this bar opened
        ups = sorted(s.up for s in window)
        downs = sorted(s.down for s in window)

        open_hat = prior.close  # gap is measured live; here the actual open is known
        high_hat = open_hat + percentile(ups, quantile) * sigma
        low_hat = open_hat - percentile(downs, quantile) * sigma

        hi = actual.high <= high_hat
        lo = actual.low >= low_hat
        highs_ok += hi
        lows_ok += lo
        both_ok += hi and lo

        actual_range = actual.high - actual.low
        range_err += abs((high_hat - low_hat) - actual_range)
        naive_err += abs((prior.high - prior.low) - actual_range)
        flat_err += abs(_mean_range(completed, j + offset, lookback) - actual_range)

        # With no drift the close prediction is the open; the baseline is the
        # same thing, so close_skill is 0 by construction until a drift model
        # is supplied. Kept so a drift can be scored against the honest prior.
        close_err += abs(open_hat - actual.close)
        close_base += abs(prior.close - actual.close)
        scored += 1

    if not scored:
        return CandleAccuracy(quantile=quantile)

    mae = range_err / scored
    mae_naive = naive_err / scored
    mae_flat = flat_err / scored
    c_mae = close_err / scored
    c_base = close_base / scored
    return CandleAccuracy(
        bars=scored,
        quantile=quantile,
        high_coverage=highs_ok / scored,
        low_coverage=lows_ok / scored,
        contained=both_ok / scored,
        range_mae=mae,
        range_mae_naive=mae_naive,
        range_mae_flat=mae_flat,
        range_skill=1.0 - mae / mae_flat if mae_flat > 0 else 0.0,
        naive_skill=1.0 - mae / mae_naive if mae_naive > 0 else 0.0,
        close_mae=c_mae,
        close_mae_baseline=c_base,
        close_skill=1.0 - c_mae / c_base if c_base > 0 else 0.0,
    )


def _mean_range(bars: Sequence[Bar], index: int, lookback: int) -> float:
    window = bars[max(0, index - lookback) : index]
    if not window:
        return 0.0
    return sum(b.high - b.low for b in window) / len(window)


def calibrate(
    bars: Sequence[Bar],
    candidates: Sequence[float] = (0.5, 0.6, 0.68, 0.75, 0.8, 0.85, 0.9, 0.95),
    lookback: int = DEFAULT_LOOKBACK,
    lam: float = EWMA_LAMBDA,
) -> list[CandleAccuracy]:
    """Score a sweep of quantiles so miscalibration is visible as a table.

    Perfect calibration is ``contained ~= q^2``-ish only if the two sides were
    independent, which they are not, so read each side's coverage against ``q``
    directly rather than the joint number.
    """
    return [evaluate(bars, quantile=q, lookback=lookback, lam=lam) for q in candidates]
