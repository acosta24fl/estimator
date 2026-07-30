"""A 5-minute-ahead price projection built from the indicators already on screen.

This is a **transparent heuristic, not a trained model and not a trading
signal.** It exists so the dashboard can say "given what these three features
currently read, here is roughly where the next five minutes would land, and by
how much they disagree." Every point of projected movement is attributed to a
named factor, and every projection is scored against what actually happened
(see :mod:`app.core.prediction_log`) so its real accuracy is visible rather
than assumed.

How it works
------------
Three features each produce a score in [-1, +1]:

* **Momentum** — the MACD histogram on the 5-minute series, normalised by
  volatility. Positive histogram => continuation upward.
* **VWAP pull** — distance from session VWAP, normalised, with the sign
  flipped: stretched far above VWAP implies a pull back down.
* **Daily structure** — the higher-high / lower-low read, as a directional bias.

They are combined with fixed weights into a single drift score, multiplied by a
volatility unit (the average 5-minute bar range — the "bar size" feature), and
capped so the projection can never run away from price.

The band around the projection is empirical: the 68th percentile of recent
absolute 5-minute close-to-close moves. It is a *typical range*, not a
confidence interval in any rigorous sense.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..models import Bar
from .features import (
    average_range,
    clamp,
    close_to_close_moves,
    macd,
    percentile,
    session_vwap,
)

HORIZON_SECONDS = 300  # one 5-minute bar

#: How much each feature contributes to the drift score. Sums to 1.
WEIGHTS = {"momentum": 0.5, "vwap": 0.3, "structure": 0.2}

#: Projection is capped at this multiple of the volatility unit.
MAX_MOVE_SIGMA = 1.5
#: Bars of completed 5-minute history required before projecting.
MIN_BARS = 40
_VOL_PERIOD = 20
_BAND_QUANTILE = 0.68
_BAND_LOOKBACK = 60

_STRUCTURE_BIAS = {
    "Uptrend": 1.0,
    "Range (bullish tilt)": 0.5,
    "Range": 0.0,
    "Range (bearish tilt)": -0.5,
    "Downtrend": -1.0,
}


@dataclass
class Factor:
    """One feature's contribution, in price points so it can be read directly."""

    key: str
    label: str
    score: float  # -1..+1
    weight: float
    points: float  # signed contribution to the projected move
    detail: str = ""

    @property
    def direction(self) -> str:
        return "up" if self.points > 0 else ("down" if self.points < 0 else "flat")

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "score": round(self.score, 4),
            "weight": self.weight,
            "points": round(self.points, 2),
            "direction": self.direction,
            "detail": self.detail,
        }


@dataclass
class Forecast:
    valid: bool = False
    reason: str = ""
    anchor_ts: int = 0  # start of the 5m bar being predicted
    target_ts: int = 0  # when that bar closes
    anchor_price: float = 0.0
    target: float = 0.0
    low: float = 0.0
    high: float = 0.0
    expected_move: float = 0.0
    volatility: float = 0.0
    band_half: float = 0.0
    direction: str = "flat"
    agreement: int = 0  # factors agreeing with the net direction
    factors: list[Factor] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "anchor_ts": self.anchor_ts,
            "target_ts": self.target_ts,
            "anchor_price": round(self.anchor_price, 2),
            "target": round(self.target, 2),
            "low": round(self.low, 2),
            "high": round(self.high, 2),
            "expected_move": round(self.expected_move, 2),
            "volatility": round(self.volatility, 2),
            "band_half": round(self.band_half, 2),
            "direction": self.direction,
            "agreement": self.agreement,
            "factors": [f.as_dict() for f in self.factors],
        }


def compute_forecast(
    bars_5m: Sequence[Bar],
    daily_structure: str,
    session,
    horizon_seconds: int = HORIZON_SECONDS,
    strength: float = 1.0,
) -> Forecast:
    """Project the close of the next 5-minute bar.

    Only *completed* bars are used, so the projection never peeks at the bar it
    is predicting.

    ``strength`` scales the whole projected move. Five-minute returns are close
    to random, so a drift signal with no real edge only adds error; damping it
    (``MNQ_FORECAST_STRENGTH`` below 1.0) shrinks the projection toward "no
    change", and 0.0 turns it into a pure no-move forecast. Measure with
    ``python -m app.backtest`` before choosing a value.
    """
    completed = [b for b in bars_5m if b.complete]
    if len(completed) < MIN_BARS:
        return Forecast(
            reason=f"needs {MIN_BARS} completed 5m bars, has {len(completed)}"
        )

    last = completed[-1]
    anchor_price = last.close
    anchor_ts = last.ts + horizon_seconds  # the bar now forming
    volatility = average_range(completed, _VOL_PERIOD)
    if volatility <= 0:
        return Forecast(reason="no measurable volatility")

    factors = [
        _momentum_factor(completed, volatility),
        _vwap_factor(completed, session, volatility),
        _structure_factor(daily_structure),
    ]

    drift = sum(f.score * f.weight for f in factors) * strength
    expected_move = clamp(
        drift * volatility, -MAX_MOVE_SIGMA * volatility, MAX_MOVE_SIGMA * volatility
    )

    # Re-express each factor in points, sharing out the capped total so the
    # parts always add up to the whole the user sees.
    raw_total = sum(f.score * f.weight for f in factors)
    for factor in factors:
        share = (factor.score * factor.weight / raw_total) if raw_total else 0.0
        factor.points = share * expected_move

    band_half = _band_half_width(completed)
    target = anchor_price + expected_move
    direction = "up" if expected_move > 0 else ("down" if expected_move < 0 else "flat")
    agreement = sum(1 for f in factors if f.direction == direction and f.points)

    return Forecast(
        valid=True,
        anchor_ts=anchor_ts,
        target_ts=anchor_ts + horizon_seconds,
        anchor_price=anchor_price,
        target=target,
        low=target - band_half,
        high=target + band_half,
        expected_move=expected_move,
        volatility=volatility,
        band_half=band_half,
        direction=direction,
        agreement=agreement,
        factors=factors,
    )


# -- individual features ----------------------------------------------------


def _momentum_factor(bars: Sequence[Bar], volatility: float) -> Factor:
    """MACD histogram, squashed into [-1, 1] against the volatility unit."""
    _, _, hist = macd([b.close for b in bars])
    latest = next((h for h in reversed(hist) if h is not None), None)
    if latest is None:
        return Factor("momentum", "Momentum (MACD)", 0.0, WEIGHTS["momentum"], 0.0,
                      "MACD still warming up")
    score = math.tanh(latest / (0.5 * volatility)) if volatility else 0.0
    return Factor(
        "momentum",
        "Momentum (MACD)",
        score,
        WEIGHTS["momentum"],
        0.0,
        f"histogram {latest:+.2f}",
    )


def _vwap_factor(bars: Sequence[Bar], session, volatility: float) -> Factor:
    """Distance from session VWAP, as a pull back toward it."""
    vwap = session_vwap(bars, session)
    if not vwap:
        return Factor("vwap", "VWAP Pull", 0.0, WEIGHTS["vwap"], 0.0, "no VWAP")
    distance = bars[-1].close - vwap[-1]
    # Flipped sign: stretched above VWAP pulls down, and vice versa.
    score = -math.tanh(distance / (2.0 * volatility)) if volatility else 0.0
    return Factor(
        "vwap",
        "VWAP Pull",
        score,
        WEIGHTS["vwap"],
        0.0,
        f"{distance:+.2f} pts from VWAP",
    )


def _structure_factor(structure: str) -> Factor:
    bias = _STRUCTURE_BIAS.get(structure, 0.0)
    return Factor(
        "structure",
        "Daily Structure",
        bias,
        WEIGHTS["structure"],
        0.0,
        structure or "unknown",
    )


def _band_half_width(bars: Sequence[Bar]) -> float:
    """Typical absolute 5-minute move, from recent history."""
    moves = close_to_close_moves(list(bars)[-_BAND_LOOKBACK:])
    if not moves:
        return 0.0
    return percentile(moves, _BAND_QUANTILE)
