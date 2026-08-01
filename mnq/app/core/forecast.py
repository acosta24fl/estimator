"""A 5-minute-ahead price projection fitted from the indicators already on screen.

This is **not a trading signal.** It is a small linear model over three features
you can already see, refit continuously against what actually happened, and
reported alongside its own measured skill.

How it works
------------
Three features are computed for every past 5-minute bar, each scaled to roughly
[-1, +1] against its *own* recent magnitude so it stays comparable across
volatility regimes:

* **Momentum** — the MACD histogram on the 5-minute series.
* **VWAP distance** — signed distance from session VWAP.
* **Daily structure** — the higher-high / lower-low read as a bias.

Those are regressed on the realised next-bar move by ridge:

    beta = (XtX + lambda*I)^-1 Xt y

and the projection is ``sum(beta_k * x_k)``. Coefficients are in points per
unit of feature, so each term *is* that feature's contribution in points, and
they sum exactly to the projection.

Why fitted rather than hand-weighted
------------------------------------
The earlier version asserted weights (0.5 / 0.3 / 0.2) and scaled by
volatility. That implicitly assumes each feature correlates almost perfectly
with the next move. For a predictor with correlation ``rho`` the MSE-optimal
coefficient is ``rho * sigma_y / sigma_x``; anything larger *provably increases*
error. Measured walk-forward on stored history, the hand-weighted version
scored -40% against a "no move" baseline while the fitted version scored +8%,
and the fitted sign of the VWAP term disagreed with the assumed one.

Fitting also fails safely: when the features carry no signal the coefficients
shrink toward zero and the projection degenerates to "no change", which is the
baseline. Guessed weights have no such guarantee.

There is no intercept: over five minutes the unconditional expected move is
essentially zero, and a fitted constant would bias every projection in one
direction for no reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..models import Bar
from . import structure as structure_mod
from .features import (
    average_range,
    clamp,
    close_to_close_moves,
    macd,
    percentile,
    session_vwap,
)

HORIZON_SECONDS = 300  # one 5-minute bar

#: Bars of completed 5-minute history before any feature is usable.
MIN_BARS = 40
#: Fitted (features, outcome) pairs required before the model will project.
MIN_FIT_SAMPLES = 200
#: Ridge penalty. Larger shrinks coefficients toward zero (toward "no move").
RIDGE_LAMBDA = 10.0
#: Projection is capped at this multiple of the volatility unit.
MAX_MOVE_SIGMA = 1.5

_VOL_PERIOD = 20
_SCALE_LOOKBACK = 200  # window for each feature's own magnitude
_BAND_QUANTILE = 0.68
_BAND_LOOKBACK = 60
_STRUCTURE_LOOKBACK = 2

FACTOR_LABELS = {
    "momentum": "Momentum (MACD)",
    "vwap": "VWAP Distance",
    "structure": "Daily Structure",
}
FACTOR_KEYS = ("momentum", "vwap", "structure")


@dataclass
class Factor:
    """One feature's contribution, in price points."""

    key: str
    label: str
    score: float  # normalised feature value, roughly [-1, 1]
    coefficient: float  # fitted points per unit score
    points: float  # score * coefficient (its share of the projection)
    detail: str = ""

    @property
    def direction(self) -> str:
        return "up" if self.points > 0 else ("down" if self.points < 0 else "flat")

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "score": round(self.score, 4),
            "coefficient": round(self.coefficient, 4),
            "points": round(self.points, 2),
            "direction": self.direction,
            "detail": self.detail,
        }


@dataclass
class Fit:
    ready: bool = False
    reason: str = ""
    samples: int = 0
    r2: float = 0.0  # in-sample; optimistic by construction
    coefficients: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "reason": self.reason,
            "samples": self.samples,
            "r2": round(self.r2, 4),
            "coefficients": {k: round(v, 4) for k, v in self.coefficients.items()},
        }


@dataclass
class Forecast:
    valid: bool = False
    reason: str = ""
    anchor_ts: int = 0
    target_ts: int = 0
    anchor_price: float = 0.0
    target: float = 0.0
    low: float = 0.0
    high: float = 0.0
    expected_move: float = 0.0
    volatility: float = 0.0
    band_half: float = 0.0
    direction: str = "flat"
    agreement: int = 0
    factors: list[Factor] = field(default_factory=list)
    fit: Fit = field(default_factory=Fit)

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
            "fit": self.fit.as_dict(),
        }


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def _trailing_rms(values: Sequence[float], index: int, lookback: int) -> float:
    """Root-mean-square of a feature over the bars up to ``index``.

    Each feature is divided by its own recent magnitude rather than by a shared
    volatility unit. That is the fix for the VWAP term: session-accumulated
    distance is on a far larger scale than a 5-minute bar range, so dividing it
    by that range drove ``tanh`` into saturation on about half of all bars and
    collapsed the feature into a bare sign bit.
    """
    start = max(0, index - lookback + 1)
    window = [v for v in values[start : index + 1] if v is not None]
    if not window:
        return 0.0
    return math.sqrt(sum(v * v for v in window) / len(window))


def build_features(
    bars_5m: Sequence[Bar], daily_bars: Sequence[Bar], session
) -> tuple[list[Bar], list[list[float]], list[str]]:
    """Per-bar feature rows for the completed 5-minute series.

    Returns (completed bars, feature rows aligned to them, structure labels).
    Every value at index ``i`` uses only information available at ``i``.
    """
    completed = [b for b in bars_5m if b.complete]
    if len(completed) < MIN_BARS:
        return completed, [], []

    closes = [b.close for b in completed]
    _, _, hist = macd(closes)
    vwap = session_vwap(completed, session)

    # Daily structure, as it was knowable on each day, mapped onto 5m bars.
    labels = _structure_per_bar(completed, daily_bars, session)

    raw_mom = [h if h is not None else 0.0 for h in hist]
    raw_vwap = [c - v for c, v in zip(closes, vwap)]

    rows: list[list[float]] = []
    for i in range(len(completed)):
        mom_scale = _trailing_rms(raw_mom, i, _SCALE_LOOKBACK)
        vwap_scale = _trailing_rms(raw_vwap, i, _SCALE_LOOKBACK)
        mom = math.tanh(raw_mom[i] / (2.0 * mom_scale)) if mom_scale > 0 else 0.0
        vwp = math.tanh(raw_vwap[i] / (2.0 * vwap_scale)) if vwap_scale > 0 else 0.0
        rows.append([mom, vwp, structure_mod.bias(labels[i])])
    return completed, rows, labels


def _structure_per_bar(
    completed: Sequence[Bar], daily_bars: Sequence[Bar], session
) -> list[str]:
    """Map each 5-minute bar to the daily structure known before its session."""
    if not daily_bars:
        return ["Range"] * len(completed)

    daily = list(daily_bars)
    labels = structure_mod.as_of_labels(daily, _STRUCTURE_LOOKBACK)
    daily_ts = [b.ts for b in daily]

    out: list[str] = []
    pointer = 0
    for bar in completed:
        session_start = session.start(bar.ts)
        # Advance to the last daily bar that closed before this session opened,
        # so today's still-forming daily bar never leaks in.
        while pointer + 1 < len(daily_ts) and daily_ts[pointer + 1] < session_start:
            pointer += 1
        out.append(labels[pointer] if daily_ts[pointer] < session_start else "Range")
    return out


# ---------------------------------------------------------------------------
# Ridge regression
# ---------------------------------------------------------------------------


def ridge_fit(X: Sequence[Sequence[float]], y: Sequence[float], lam: float) -> list[float]:
    """Solve (XtX + lam*I) beta = Xt y by Gaussian elimination.

    No intercept — see the module docstring. Returns zeros if the system is
    singular, which degrades the projection to "no move" rather than exploding.
    """
    if not X or not y:
        return []
    k = len(X[0])
    a = [[sum(X[t][i] * X[t][j] for t in range(len(X))) + (lam if i == j else 0.0)
          for j in range(k)] for i in range(k)]
    b = [sum(X[t][i] * y[t] for t in range(len(X))) for i in range(k)]

    for i in range(k):
        pivot = max(range(i, k), key=lambda r: abs(a[r][i]))
        if abs(a[pivot][i]) < 1e-12:
            return [0.0] * k
        a[i], a[pivot] = a[pivot], a[i]
        b[i], b[pivot] = b[pivot], b[i]
        for r in range(k):
            if r == i:
                continue
            factor = a[r][i] / a[i][i]
            for c in range(i, k):
                a[r][c] -= factor * a[i][c]
            b[r] -= factor * b[i]
    return [b[i] / a[i][i] for i in range(k)]


def _r_squared(X, y, beta) -> float:
    if not y:
        return 0.0
    mean_y = sum(y) / len(y)
    ss_tot = sum((v - mean_y) ** 2 for v in y)
    ss_res = sum((y[t] - sum(beta[i] * X[t][i] for i in range(len(beta)))) ** 2
                 for t in range(len(y)))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------


def compute_forecast(
    bars_5m: Sequence[Bar],
    daily_bars: Sequence[Bar],
    session,
    horizon_seconds: int = HORIZON_SECONDS,
    strength: float = 1.0,
    ridge_lambda: float = RIDGE_LAMBDA,
    min_fit_samples: int = MIN_FIT_SAMPLES,
) -> Forecast:
    """Project the close of the next 5-minute bar.

    Only completed bars are used, and the model is fitted only on pairs whose
    outcome is already known, so the projection never sees the bar it predicts.
    """
    completed, rows, labels = build_features(bars_5m, daily_bars, session)
    if len(completed) < MIN_BARS:
        return Forecast(reason=f"needs {MIN_BARS} completed 5m bars, has {len(completed)}")

    last = completed[-1]
    anchor_price = last.close
    anchor_ts = last.ts + horizon_seconds
    volatility = average_range(completed, _VOL_PERIOD)
    if volatility <= 0:
        return Forecast(reason="no measurable volatility")

    # Training pairs: features at i, realised move from i to i+1. The final bar
    # has no outcome yet, so it is the one we predict rather than train on.
    X = rows[MIN_BARS:-1]
    y = [completed[i + 1].close - completed[i].close for i in range(MIN_BARS, len(completed) - 1)]

    if len(y) < min_fit_samples:
        return Forecast(
            reason=f"learning: {len(y)}/{min_fit_samples} fitted samples",
            fit=Fit(ready=False, samples=len(y),
                    reason=f"needs {min_fit_samples} samples, has {len(y)}"),
        )

    beta = ridge_fit(X, y, ridge_lambda)
    fit = Fit(
        ready=True,
        samples=len(y),
        r2=_r_squared(X, y, beta),
        coefficients={k: beta[i] for i, k in enumerate(FACTOR_KEYS)},
    )

    scores = rows[-1]
    raw_move = sum(beta[i] * scores[i] for i in range(len(beta))) * strength
    expected_move = clamp(raw_move, -MAX_MOVE_SIGMA * volatility, MAX_MOVE_SIGMA * volatility)
    # Preserve the identity "contributions sum to the projection" when clamped.
    shrink = (expected_move / raw_move) if raw_move else 1.0

    factors = [
        Factor(
            key=key,
            label=FACTOR_LABELS[key],
            score=scores[i],
            coefficient=beta[i],
            points=beta[i] * scores[i] * strength * shrink,
            detail=_detail(key, scores[i], beta[i], labels[-1] if labels else ""),
        )
        for i, key in enumerate(FACTOR_KEYS)
    ]

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
        fit=fit,
    )


def _detail(key: str, score: float, coefficient: float, structure_label: str) -> str:
    if key == "structure":
        return f"{structure_label or 'unknown'} (fitted {coefficient:+.1f} pts/unit)"
    return f"score {score:+.2f} x fitted {coefficient:+.1f} pts/unit"


def _band_half_width(bars: Sequence[Bar]) -> float:
    """Typical absolute 5-minute move, from recent history."""
    moves = close_to_close_moves(list(bars)[-_BAND_LOOKBACK:])
    if not moves:
        return 0.0
    return percentile(moves, _BAND_QUANTILE)
