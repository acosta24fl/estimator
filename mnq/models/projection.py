"""Trend direction and expected move size.

The dashboard needs to answer "which way, and by how many points". The model
does not answer that directly, and the obvious shortcut is wrong in a specific
way worth spelling out.

``SharedRegressor`` predicts the forward percent return by minimising squared
error. Financial returns are mostly noise, so the loss-minimising prediction is
heavily shrunk toward the mean - a regressor that is genuinely informative will
still output tiny numbers, because outputting the true magnitude would be
punished on the many occasions the move does not happen. Converting that
straight to points produces projections far smaller than any move worth
trading, and it would be wrong on purpose.

What this module does instead: bucket the *out-of-sample* predictions by model
confidence, and measure what price actually did afterwards in each bucket. A
live score is then matched to its bucket and reported with that bucket's
realised distribution - a median and an interquartile range, not a false point
estimate.

That makes the projection empirical rather than theoretical. It also makes it
honest about uncertainty: if the 25th percentile of a bucket is negative, the
dashboard says so instead of drawing a confident arrow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..labeling import forward_return

#: Confidence edges. Deliberately coarse: finer buckets look more precise while
#: putting too few observations in each to measure anything.
DEFAULT_EDGES: tuple[float, ...] = (0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 1.01)

#: A bucket below this many observations is not reported as calibrated.
MIN_BUCKET = 40


@dataclass(frozen=True)
class Projection:
    """Where price is expected to go, and how sure that is."""

    direction: str          # "up" | "down" | "flat"
    confidence: float       # the winning side's meta probability
    expected_points: float  # median realised move for this confidence bucket
    low_points: float       # 25th percentile
    high_points: float      # 75th percentile
    hit_rate: float         # share of analogues that moved the projected way
    n_analogues: int
    horizon_bars: int
    basis: str              # "calibrated" | "model" | "unavailable"
    note: str = ""
    model_points: float = 0.0   # the raw regressor number, for comparison

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Calibration:
    """Realised move distributions, bucketed by model confidence."""

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    horizon_bars: int = 12
    edges: tuple[float, ...] = DEFAULT_EDGES

    @property
    def available(self) -> bool:
        return not self.table.empty

    def to_records(self) -> list[dict[str, Any]]:
        return [] if self.table.empty else self.table.reset_index().to_dict("records")


def build_calibration(
    predictions: pd.DataFrame,
    matrix: pd.DataFrame,
    cfg: Config,
    edges: tuple[float, ...] = DEFAULT_EDGES,
) -> Calibration:
    """Measure what price did after each confidence level, out of sample.

    ``predictions`` is the walk-forward frame (``long_p_meta`` /
    ``short_p_meta``); ``matrix`` supplies prices. Only bars present in both are
    used, so this inherits the walk-forward's out-of-sample guarantee.
    """
    horizon = cfg.labels.fwd_return_bars
    if predictions is None or predictions.empty or matrix is None or matrix.empty:
        return Calibration(pd.DataFrame(), horizon, edges)

    idx = predictions.index.intersection(matrix.index)
    if len(idx) < MIN_BUCKET:
        return Calibration(pd.DataFrame(), horizon, edges)

    # Percent return converted to points at the price it was measured from, so
    # a 0.5% move means more points at 20,000 than at 15,000 - which is the
    # whole reason not to hard-code a point target across a long history.
    pct = forward_return(matrix, horizon).reindex(idx)
    entry = matrix["open"].shift(-1).reindex(idx)
    points = pct / 100.0 * entry

    rows = []
    for side, sign in (("long", 1.0), ("short", -1.0)):
        col = f"{side}_p_meta"
        if col not in predictions:
            continue
        conf = predictions[col].reindex(idx)
        # Signed so that "favourable" is positive for both directions: a short
        # is right when price falls.
        realised = sign * points

        frame = pd.DataFrame({"conf": conf, "move": realised}).dropna()
        if frame.empty:
            continue

        bucket = pd.cut(frame["conf"], bins=list(edges), right=False)
        for interval, grp in frame.groupby(bucket, observed=True):
            if len(grp) < MIN_BUCKET:
                continue
            moves = grp["move"]
            rows.append(
                {
                    "side": side,
                    "lo": float(interval.left),
                    "hi": float(interval.right),
                    "n": int(len(grp)),
                    "median_points": float(moves.median()),
                    "p25_points": float(moves.quantile(0.25)),
                    "p75_points": float(moves.quantile(0.75)),
                    "mean_points": float(moves.mean()),
                    "hit_rate": float((moves > 0).mean()),
                }
            )

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.set_index(["side", "lo"]).sort_index()
    return Calibration(table, horizon, edges)


def project(
    score: dict[str, Any] | None,
    calibration: Calibration | None,
    cfg: Config,
) -> Projection:
    """Turn one live score into a direction and an expected move.

    Falls back to the raw regressor when no calibration bucket covers the
    current confidence, and says which basis it used either way.
    """
    horizon = cfg.labels.fwd_return_bars
    if not score:
        return Projection(
            "flat", 0.0, 0.0, 0.0, 0.0, 0.0, 0, horizon, "unavailable",
            "no live score yet - the model needs a warm-up window of bars",
        )

    p_long = float(score.get("p_long", 0.0))
    p_short = float(score.get("p_short", 0.0))
    close = float(score.get("close", 0.0))

    if p_long >= p_short:
        side, confidence, direction = "long", p_long, "up"
    else:
        side, confidence, direction = "short", p_short, "down"

    # The raw model number, kept for comparison rather than headline use.
    model_points = float(score.get("fwd_pred", 0.0)) / 100.0 * close
    if side == "short":
        model_points = -model_points

    gate = cfg.trade.min_probability
    below_gate = confidence < gate
    if below_gate:
        direction = "flat"

    # The gate is the more important fact whenever it applies: "no trade,
    # confidence too low" answers the reader's question, while an explanation
    # about regressor shrinkage does not.
    gate_note = (
        f"confidence {confidence:.2f} is below the {gate:.2f} entry gate - "
        f"no trade, direction shown for context only"
    )

    row = _lookup(calibration, side, confidence)
    if row is None:
        fallback = (
            "no calibration bucket covers this confidence; showing the raw "
            "regressor output, which is shrunk toward zero by design"
        )
        return Projection(
            direction, confidence,
            abs(model_points), 0.0, 0.0, 0.0, 0, horizon, "model",
            f"{gate_note}. {fallback}" if below_gate else fallback,
            model_points,
        )

    return Projection(
        direction=direction,
        confidence=confidence,
        expected_points=float(row["median_points"]),
        low_points=float(row["p25_points"]),
        high_points=float(row["p75_points"]),
        hit_rate=float(row["hit_rate"]),
        n_analogues=int(row["n"]),
        horizon_bars=horizon,
        basis="calibrated",
        note=_describe(row, direction, gate, confidence),
        model_points=model_points,
    )


def _lookup(calibration: Calibration | None, side: str, confidence: float):
    if calibration is None or not calibration.available:
        return None
    try:
        block = calibration.table.loc[side]
    except KeyError:
        return None
    for lo, row in block.iterrows():
        if lo <= confidence < row["hi"]:
            return row
    return None


def _describe(row, direction: str, gate: float, confidence: float) -> str:
    n, hit = int(row["n"]), float(row["hit_rate"])
    lo, hi = float(row["p25_points"]), float(row["p75_points"])

    if direction == "flat":
        return (
            f"confidence {confidence:.2f} is below the {gate:.2f} entry gate - "
            f"no trade, direction shown for context only"
        )
    if lo < 0:
        return (
            f"{n} historical analogues; the middle half ran {lo:+.0f} to "
            f"{hi:+.0f} points, so the range still spans a losing outcome"
        )
    return (
        f"{n} historical analogues at this confidence moved favourably "
        f"{hit:.0%} of the time"
    )
