"""Draws the 5-minute projection and its track record.

What you see on the chart:

* a bright **ray** leaving the last price and pointing to where the next
  5-minute bar is projected to close, with the target price labelled on the
  axis,
* a shaded **cone** (dashed upper/lower lines) showing the typical range around
  that target — the projection is a centre of gravity, not a promise,
* a dotted **track** of past projections laid over the candles they predicted,
  so the model's history is visible next to what actually happened.

The metrics panel breaks the projected move into the exact points contributed
by each feature, and reports measured accuracy against a "price doesn't move"
baseline. If the model is not beating that baseline, it says so.
"""

from __future__ import annotations

from ..core.forecast import compute_forecast
from . import register
from .base import (
    PANE_PRICE,
    Indicator,
    IndicatorContext,
    IndicatorResult,
    RenderSpec,
    SeriesSpec,
    Stat,
)

_UP = "#2e9e6b"
_DOWN = "#d1495b"
_PROJECTION = "#5ad2f4"
_TRACK = "#b07cd8"

#: The ray is only legible when 5 minutes spans a meaningful part of a bar.
_MAX_TF_SECONDS = 900


@register
class FiveMinuteProjection(Indicator):
    key = "projection"
    name = "5-Minute Projection"
    order = 5
    description = (
        "Heuristic projection of the next 5-minute close from momentum, VWAP "
        "pull and daily structure. Not a trading signal."
    )
    params = {"accuracy_window": 100}

    render = RenderSpec(
        pane=PANE_PRICE,
        series=(
            SeriesSpec(
                key="projection",
                label="5m Projection",
                type="line",
                color=_PROJECTION,
                line_width=3,
                price_line=True,
            ),
            SeriesSpec(
                key="high",
                label="Projection High",
                type="line",
                color=_PROJECTION,
                line_width=1,
                line_style=2,
            ),
            SeriesSpec(
                key="low",
                label="Projection Low",
                type="line",
                color=_PROJECTION,
                line_width=1,
                line_style=2,
            ),
            SeriesSpec(
                key="track",
                label="Past Projections",
                type="line",
                color=_TRACK,
                line_width=1,
                line_style=1,
            ),
        ),
    )

    def compute(self, ctx: IndicatorContext) -> IndicatorResult:
        result = IndicatorResult(
            series={"projection": [], "high": [], "low": [], "track": []}
        )

        bars_5m = list(ctx.bars_5m)
        if not bars_5m:
            result.stats = [Stat("projection_state", "Projection", "No 5m data")]
            return result

        structure = _structure_from(ctx)
        forecast = compute_forecast(
            bars_5m, structure, ctx.session, strength=ctx.settings.forecast_strength
        )

        if not forecast.valid:
            result.stats = [
                Stat(
                    "projection_state",
                    "Projection",
                    "Warming up",
                    hint=forecast.reason,
                )
            ]
            result.stats += self._accuracy_stats(ctx, bars_5m)
            return result

        # The ray spans from the last completed close to the projected close.
        if ctx.timeframe.nominal_seconds <= _MAX_TF_SECONDS:
            start_ts = forecast.anchor_ts
            end_ts = forecast.target_ts
            result.series["projection"] = [
                {"time": start_ts, "value": round(forecast.anchor_price, 2)},
                {"time": end_ts, "value": round(forecast.target, 2)},
            ]
            result.series["high"] = [
                {"time": start_ts, "value": round(forecast.anchor_price, 2)},
                {"time": end_ts, "value": round(forecast.high, 2)},
            ]
            result.series["low"] = [
                {"time": start_ts, "value": round(forecast.anchor_price, 2)},
                {"time": end_ts, "value": round(forecast.low, 2)},
            ]

        result.series["track"] = self._track(ctx, bars_5m)
        result.stats = self._forecast_stats(forecast) + self._accuracy_stats(ctx, bars_5m)
        return result

    # -- pieces ------------------------------------------------------------

    @staticmethod
    def _track(ctx: IndicatorContext, bars_5m: list) -> list[dict]:
        """Past projections, plotted on the bar each one predicted."""
        if ctx.predictions is None:
            return []
        return [
            {"time": s.prediction.anchor_ts, "value": round(s.prediction.target, 2)}
            for s in ctx.predictions.scored(bars_5m)
        ]

    @staticmethod
    def _forecast_stats(forecast) -> list[Stat]:
        tone = {"up": "up", "down": "down"}.get(forecast.direction, "neutral")
        arrow = {"up": "Up", "down": "Down"}.get(forecast.direction, "Flat")

        stats = [
            Stat(
                "projection_target",
                "Projected Close (5m)",
                round(forecast.target, 2),
                tone=tone,
                hint="Where the forming 5-minute bar is projected to close.",
            ),
            Stat(
                "projection_move",
                "Expected Move",
                round(forecast.expected_move, 2),
                unit="pts",
                tone=tone,
                signed=True,
            ),
            Stat(
                "projection_range",
                "Likely Range",
                f"{forecast.low:,.2f} – {forecast.high:,.2f}",
                tone="neutral",
                hint="Typical 5-minute travel (68th percentile of recent moves).",
            ),
            Stat(
                "projection_direction",
                "Direction",
                arrow,
                tone=tone,
                hint=f"{forecast.agreement} of {len(forecast.factors)} factors agree.",
            ),
        ]
        # Each factor's share of the projected move, in points.
        for factor in forecast.factors:
            stats.append(
                Stat(
                    f"projection_factor_{factor.key}",
                    f"  ↳ {factor.label}",
                    round(factor.points, 2),
                    unit="pts",
                    tone={"up": "up", "down": "down"}.get(factor.direction, "neutral"),
                    signed=True,
                    hint=factor.detail,
                )
            )
        return stats

    def _accuracy_stats(self, ctx: IndicatorContext, bars_5m: list) -> list[Stat]:
        if ctx.predictions is None:
            return []
        acc = ctx.predictions.accuracy(bars_5m, int(self.params["accuracy_window"]))
        if not acc["count"]:
            return [
                Stat(
                    "projection_accuracy",
                    "Measured Accuracy",
                    "No scored predictions yet",
                    hint="Each projection is scored once its 5-minute bar closes.",
                )
            ]

        error = acc["mean_abs_error"]
        baseline = acc["baseline_abs_error"]
        beats = error < baseline

        if acc["direction_rate"] is None:
            direction_stat = Stat(
                "projection_direction_rate",
                "Direction Correct",
                "n/a",
                hint="No directional calls yet — every projection has been flat.",
            )
        else:
            direction = acc["direction_rate"] * 100
            direction_stat = Stat(
                "projection_direction_rate",
                "Direction Correct",
                round(direction, 1),
                unit="%",
                precision=1,
                tone="up" if direction > 55 else ("down" if direction < 45 else "neutral"),
                hint=(
                    f"Over {acc['directional_count']} directional calls "
                    f"(of {acc['count']} scored). 50% is a coin flip."
                ),
            )

        return [
            direction_stat,
            Stat(
                "projection_band_rate",
                "Landed In Range",
                round(acc["band_rate"] * 100, 1),
                unit="%",
                precision=1,
                tone="neutral",
            ),
            Stat(
                "projection_error",
                "Avg Error",
                round(error, 2),
                unit="pts",
                tone="up" if beats else "down",
                hint=(
                    f"Baseline (assume no move): {baseline:.2f} pts. "
                    + ("Model is beating it." if beats else "Model is NOT beating it.")
                ),
            ),
            Stat(
                "projection_verdict",
                "Vs No-Move Baseline",
                "Adding value" if beats else "Not adding value",
                tone="up" if beats else "down",
                hint="If the model cannot beat assuming price stays put, ignore it.",
            ),
        ]


def _structure_from(ctx: IndicatorContext) -> str:
    """Read the daily HH/LL verdict from the structure indicator."""
    from . import get as get_indicator

    try:
        indicator = get_indicator("daily_structure")
    except KeyError:
        return ""
    stat = next(
        (s for s in indicator.compute(ctx).stats if s.key == "structure"), None
    )
    return str(stat.value) if stat else ""
