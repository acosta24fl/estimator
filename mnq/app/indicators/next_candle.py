"""Draws the predicted envelope for the next bar of whichever timeframe is shown.

Unlike the 5-minute projection, which commits to a single target price, this
draws the two levels the next candle is *unlikely to exceed* — the part of a
candle that can actually be predicted, because excursion size tracks volatility
and volatility clusters.

The levels carry price labels so they can be read off the axis directly, which
is what makes them usable as stop and target references rather than decoration.

The entry readout comes from :mod:`app.core.entries` and is scored against the
conservative fill model whenever the minute series can supply one. When the two
fill models disagree the stat says so instead of quietly showing the flattering
number.
"""

from __future__ import annotations

from ..core.candle import DEFAULT_LOOKBACK, predict
from ..core.entries import LONG, SHORT, MINUTE_SECONDS, recommend
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

#: Bars of the displayed timeframe the envelope is drawn across.
_DRAW_BARS = 24


@register
class NextCandle(Indicator):
    key = "next_candle"
    name = "Next Candle"
    order = 6
    description = "Envelope the next bar is unlikely to exceed, plus where to enter it."
    params = {"quantile": 0.80, "lookback": DEFAULT_LOOKBACK}

    def __init__(self) -> None:
        # The entry study sweeps every stored bar, which is ~90ms on the
        # 1-minute series — too much to repeat on a 10-second poll when the
        # answer only moves when a bar closes. Registered indicators are
        # singletons, so one slot per (timeframe, last bar, side) is enough.
        self._entry_cache: dict[tuple[str, int], tuple[int, object]] = {}

    render = RenderSpec(
        pane=PANE_PRICE,
        series=(
            SeriesSpec(
                key="next_high",
                label="Next Bar High",
                color="#4fbf8b",
                line_width=1,
                line_style=2,
                price_line=True,
            ),
            SeriesSpec(
                key="next_low",
                label="Next Bar Low",
                color="#e06c75",
                line_width=1,
                line_style=2,
                price_line=True,
            ),
        ),
    )

    def compute(self, ctx: IndicatorContext) -> IndicatorResult:
        result = IndicatorResult(series={"next_high": [], "next_low": []})
        completed = [b for b in ctx.bars if b.complete]
        step = ctx.timeframe.nominal_seconds
        quantile = float(self.params["quantile"])

        forecast = predict(
            completed,
            quantile=quantile,
            lookback=int(self.params["lookback"]),
            timeframe=ctx.timeframe.key,
            step_seconds=step,
        )
        if not forecast.valid:
            result.stats = [Stat("next_candle_off", "Next Candle", "—", hint=forecast.reason)]
            return result

        # Draw each level flat across the recent bars and one step into the
        # future, so the envelope reads as a level the coming bar has to beat
        # rather than as a series with shape of its own.
        span = [b.ts for b in completed[-_DRAW_BARS:]] + [forecast.ts]
        result.series["next_high"] = [
            {"time": ts, "value": round(forecast.high, 2)} for ts in span
        ]
        result.series["next_low"] = [
            {"time": ts, "value": round(forecast.low, 2)} for ts in span
        ]

        result.stats = self._stats(ctx, forecast, quantile, step)
        return result

    def _stats(self, ctx, forecast, quantile: float, step: int) -> list[Stat]:
        pct = f"{quantile:.0%}"
        stats = [
            Stat(
                "next_high",
                f"Next Bar High ({pct})",
                round(forecast.high, 2),
                hint=f"Stays below this about {pct} of the time, measured walk-forward.",
            ),
            Stat("next_low", f"Next Bar Low ({pct})", round(forecast.low, 2)),
            Stat(
                "next_range",
                "Predicted Range",
                round(forecast.predicted_range, 2),
                unit="pts",
                hint=f"Typical bar is {forecast.typical_high - forecast.typical_low:,.1f} pts.",
            ),
            Stat(
                "next_sigma",
                "Volatility Unit",
                round(forecast.sigma, 2),
                unit="pts",
                hint="EWMA of Garman-Klass variance. All depths are multiples of this.",
            ),
        ]

        direction = self._side(ctx)
        if direction is None:
            stats.append(
                Stat(
                    "next_entry",
                    "Entry",
                    "no call",
                    hint="Entry placement is only meaningful once there is a direction.",
                )
            )
            return stats

        rec = self._recommendation(ctx, direction, step)
        side = "long" if direction == LONG else "short"

        if rec.best is None:
            stats.append(
                Stat("next_entry", "Entry", "at the open", tone="warn", hint=rec.reason)
            )
        else:
            offset_pts = rec.best.points(rec.best.offset)
            where = "below" if direction == LONG else "above"
            label = (
                "at the open" if offset_pts <= 0
                else f"{offset_pts:,.2f} pts {where} open"
            )
            stats.append(
                Stat(
                    "next_entry",
                    f"Entry ({side})",
                    label,
                    tone="warn" if not rec.trustworthy else "neutral",
                    hint=rec.reason,
                )
            )

        reference = rec.best or rec.at_market
        if reference is not None:
            stats += [
                Stat(
                    "next_stop",
                    "  ↳ Stop",
                    round(reference.points(reference.p90_adverse), 2),
                    unit="pts",
                    hint="Covers 90% of the drawdown seen after this entry.",
                ),
                Stat(
                    "next_target",
                    "  ↳ Target",
                    round(reference.points(reference.median_favourable), 2),
                    unit="pts",
                    hint="Median favourable excursion from this entry.",
                ),
            ]
        if not rec.trustworthy:
            stats.append(
                Stat(
                    "next_entry_warning",
                    "  ↳ Fill check",
                    "not confirmed",
                    tone="warn",
                    hint=(
                        "Profitable only if a single printed tick fills you. "
                        "Requiring a full minute to close through the level "
                        "reverses it."
                    ),
                )
            )
        return stats

    def _recommendation(self, ctx: IndicatorContext, direction: int, step: int):
        completed = [b for b in ctx.bars if b.complete]
        last_ts = completed[-1].ts if completed else 0
        # Keyed by timeframe and side so several clients on different
        # timeframes each keep their own slot; the bar timestamp is the
        # freshness check rather than part of the key, so the map stays bounded
        # by the number of timeframes instead of growing with every closed bar.
        key = (ctx.timeframe.key, direction)
        cached_ts, cached = self._entry_cache.get(key, (None, None))
        if cached is not None and cached_ts == last_ts:
            return cached
        rec = recommend(
            completed,
            timeframe=ctx.timeframe.key,
            direction=direction,
            cost_points=ctx.settings.paper_cost_points,
            minutes=ctx.minute_bars if step > MINUTE_SECONDS else (),
            step_seconds=step,
        )
        self._entry_cache[key] = (last_ts, rec)
        return rec

    @staticmethod
    def _side(ctx: IndicatorContext) -> int | None:
        """The direction the dashboard is currently calling, if any.

        An open simulated position wins over the headline call: while a trade
        is live, the entry question is about *that* trade's side.
        """
        paper = getattr(ctx, "paper", None)
        position = getattr(paper, "open_trade", None) if paper is not None else None
        if position is not None:
            return LONG if position.direction > 0 else SHORT
        call = getattr(getattr(ctx, "outlook", None), "direction", None)
        if call == "bullish":
            return LONG
        if call == "bearish":
            return SHORT
        return None
