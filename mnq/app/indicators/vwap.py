"""Session VWAP and the current distance from it.

VWAP is volume-weighted typical price accumulated from the session open and
reset at each new futures trade date (18:00 ET), which is what a futures
platform shows.  It is computed on the *selected* timeframe's bars, so the line
you see is exactly the line the numbers describe.
"""

from __future__ import annotations

from ..core.features import session_vwap
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


@register
class SessionVwap(Indicator):
    key = "vwap"
    name = "Session VWAP"
    order = 10
    description = "Volume-weighted average price, reset each futures trade date."

    render = RenderSpec(
        pane=PANE_PRICE,
        series=(
            SeriesSpec(
                key="vwap",
                label="VWAP",
                type="line",
                color="#f2b53c",
                line_width=2,
            ),
        ),
    )

    def compute(self, ctx: IndicatorContext) -> IndicatorResult:
        result = IndicatorResult(series={"vwap": []})
        if not ctx.bars:
            return result

        # VWAP is a within-session measure.  On a timeframe whose bars are each
        # a whole session or longer it would reset every bar and simply trace
        # the typical price, so we report it as not applicable rather than
        # drawing a meaningless line.
        if ctx.timeframe.nominal_seconds >= 86_400:
            result.stats = [
                Stat(
                    "vwap_na",
                    "VWAP",
                    "n/a on this timeframe",
                    tone="neutral",
                    hint="Session VWAP resets each trade date; use an intraday timeframe.",
                )
            ]
            return result

        values = session_vwap(ctx.bars, ctx.session)
        for bar, value in zip(ctx.bars, values):
            result.series["vwap"].append({"time": bar.ts, "value": round(value, 2)})

        last = ctx.bars[-1]
        latest = values[-1] if values else None
        if latest is None:
            return result

        session_start = ctx.session.start(last.ts)
        session_bars = sum(1 for b in ctx.bars if ctx.session.start(b.ts) == session_start)

        distance = last.close - latest
        distance_pct = (distance / latest * 100.0) if latest else 0.0
        tone = "up" if distance > 0 else ("down" if distance < 0 else "neutral")

        result.stats = [
            Stat("vwap", "VWAP", round(latest, 2), unit="", tone="neutral"),
            Stat(
                "vwap_distance",
                "VWAP Distance",
                round(distance, 2),
                unit="pts",
                tone=tone,
                signed=True,
                hint="Last close minus session VWAP.",
            ),
            Stat(
                "vwap_distance_pct",
                "VWAP Distance %",
                round(distance_pct, 3),
                unit="%",
                tone=tone,
                precision=3,
                signed=True,
            ),
            Stat(
                "vwap_side",
                "VWAP Side",
                "Above" if distance > 0 else ("Below" if distance < 0 else "At"),
                tone=tone,
                hint=f"{session_bars} bar(s) into the current session.",
            ),
        ]
        return result
