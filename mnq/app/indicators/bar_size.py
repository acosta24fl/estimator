"""Bar size (range) for the selected timeframe.

Plots each bar's high-low range as a histogram with a rolling average, and
reports how the current bar compares — the quick read on whether the market is
expanding or contracting on this timeframe.
"""

from __future__ import annotations

from . import register
from .base import (
    PANE_OWN,
    Indicator,
    IndicatorContext,
    IndicatorResult,
    RenderSpec,
    SeriesSpec,
    Stat,
)


@register
class BarSize(Indicator):
    key = "bar_size"
    name = "Bar Size"
    order = 20
    description = "High-low range per bar, with a rolling average for context."
    params = {"average_period": 20}

    render = RenderSpec(
        pane=PANE_OWN,
        height=120,
        series=(
            SeriesSpec(
                key="range",
                label="Range",
                type="histogram",
                up_color="#3f7fb8",
                down_color="#3f7fb8",
            ),
            SeriesSpec(
                key="average",
                label="Average",
                type="line",
                color="#f2b53c",
                line_width=1,
                line_style=2,
            ),
        ),
    )

    def compute(self, ctx: IndicatorContext) -> IndicatorResult:
        period = int(self.params["average_period"])
        result = IndicatorResult(series={"range": [], "average": []})
        if not ctx.bars:
            return result

        ranges: list[float] = []
        for bar in ctx.bars:
            rng = bar.range
            ranges.append(rng)
            result.series["range"].append({"time": bar.ts, "value": round(rng, 2)})
            if len(ranges) >= period:
                avg = sum(ranges[-period:]) / period
                result.series["average"].append({"time": bar.ts, "value": round(avg, 2)})

        last = ctx.bars[-1]
        window = ranges[-period:]
        avg = sum(window) / len(window)
        ratio = (last.range / avg) if avg else 0.0

        if ratio >= 1.5:
            tone, verdict = "up", "Expanding"
        elif ratio <= 0.6:
            tone, verdict = "down", "Contracting"
        else:
            tone, verdict = "neutral", "Normal"

        body_pct = (last.body / last.range * 100.0) if last.range else 0.0

        result.stats = [
            Stat(
                "bar_size",
                f"Bar Size ({ctx.timeframe.key})",
                round(last.range, 2),
                unit="pts",
                tone="neutral",
                hint="Live bar" if not last.complete else "Closed bar",
            ),
            Stat(
                "bar_size_avg",
                f"Avg Size ({len(window)})",
                round(avg, 2),
                unit="pts",
                tone="neutral",
            ),
            Stat("bar_size_ratio", "Size vs Avg", round(ratio, 2), unit="x", tone=tone),
            Stat(
                "bar_size_body",
                "Body / Range",
                round(body_pct, 1),
                unit="%",
                tone="neutral",
                precision=1,
                hint=f"{verdict} range on {ctx.timeframe.label}.",
            ),
        ]
        return result
