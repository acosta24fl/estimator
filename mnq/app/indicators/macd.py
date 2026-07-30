"""MACD (12, 26, 9) for whichever timeframe is selected.

Because every timeframe is folded from the same 1-minute series, the MACD you
see on 15m is computed from exactly the 15m candles on screen.
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
    ema,
)


@register
class Macd(Indicator):
    key = "macd"
    name = "MACD"
    order = 30
    description = "Moving-average convergence/divergence with signal line and histogram."
    params = {"fast": 12, "slow": 26, "signal": 9}

    render = RenderSpec(
        pane=PANE_OWN,
        height=140,
        zero_line=True,
        series=(
            SeriesSpec(
                key="histogram",
                label="Histogram",
                type="histogram",
                up_color="#2e9e6b",
                down_color="#d1495b",
            ),
            SeriesSpec(key="macd", label="MACD", type="line", color="#4c9aff", line_width=2),
            SeriesSpec(
                key="signal", label="Signal", type="line", color="#f2b53c", line_width=1
            ),
        ),
    )

    def compute(self, ctx: IndicatorContext) -> IndicatorResult:
        fast_p = int(self.params["fast"])
        slow_p = int(self.params["slow"])
        signal_p = int(self.params["signal"])

        result = IndicatorResult(series={"macd": [], "signal": [], "histogram": []})
        closes = [b.close for b in ctx.bars]
        if len(closes) < slow_p:
            result.stats = [
                Stat(
                    "macd_state",
                    "MACD",
                    "Warming up",
                    tone="neutral",
                    hint=f"Needs {slow_p} bars, have {len(closes)}.",
                )
            ]
            return result

        fast = ema(closes, fast_p)
        slow = ema(closes, slow_p)

        macd_line: list[float | None] = [
            (f - s) if (f is not None and s is not None) else None
            for f, s in zip(fast, slow)
        ]

        # The signal EMA runs over the MACD line's defined region only.
        defined_idx = [i for i, v in enumerate(macd_line) if v is not None]
        signal_line: list[float | None] = [None] * len(macd_line)
        if len(defined_idx) >= signal_p:
            dense = [macd_line[i] for i in defined_idx]  # type: ignore[misc]
            sig = ema(dense, signal_p)
            for pos, i in enumerate(defined_idx):
                signal_line[i] = sig[pos]

        last_macd = last_signal = last_hist = None
        prev_hist = None
        for i, bar in enumerate(ctx.bars):
            m = macd_line[i]
            s = signal_line[i]
            if m is not None:
                result.series["macd"].append({"time": bar.ts, "value": round(m, 3)})
                last_macd = m
            if s is not None:
                result.series["signal"].append({"time": bar.ts, "value": round(s, 3)})
                last_signal = s
            if m is not None and s is not None:
                hist = m - s
                result.series["histogram"].append(
                    {"time": bar.ts, "value": round(hist, 3)}
                )
                prev_hist = last_hist
                last_hist = hist

        if last_macd is None:
            return result

        tone = "neutral"
        state = "Flat"
        if last_signal is None:
            # MACD line exists but the signal EMA has not warmed up yet.
            state = "Warming up"
        elif last_hist is not None:
            tone = "up" if last_hist > 0 else ("down" if last_hist < 0 else "neutral")
            state = "Bullish" if last_hist > 0 else ("Bearish" if last_hist < 0 else "Flat")
            if prev_hist is not None and (prev_hist <= 0 < last_hist):
                state = "Bullish cross"
            elif prev_hist is not None and (prev_hist >= 0 > last_hist):
                state = "Bearish cross"

        result.stats = [
            Stat("macd_value", f"MACD ({fast_p},{slow_p})", round(last_macd, 2), tone="neutral"),
            Stat(
                "macd_signal",
                f"Signal ({signal_p})",
                None if last_signal is None else round(last_signal, 2),
                tone="neutral",
            ),
            Stat(
                "macd_hist",
                "MACD Histogram",
                None if last_hist is None else round(last_hist, 2),
                tone=tone,
                signed=True,
            ),
            Stat("macd_state", "MACD State", state, tone=tone),
        ]
        return result
