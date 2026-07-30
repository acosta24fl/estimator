"""Daily market structure: higher highs / lower lows.

Swing points are fractal pivots on the *daily* series (a high with ``lookback``
lower highs on each side, and the mirror for lows).  Each swing is then labelled
against the previous one of the same kind:

* HH — higher high, LH — lower high
* HL — higher low,  LL — lower low

Two step-lines carry the most recent *confirmed* swing levels onto whichever
timeframe you are viewing.  A pivot is only confirmed ``lookback`` bars after it
prints, so the lines use that confirmation time — they never show a level
earlier than it could actually have been known.
"""

from __future__ import annotations

from typing import Sequence

from ..models import Bar
from . import register
from .base import (
    PANE_PRICE,
    Indicator,
    IndicatorContext,
    IndicatorResult,
    Marker,
    RenderSpec,
    SeriesSpec,
    Stat,
)

_UP = "#2e9e6b"
_DOWN = "#d1495b"


def find_pivots(bars: Sequence[Bar], lookback: int) -> tuple[list[int], list[int]]:
    """Return (indices of pivot highs, indices of pivot lows)."""
    highs: list[int] = []
    lows: list[int] = []
    n = len(bars)
    for i in range(lookback, n - lookback):
        window = range(i - lookback, i + lookback + 1)
        if all(bars[i].high > bars[j].high for j in window if j != i):
            highs.append(i)
        if all(bars[i].low < bars[j].low for j in window if j != i):
            lows.append(i)
    return highs, lows


@register
class DailyStructure(Indicator):
    key = "daily_structure"
    name = "Daily Higher Highs / Lower Lows"
    order = 40
    description = "Fractal swing pivots on the daily series, labelled HH/LH/HL/LL."
    params = {"lookback": 2, "max_markers": 60}

    render = RenderSpec(
        pane=PANE_PRICE,
        series=(
            SeriesSpec(
                key="swing_high",
                label="Daily Swing High",
                type="line",
                color=_UP,
                line_width=1,
                line_style=2,
                autoscale=False,
            ),
            SeriesSpec(
                key="swing_low",
                label="Daily Swing Low",
                type="line",
                color=_DOWN,
                line_width=1,
                line_style=2,
                autoscale=False,
            ),
        ),
    )

    def compute(self, ctx: IndicatorContext) -> IndicatorResult:
        lookback = int(self.params["lookback"])
        result = IndicatorResult(series={"swing_high": [], "swing_low": []})
        daily = list(ctx.daily_bars)
        if len(daily) < lookback * 2 + 2:
            result.stats = [
                Stat(
                    "structure",
                    "Daily Structure",
                    "Warming up",
                    hint=f"Needs {lookback * 2 + 2} daily bars, have {len(daily)}.",
                )
            ]
            return result

        high_idx, low_idx = find_pivots(daily, lookback)

        # Label each swing against the previous swing of the same kind.
        labelled_highs: list[tuple[int, float, str]] = []  # (index, price, label)
        prev = None
        for i in high_idx:
            price = daily[i].high
            label = "HH" if prev is not None and price > prev else ("LH" if prev is not None else "H")
            labelled_highs.append((i, price, label))
            prev = price

        labelled_lows: list[tuple[int, float, str]] = []
        prev = None
        for i in low_idx:
            price = daily[i].low
            label = "HL" if prev is not None and price > prev else ("LL" if prev is not None else "L")
            labelled_lows.append((i, price, label))
            prev = price

        # Step-lines use the confirmation timestamp, not the pivot's own time.
        def confirmed(swings: list[tuple[int, float, str]]) -> list[tuple[int, float]]:
            out = []
            for idx, price, _ in swings:
                confirm_idx = min(idx + lookback, len(daily) - 1)
                out.append((daily[confirm_idx].ts, price))
            return out

        self._step_line(result.series["swing_high"], confirmed(labelled_highs), ctx.bars)
        self._step_line(result.series["swing_low"], confirmed(labelled_lows), ctx.bars)

        # Markers only make sense where each daily bar is on screen.
        if ctx.timeframe.key == "1d":
            max_markers = int(self.params["max_markers"])
            marks: list[Marker] = []
            for idx, price, label in labelled_highs:
                marks.append(
                    Marker(
                        time=daily[idx].ts,
                        position="aboveBar",
                        shape="arrowDown",
                        color=_UP if label == "HH" else _DOWN,
                        text=label,
                    )
                )
            for idx, price, label in labelled_lows:
                marks.append(
                    Marker(
                        time=daily[idx].ts,
                        position="belowBar",
                        shape="arrowUp",
                        color=_UP if label == "HL" else _DOWN,
                        text=label,
                    )
                )
            marks.sort(key=lambda m: m.time)
            result.markers = marks[-max_markers:]

        result.stats = self._stats(labelled_highs, labelled_lows, daily)
        return result

    @staticmethod
    def _step_line(
        out: list[dict], swings: list[tuple[int, float]], bars: Sequence[Bar]
    ) -> None:
        """Carry the latest known swing level across the displayed bars."""
        if not swings or not bars:
            return
        pos = 0
        level: float | None = None
        for bar in bars:
            while pos < len(swings) and swings[pos][0] <= bar.ts:
                level = swings[pos][1]
                pos += 1
            if level is not None:
                out.append({"time": bar.ts, "value": round(level, 2)})

    @staticmethod
    def _stats(
        highs: list[tuple[int, float, str]],
        lows: list[tuple[int, float, str]],
        daily: list[Bar],
    ) -> list[Stat]:
        last_high = highs[-1] if highs else None
        last_low = lows[-1] if lows else None

        structure, tone = "Range", "neutral"
        if last_high and last_low:
            hl = last_high[2]
            ll = last_low[2]
            if hl == "HH" and ll == "HL":
                structure, tone = "Uptrend", "up"
            elif hl == "LH" and ll == "LL":
                structure, tone = "Downtrend", "down"
            elif hl == "HH" or ll == "HL":
                structure, tone = "Range (bullish tilt)", "up"
            elif hl == "LH" or ll == "LL":
                structure, tone = "Range (bearish tilt)", "down"

        # How many consecutive higher highs / lower lows are on the tape.
        streak_hh = _streak([h[2] for h in highs], "HH")
        streak_ll = _streak([l[2] for l in lows], "LL")

        last_ts = daily[-1].ts if daily else 0
        day = 86400

        stats = [
            Stat("structure", "Daily Structure", structure, tone=tone),
            Stat(
                "last_daily_high",
                "Last Daily Swing High",
                None if not last_high else round(last_high[1], 2),
                tone="up" if last_high and last_high[2] == "HH" else "down",
                hint=None if not last_high else f"{last_high[2]}, {max(0, (last_ts - daily[last_high[0]].ts) // day)}d ago",
            ),
            Stat(
                "last_daily_low",
                "Last Daily Swing Low",
                None if not last_low else round(last_low[1], 2),
                tone="up" if last_low and last_low[2] == "HL" else "down",
                hint=None if not last_low else f"{last_low[2]}, {max(0, (last_ts - daily[last_low[0]].ts) // day)}d ago",
            ),
            Stat(
                "structure_streak",
                "HH / LL Streak",
                f"{streak_hh} HH / {streak_ll} LL",
                tone=tone,
                hint="Consecutive higher highs and lower lows on the daily.",
            ),
        ]
        return stats


def _streak(labels: list[str], target: str) -> int:
    count = 0
    for label in reversed(labels):
        if label == target:
            count += 1
        else:
            break
    return count
