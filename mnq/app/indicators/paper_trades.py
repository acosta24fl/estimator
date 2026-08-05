"""Draws simulated trades on the chart and reports the running tally.

Entries and exits appear as arrows on the candles, and the metrics panel shows
the open position plus profit factor, win rate and net result for everything
closed so far. All simulated — no order ever leaves this machine.

Marker times are snapped to the displayed timeframe's buckets. A trade opened
on a 10-minute boundary has no matching candle on the 15-minute chart, and a
marker whose timestamp does not land on a bar is silently dropped by the chart
library, so the snap is what keeps the arrows visible on every timeframe.
"""

from __future__ import annotations

from ..core.trading import DOLLARS_PER_POINT
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

_LONG = "#2e9e6b"
_SHORT = "#d1495b"
_EXIT_WIN = "#5ad2f4"
_EXIT_LOSS = "#f2b53c"


@register
class PaperTrades(Indicator):
    key = "paper_trades"
    name = "Paper Trades (simulated)"
    order = 7
    description = "Simulated entries and exits from the directional call."
    params = {"max_markers": 80}

    render = RenderSpec(
        pane=PANE_PRICE,
        series=(
            SeriesSpec(
                key="entry_price",
                label="Open Position Entry",
                type="line",
                color="#b07cd8",
                line_width=1,
                line_style=2,
                price_line=True,
                autoscale=False,
            ),
        ),
    )

    def compute(self, ctx: IndicatorContext) -> IndicatorResult:
        result = IndicatorResult(series={"entry_price": []})
        trader = getattr(ctx, "paper", None)
        if trader is None:
            result.stats = [
                Stat("paper_off", "Paper Trading", "Off", hint="Set MNQ_PAPER_TRADING=1.")
            ]
            return result

        bucket = ctx.timeframe.bucketer
        max_markers = int(self.params["max_markers"])
        markers: list[Marker] = []

        for trade in trader.trades()[-max_markers:]:
            markers.append(
                Marker(
                    time=bucket.start(trade.opened_ts),
                    position="belowBar" if trade.direction > 0 else "aboveBar",
                    shape="arrowUp" if trade.direction > 0 else "arrowDown",
                    color=_LONG if trade.direction > 0 else _SHORT,
                    text=f"{'BUY' if trade.direction > 0 else 'SELL'} {trade.entry:,.2f}",
                )
            )
            if not trade.is_open and trade.closed_ts is not None:
                won = (trade.net_points or 0.0) > 0
                markers.append(
                    Marker(
                        time=bucket.start(trade.closed_ts),
                        position="aboveBar" if trade.direction > 0 else "belowBar",
                        shape="square",
                        color=_EXIT_WIN if won else _EXIT_LOSS,
                        text=f"{trade.net_points:+.2f}",
                    )
                )

        markers.sort(key=lambda m: m.time)
        result.markers = markers

        # A dashed line at the live entry, so an open position is obvious.
        position = trader.open_trade
        if position is not None and ctx.bars:
            result.series["entry_price"] = [
                {"time": bar.ts, "value": round(position.entry, 2)}
                for bar in ctx.bars
                if bar.ts >= bucket.start(position.opened_ts)
            ]

        result.stats = self._stats(trader, ctx)
        return result

    @staticmethod
    def _stats(trader, ctx: IndicatorContext) -> list[Stat]:
        price = ctx.bars[-1].close if ctx.bars else None
        stats = trader.stats()
        out: list[Stat] = []

        position = trader.open_trade
        if position is None:
            out.append(Stat("paper_position", "Position", "Flat", tone="neutral"))
        else:
            unrealised = position.unrealised(price) if price is not None else None
            tone = "neutral" if unrealised is None else ("up" if unrealised > 0 else "down")
            out.append(
                Stat(
                    "paper_position",
                    "Position",
                    f"{position.side.upper()} @ {position.entry:,.2f}",
                    tone="up" if position.direction > 0 else "down",
                    hint=position.reason,
                )
            )
            out.append(
                Stat(
                    "paper_open_pnl",
                    "  ↳ Unrealised",
                    None if unrealised is None else round(unrealised, 2),
                    unit="pts",
                    tone=tone,
                    signed=True,
                )
            )

        if not stats.trades:
            out.append(
                Stat(
                    "paper_none",
                    "Closed Trades",
                    0,
                    precision=0,
                    hint="A trade opens when the call is bullish or bearish.",
                )
            )
            return out

        net_tone = "up" if stats.net_points > 0 else ("down" if stats.net_points < 0 else "neutral")
        pf = stats.profit_factor
        out += [
            Stat("paper_trades", "Closed Trades", stats.trades, precision=0, tone="neutral"),
            Stat(
                "paper_net",
                "Net Result",
                round(stats.net_points, 2),
                unit="pts",
                tone=net_tone,
                signed=True,
                hint=f"${stats.net_points * DOLLARS_PER_POINT:,.2f} on one contract, "
                f"after {stats.cost_points} pts round-turn cost.",
            ),
            Stat(
                "paper_pf",
                "Profit Factor",
                None if pf is None else round(pf, 2),
                tone="up" if (pf or 0) > 1 else "down",
                hint="Gross profit / gross loss. Above 1 is profitable.",
            ),
            Stat(
                "paper_win_rate",
                "Win Rate",
                None if stats.win_rate is None else round(stats.win_rate * 100, 1),
                unit="%",
                precision=1,
                tone="neutral",
                hint=f"{stats.wins} wins / {stats.losses} losses.",
            ),
            Stat(
                "paper_drawdown",
                "Max Drawdown",
                round(stats.max_drawdown, 2),
                unit="pts",
                tone="neutral",
            ),
        ]
        return out
