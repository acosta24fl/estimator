"""Event-driven backtester.

Walks the 5m series one bar at a time and never looks forward. On each bar it
manages any open position first, then considers a new entry - the same order the
live loop uses.

The engine consumes a frame of *out-of-sample* model predictions produced by
``walk_forward_evaluate``. It does not fit anything. That separation is what
keeps the reported P&L honest: every probability it acts on came from a model
that had not seen the bar it is acting on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..config import Config
from ..labeling import LONG, SHORT
from ..trade.manager import Bar, Trade, TradeManager
from ..trade.signals import build_signal, select_direction
from .metrics import compute_metrics, trades_to_frame

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    trades: list[Trade]
    metrics: dict[str, Any]
    equity: pd.Series
    frame: pd.DataFrame

    def report(self, title: str = "Backtest") -> str:
        from .metrics import format_report

        return format_report(self.metrics, title)


def run_backtest(
    matrix: pd.DataFrame,
    predictions: pd.DataFrame,
    cfg: Config,
    verbose: bool = False,
) -> BacktestResult:
    """Simulate the strategy over ``matrix`` using ``predictions``.

    ``matrix`` supplies OHLC and ATR; ``predictions`` supplies
    ``long_p_meta`` / ``short_p_meta`` per bar. Bars without a prediction are
    still traversed (open trades must keep being managed) but cannot open a
    new position.
    """
    tcfg = cfg.trade
    lcfg = cfg.labels
    manager = TradeManager(tcfg)

    # Restrict to the span the walk-forward actually covered.
    if predictions.empty:
        raise ValueError("no predictions supplied; run walk-forward evaluation first")
    start, end = predictions.index.min(), predictions.index.max()
    sim = matrix.loc[(matrix.index >= start) & (matrix.index <= end)]
    if sim.empty:
        raise ValueError("prediction index does not overlap the price matrix")

    preds = predictions.reindex(sim.index)
    p_long = preds.get("long_p_meta", pd.Series(np.nan, index=sim.index)).to_numpy(float)
    p_short = preds.get("short_p_meta", pd.Series(np.nan, index=sim.index)).to_numpy(float)

    o = sim["open"].to_numpy(float)
    h = sim["high"].to_numpy(float)
    l = sim["low"].to_numpy(float)
    c = sim["close"].to_numpy(float)
    v = sim["volume"].to_numpy(float) if "volume" in sim else np.zeros(len(sim))
    atr = sim["atr"].to_numpy(float)
    times = sim.index

    closed: list[Trade] = []
    open_trades: list[Trade] = []
    last_entry_bar = -10**9

    for i in range(len(sim) - 1):
        bar = Bar(
            timestamp=times[i],
            open=o[i], high=h[i], low=l[i], close=c[i],
            volume=v[i], atr=atr[i],
        )

        # ---- 1. manage existing positions --------------------------------
        still_open: list[Trade] = []
        for trade in open_trades:
            # Continuation probability = this bar's probability for the side the
            # trade is on. It is the same number the live monitor reads.
            cont = p_long[i] if trade.direction == LONG else p_short[i]
            cont = None if cont != cont else float(cont)
            manager.on_bar(trade, bar, cont)
            if trade.is_open:
                still_open.append(trade)
            else:
                closed.append(trade)
                if verbose:
                    log.info(
                        "exit  %s %s @ %.2f (%s) %.1f pts",
                        trade.exit_time, "LONG" if trade.direction == LONG else "SHORT",
                        trade.exit_price, trade.exit_reason, trade.realised_points,
                    )
        open_trades = still_open

        # ---- 2. consider a new entry -------------------------------------
        if len(open_trades) >= tcfg.max_concurrent_trades:
            continue
        if i - last_entry_bar < tcfg.cooldown_bars:
            continue

        choice = select_direction(p_long[i], p_short[i], tcfg)
        if choice is None:
            continue
        direction, prob = choice

        # Entry fills at the NEXT bar's open: the decision uses this bar's close,
        # which is only known once the bar is over.
        entry_price = o[i + 1]
        signal = build_signal(
            timestamp=times[i + 1],
            direction=direction,
            entry=entry_price,
            atr=atr[i],
            probability=prob,
            lcfg=lcfg,
            tcfg=tcfg,
            components={
                "p_long": float(p_long[i]) if p_long[i] == p_long[i] else 0.0,
                "p_short": float(p_short[i]) if p_short[i] == p_short[i] else 0.0,
            },
        )
        if signal is None:
            continue

        trade = Trade(
            direction=direction,
            entry_time=times[i + 1],
            entry_price=signal.entry,
            initial_stop=signal.stop,
            target=signal.target,
            atr_at_entry=float(atr[i]),
            contracts=tcfg.contracts,
            probability=prob,
        )
        open_trades.append(trade)
        last_entry_bar = i
        if verbose:
            log.info(
                "entry %s %s @ %.2f stop %.2f tgt %.2f p=%.3f",
                trade.entry_time, signal.side, signal.entry, signal.stop,
                signal.target, prob,
            )

    # Flatten anything still open on the final bar so P&L is complete.
    if open_trades:
        last = len(sim) - 1
        final_bar = Bar(
            timestamp=times[last], open=o[last], high=h[last],
            low=l[last], close=c[last], volume=v[last], atr=atr[last],
        )
        for trade in open_trades:
            trade.exit_time = final_bar.timestamp
            trade.exit_price = final_bar.close
            trade.exit_reason = "end_of_data"
            closed.append(trade)

    metrics = compute_metrics(closed, tcfg)
    frame = trades_to_frame(closed, tcfg)
    equity = (
        frame.set_index("exit_time")["net_usd"].cumsum()
        if not frame.empty
        else pd.Series(dtype=float)
    )
    return BacktestResult(trades=closed, metrics=metrics, equity=equity, frame=frame)
