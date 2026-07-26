"""Open-trade lifecycle: stops, targets, trailing, and early exit.

This module is deliberately the *only* place trade management logic lives. The
backtester and the live monitor both drive it bar by bar with the same inputs,
so a rule that was validated in a backtest is literally the same code that runs
on a real position. Forking this logic into a separate live implementation is
the usual way a strategy comes to behave nothing like its backtest.

Exit precedence within a bar is fixed and pessimistic: stop, then target, then
discretionary exits. 5m OHLC does not reveal the intra-bar path, so when a bar
straddles both barriers the loss is booked.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

from ..config import POINT_VALUE_USD, TICK_SIZE, TradeConfig

LONG = 1
SHORT = -1

# Exit reasons, surfaced in Telegram messages and backtest reports.
EXIT_STOP = "stop_loss"
EXIT_TARGET = "take_profit"
EXIT_TRAIL = "trailing_stop"
EXIT_EARLY = "early_exit_signal"
EXIT_TIME = "time_stop"
EXIT_FLAT = "manual_flat"


@dataclass
class Bar:
    """One price bar. Mirrors the live webhook payload field-for-field."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    atr: float = 0.0


@dataclass
class Trade:
    """A position from entry to exit."""

    direction: int
    entry_time: datetime
    entry_price: float
    initial_stop: float
    target: float
    atr_at_entry: float
    contracts: int = 1
    probability: float = 0.0

    stop: float = 0.0
    bars_held: int = 0
    extreme: float = 0.0          # best price reached, in the trade's favour
    mfe_points: float = 0.0       # max favourable excursion
    mae_points: float = 0.0       # max adverse excursion
    breakeven_moved: bool = False
    trailing_active: bool = False

    exit_time: datetime | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    exit_probability: float | None = None

    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stop == 0.0:
            self.stop = self.initial_stop
        if self.extreme == 0.0:
            self.extreme = self.entry_price

    # ------------------------------------------------------------- geometry

    @property
    def risk_points(self) -> float:
        """Distance from entry to the *initial* stop. R is always measured
        against original risk, never the trailed stop - otherwise R inflates as
        the stop moves and every rule keyed to R silently drifts."""
        return abs(self.entry_price - self.initial_stop)

    @property
    def reward_points(self) -> float:
        return abs(self.target - self.entry_price)

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    def unrealised_points(self, price: float) -> float:
        return self.direction * (price - self.entry_price)

    def r_multiple(self, price: float) -> float:
        risk = self.risk_points
        return self.unrealised_points(price) / risk if risk > 0 else 0.0

    @property
    def realised_points(self) -> float:
        if self.exit_price is None:
            return 0.0
        return self.direction * (self.exit_price - self.entry_price)

    def net_usd(self, tcfg: TradeConfig) -> float:
        """Realised P&L after slippage and commission, both sides."""
        if self.exit_price is None:
            return 0.0
        slip = tcfg.slippage_ticks * TICK_SIZE * 2.0
        gross = (self.realised_points - slip) * POINT_VALUE_USD * self.contracts
        fees = tcfg.commission_usd_per_side * 2.0 * self.contracts
        return gross - fees

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["direction"] = "LONG" if self.direction == LONG else "SHORT"
        d["realised_points"] = self.realised_points
        d["risk_points"] = self.risk_points
        return d


class TradeManager:
    """Applies the management rules to an open trade, one bar at a time."""

    def __init__(self, cfg: TradeConfig):
        self.cfg = cfg

    def on_bar(
        self, trade: Trade, bar: Bar, continuation_prob: float | None = None
    ) -> Trade:
        """Advance ``trade`` through ``bar``, closing it if a rule fires.

        ``continuation_prob`` is the model's current probability that this
        direction still wins from here. Passing ``None`` disables the early
        exit, which is what happens when the model cannot score the bar.
        """
        if not trade.is_open:
            return trade

        trade.bars_held += 1
        self._update_excursions(trade, bar)

        # 1. Hard stop. Checked first, and against the bar's extreme rather than
        #    its close, because a stop is a resting order.
        if self._stop_touched(trade, bar):
            reason = EXIT_TRAIL if trade.trailing_active or trade.breakeven_moved else EXIT_STOP
            # Gapping through the stop fills at the open, not at the stop price.
            fill = self._gap_adjusted_stop(trade, bar)
            return self._close(trade, bar, fill, reason, continuation_prob)

        # 2. Target.
        if self._target_touched(trade, bar):
            return self._close(trade, bar, trade.target, EXIT_TARGET, continuation_prob)

        # 3. Stop maintenance for bars that survived.
        self._apply_breakeven(trade)
        self._apply_trailing(trade, bar)

        # 4. Discretionary exit: the model no longer believes in the move.
        if (
            self.cfg.enable_monitor
            and continuation_prob is not None
            and trade.bars_held >= self.cfg.early_exit_min_bars
            and continuation_prob < self.cfg.early_exit_prob
        ):
            return self._close(trade, bar, bar.close, EXIT_EARLY, continuation_prob)

        # 5. Time stop: capital tied up in a trade going nowhere.
        if trade.bars_held >= self.cfg.max_hold_bars:
            if trade.r_multiple(bar.close) <= self.cfg.time_stop_r:
                return self._close(trade, bar, bar.close, EXIT_TIME, continuation_prob)

        return trade

    # ------------------------------------------------------------- internals

    def _update_excursions(self, trade: Trade, bar: Bar) -> None:
        if trade.direction == LONG:
            trade.extreme = max(trade.extreme, bar.high)
            trade.mfe_points = max(trade.mfe_points, bar.high - trade.entry_price)
            trade.mae_points = max(trade.mae_points, trade.entry_price - bar.low)
        else:
            trade.extreme = min(trade.extreme, bar.low)
            trade.mfe_points = max(trade.mfe_points, trade.entry_price - bar.low)
            trade.mae_points = max(trade.mae_points, bar.high - trade.entry_price)

    def _stop_touched(self, trade: Trade, bar: Bar) -> bool:
        if trade.direction == LONG:
            return bar.low <= trade.stop
        return bar.high >= trade.stop

    def _target_touched(self, trade: Trade, bar: Bar) -> bool:
        if trade.direction == LONG:
            return bar.high >= trade.target
        return bar.low <= trade.target

    def _gap_adjusted_stop(self, trade: Trade, bar: Bar) -> float:
        """If the bar opened beyond the stop, that is the realistic fill."""
        if trade.direction == LONG:
            return min(trade.stop, bar.open)
        return max(trade.stop, bar.open)

    def _apply_breakeven(self, trade: Trade) -> None:
        if trade.breakeven_moved or self.cfg.breakeven_at_r <= 0:
            return
        risk = trade.risk_points
        if risk <= 0:
            return
        if trade.mfe_points / risk >= self.cfg.breakeven_at_r:
            # One tick beyond entry so the trade is free after costs, not merely
            # flat on price.
            offset = TICK_SIZE * trade.direction
            candidate = trade.entry_price + offset
            trade.stop = max(trade.stop, candidate) if trade.direction == LONG else min(trade.stop, candidate)
            trade.breakeven_moved = True

    def _apply_trailing(self, trade: Trade, bar: Bar) -> None:
        if self.cfg.trail_atr_mult <= 0:
            return
        risk = trade.risk_points
        if risk <= 0:
            return
        if trade.mfe_points / risk < self.cfg.trail_start_r:
            return

        atr = bar.atr if bar.atr > 0 else trade.atr_at_entry
        if atr <= 0:
            return
        distance = self.cfg.trail_atr_mult * atr

        if trade.direction == LONG:
            candidate = trade.extreme - distance
            # A stop only ever ratchets toward profit.
            if candidate > trade.stop:
                trade.stop = candidate
                trade.trailing_active = True
        else:
            candidate = trade.extreme + distance
            if candidate < trade.stop:
                trade.stop = candidate
                trade.trailing_active = True

    def _close(
        self,
        trade: Trade,
        bar: Bar,
        price: float,
        reason: str,
        prob: float | None,
    ) -> Trade:
        trade.exit_time = bar.timestamp
        trade.exit_price = float(price)
        trade.exit_reason = reason
        trade.exit_probability = prob
        return trade
