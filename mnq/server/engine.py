"""The live trading engine.

Owns the loop that runs against real incoming data:

* every minute, an arriving bar is folded into the store and any open trade is
  re-checked against it;
* every ``signal_interval_minutes`` (10 by default), the feature matrix is
  rebuilt and both direction models are scored for a possible new entry.

Trade management deliberately runs on the *minute* cadence rather than the
10-minute one. A stop that is only inspected every 10 minutes is not a stop.

The engine reuses :class:`~mnq.trade.manager.TradeManager` and
:func:`~mnq.trade.signals.build_signal` unchanged from the backtest, so live
behaviour matches tested behaviour by construction rather than by discipline.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..data.store import BarStore
from ..features.builder import add_session_features, build_feature_matrix
from ..labeling import LONG, SHORT
from ..models.train import load_bundle
from ..notify.telegram import TelegramNotifier, format_heartbeat
from ..trade.manager import Bar, Trade, TradeManager
from ..trade.signals import Signal, build_signal, select_direction

log = logging.getLogger(__name__)


class LiveEngine:
    """Stateful live signal generator and position monitor."""

    def __init__(self, cfg: Config, store: BarStore | None = None, load_models: bool = True):
        self.cfg = cfg
        self.store = store or BarStore(
            cfg.path(cfg.server.bar_store_path), cfg.server.max_bars_retained
        )
        self.notifier = TelegramNotifier(cfg.telegram)
        self.manager = TradeManager(cfg.trade)

        self._lock = threading.RLock()
        self.open_trades: list[Trade] = []
        self.closed_trades: list[Trade] = []
        self.last_signal_time: datetime | None = None
        self.last_evaluation: datetime | None = None
        self.signals_today = 0
        self._today = datetime.now(timezone.utc).date()

        self.bundle: dict[str, Any] | None = None
        if load_models:
            try:
                self.bundle = load_bundle(cfg)
                log.info(
                    "loaded models trained at %s (%d features)",
                    self.bundle["trained_at"], len(self.bundle["feature_names"]),
                )
            except FileNotFoundError as exc:
                # Not fatal: the webhook should still accept and store bars so
                # history accumulates while a model is being trained.
                log.error("%s — running in data-collection mode only", exc)

    # ------------------------------------------------------------ ingestion

    def on_minute_bar(self, bar_time: datetime, ohlcv: dict[str, float]) -> dict[str, Any]:
        """Handle one inbound TradingView bar: store it, then manage positions."""
        ts = self.store.add_bar(
            bar_time,
            ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"],
            ohlcv.get("volume", 0.0),
        )
        events = self._manage_open_trades(ts, ohlcv)
        return {"stored": str(ts), "bars": len(self.store), "events": events}

    def on_price(self, ts: datetime, price: float, volume: float = 0.0) -> dict[str, Any]:
        """Handle a close-only alert."""
        stamp = self.store.add_price(ts, price, volume)
        ohlcv = {"open": price, "high": price, "low": price, "close": price, "volume": volume}
        events = self._manage_open_trades(stamp, ohlcv)
        return {"stored": str(stamp), "bars": len(self.store), "events": events}

    # ------------------------------------------------------- trade monitoring

    def _manage_open_trades(
        self, ts: datetime, ohlcv: dict[str, float]
    ) -> list[dict[str, Any]]:
        """Drive every open trade through the newest bar."""
        with self._lock:
            if not self.open_trades:
                return []

            atr = self._current_atr()
            bar = Bar(
                timestamp=ts,
                open=ohlcv["open"], high=ohlcv["high"],
                low=ohlcv["low"], close=ohlcv["close"],
                volume=ohlcv.get("volume", 0.0),
                atr=atr or 0.0,
            )

            events: list[dict[str, Any]] = []
            still_open: list[Trade] = []
            for trade in self.open_trades:
                prev_stop = trade.stop
                cont = self._continuation_probability(trade.direction)
                self.manager.on_bar(trade, bar, cont)

                if trade.is_open:
                    still_open.append(trade)
                    # Only announce a stop that actually moved, so the phone is
                    # not buzzing every minute.
                    if abs(trade.stop - prev_stop) > 1e-9:
                        note = (
                            "Stop moved to breakeven."
                            if trade.breakeven_moved and not trade.trailing_active
                            else "Trailing stop advanced."
                        )
                        self.notifier.send_update(trade, bar.close, note)
                        events.append({"type": "stop_moved", "stop": trade.stop})
                else:
                    self.closed_trades.append(trade)
                    note = ""
                    if trade.exit_reason == "early_exit_signal":
                        note = (
                            "Model confidence in continuation fell below "
                            f"{self.cfg.trade.early_exit_prob:.0%} — closed early."
                        )
                    self.notifier.send_exit(trade, note)
                    events.append(
                        {
                            "type": "exit",
                            "reason": trade.exit_reason,
                            "points": round(trade.realised_points, 2),
                        }
                    )
                    log.info(
                        "closed %s: %s %.1f pts",
                        "LONG" if trade.direction == LONG else "SHORT",
                        trade.exit_reason, trade.realised_points,
                    )

            self.open_trades = still_open
            self._save_state()
            return events

    def _continuation_probability(self, direction: int) -> float | None:
        """Current model probability that ``direction`` still wins from here.

        This is the trend-continuation judgement the monitor acts on. Returns
        ``None`` when the models cannot score the current bar, which disables
        the early exit rather than guessing.
        """
        try:
            scored = self.score_latest()
        except Exception as exc:  # noqa: BLE001 - never break management on a model error
            log.warning("continuation scoring failed: %s", exc)
            return None
        if scored is None:
            return None
        return scored["p_long"] if direction == LONG else scored["p_short"]

    def _current_atr(self) -> float | None:
        frames = self.store.frames()
        if frames["5m"].empty or len(frames["5m"]) < self.cfg.features.atr_window + 2:
            return None
        from .. import indicators as ind

        f = frames["5m"]
        series = ind.atr(f["high"], f["low"], f["close"], self.cfg.features.atr_window)
        value = series.iloc[-1]
        return float(value) if pd.notna(value) else None

    # ------------------------------------------------------------- scoring

    def build_matrix(self) -> pd.DataFrame | None:
        """Feature matrix from live bars, or ``None`` if history is too short."""
        frames = self.store.frames()
        if frames["5m"].empty or len(frames["5m"]) < 60:
            return None
        matrix = build_feature_matrix(frames, self.cfg.features)
        return add_session_features(matrix)

    def score_latest(self) -> dict[str, Any] | None:
        """Score the most recent completed 5m bar for both directions."""
        if self.bundle is None:
            return None
        matrix = self.build_matrix()
        if matrix is None or matrix.empty:
            return None

        names = self.bundle["feature_names"]
        missing = [c for c in names if c not in matrix.columns]
        if missing:
            log.error("live matrix missing %d features (e.g. %s)", len(missing), missing[:5])
            return None

        row = matrix.iloc[[-1]]
        X = row[names]
        # A row that is mostly NaN means the warm-up is incomplete; scoring it
        # would produce a confident number from almost no information.
        if X.notna().mean(axis=1).iloc[0] < 0.6:
            log.info("insufficient warm-up for a reliable score; skipping")
            return None

        reg = self.bundle["shared_regressor"]
        fwd = reg.predict(X)

        out: dict[str, Any] = {
            "timestamp": row.index[-1],
            "close": float(row["close"].iloc[0]),
            "atr": float(row["atr"].iloc[0]) if pd.notna(row["atr"].iloc[0]) else 0.0,
            "fwd_pred": float(fwd[0]),
        }
        for name, direction in (("long", LONG), ("short", SHORT)):
            ens = self.bundle["directions"][name]
            comp = ens.predict_components(X, fwd)
            out[f"p_{name}"] = float(comp["p_meta"].iloc[0])
            out[f"{name}_components"] = {
                k: float(comp[k].iloc[0]) for k in ("p_xgb", "p_lgbm", "p_meta")
            }
        return out

    # ------------------------------------------------------- signal cadence

    def evaluate(self, force: bool = False) -> Signal | None:
        """Run one signal evaluation. Called on the 10-minute cadence."""
        with self._lock:
            now = datetime.now(timezone.utc)
            self._roll_day(now)
            self.last_evaluation = now

            if self.bundle is None:
                log.debug("no models loaded; evaluation skipped")
                return None
            if len(self.open_trades) >= self.cfg.trade.max_concurrent_trades and not force:
                log.debug("max concurrent trades reached; not evaluating entries")
                return None
            if self._in_cooldown(now) and not force:
                log.debug("in cooldown; not evaluating entries")
                return None

            scored = self.score_latest()
            if scored is None:
                return None

            choice = select_direction(scored["p_long"], scored["p_short"], self.cfg.trade)
            if choice is None:
                log.info(
                    "no signal (p_long=%.3f p_short=%.3f, need %.2f)",
                    scored["p_long"], scored["p_short"], self.cfg.trade.min_probability,
                )
                return None
            direction, prob = choice

            # Enter at the live price, not at the closed bar's close.
            entry = self.store.last_price() or scored["close"]
            name = "long" if direction == LONG else "short"
            signal = build_signal(
                timestamp=now,
                direction=direction,
                entry=float(entry),
                atr=scored["atr"],
                probability=prob,
                lcfg=self.cfg.labels,
                tcfg=self.cfg.trade,
                components={**scored[f"{name}_components"], "fwd_pred": scored["fwd_pred"]},
                context={
                    "bar": str(scored["timestamp"]),
                    "p_long": round(scored["p_long"], 3),
                    "p_short": round(scored["p_short"], 3),
                },
            )
            if signal is None:
                log.info(
                    "setup rejected by gates (atr=%.1f, target=%.1f pts < %.1f min)",
                    scored["atr"], scored["atr"] * self.cfg.labels.tp_atr_mult,
                    self.cfg.trade.min_edge_points,
                )
                return None

            trade = Trade(
                direction=direction,
                entry_time=now,
                entry_price=signal.entry,
                initial_stop=signal.stop,
                target=signal.target,
                atr_at_entry=signal.atr,
                contracts=self.cfg.trade.contracts,
                probability=prob,
            )
            self.open_trades.append(trade)
            self.last_signal_time = now
            self.signals_today += 1

            self.notifier.send_signal(signal)
            log.info(
                "SIGNAL %s @ %.2f stop %.2f target %.2f p=%.3f",
                signal.side, signal.entry, signal.stop, signal.target, prob,
            )
            self._save_state()
            return signal

    def _in_cooldown(self, now: datetime) -> bool:
        if self.last_signal_time is None:
            return False
        elapsed = (now - self.last_signal_time).total_seconds() / 60.0
        return elapsed < self.cfg.trade.cooldown_bars * 5

    def _roll_day(self, now: datetime) -> None:
        if now.date() != self._today:
            self._today = now.date()
            self.signals_today = 0

    # --------------------------------------------------------------- status

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "bars": len(self.store),
                "last_bar": str(self.store.last_bar_time()),
                "last_price": self.store.last_price(),
                "open_trades": len(self.open_trades),
                "closed_trades": len(self.closed_trades),
                "signals_today": self.signals_today,
                "last_evaluation": str(self.last_evaluation),
                "models_loaded": self.bundle is not None,
                "trades": [t.to_dict() for t in self.open_trades],
            }

    def send_heartbeat(self) -> None:
        self.notifier.send(format_heartbeat(self.status()))

    def flatten_all(self, reason: str = "manual_flat") -> list[Trade]:
        """Close everything at the last price - the panic button."""
        with self._lock:
            price = self.store.last_price()
            now = datetime.now(timezone.utc)
            closed = []
            for trade in self.open_trades:
                trade.exit_time = now
                trade.exit_price = float(price) if price else trade.entry_price
                trade.exit_reason = reason
                self.closed_trades.append(trade)
                self.notifier.send_exit(trade, "Flattened by operator request.")
                closed.append(trade)
            self.open_trades = []
            self._save_state()
            return closed

    def _save_state(self) -> None:
        """Persist open positions so a restart does not orphan a live trade."""
        path = self.cfg.path(self.cfg.server.state_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "open_trades": [t.to_dict() for t in self.open_trades],
                "signals_today": self.signals_today,
                "last_signal_time": (
                    self.last_signal_time.isoformat() if self.last_signal_time else None
                ),
            }
            path.write_text(json.dumps(payload, indent=2, default=str))
        except Exception as exc:  # noqa: BLE001 - persistence must not break trading
            log.warning("could not save state: %s", exc)
