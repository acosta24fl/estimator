"""Autonomous operation: poll, score, paper-trade, retrain, repeat.

One long-lived local process that keeps the system working without anyone
driving it:

* **poll** the market on a fixed cadence and merge new bars into the cache,
* **manage** open paper positions bar by bar through the same TradeManager the
  backtest uses,
* **evaluate** for new signals on the configured signal interval,
* **journal** every closed trade and compare the live win rate to the backtest,
* **retrain** on a slow cadence so the model does not go stale,
* **serve** all of it to the dashboard.

Two decisions worth stating plainly.

**It paper-trades. It does not place orders.** There is no broker integration
in this repository. The system passed its permutation test at p=0.0067, which
justifies forward testing and nothing more - and the honest thing an autonomous
loop can do at that stage is generate the forward evidence, not spend money on
its absence.

**Retraining happens on a copy and is swapped in atomically.** A retrain that
failed halfway while the live engine held the models would leave the process
scoring with a half-written bundle, and nothing downstream would notice.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from .config import ARTIFACT_DIR, Config
from .paper import PaperJournal, baseline_from_artifacts, compare_to_backtest, trade_to_row

log = logging.getLogger(__name__)

#: Yahoo revises recent bars; re-pulling a short window each cycle picks up
#: those corrections without re-downloading two years every minute.
POLL_LOOKBACK = "5d"


@dataclass
class AutopilotConfig:
    poll_seconds: int = 300           # market data refresh
    signal_seconds: int = 600         # how often to look for a new signal
    retrain_hours: float = 168.0      # weekly
    retrain_on_start: bool = False
    journal_name: str = "paper_trades.csv"
    enabled_retrain: bool = True


@dataclass
class AutopilotState:
    started_at: datetime | None = None
    last_poll: datetime | None = None
    last_signal_check: datetime | None = None
    last_retrain: datetime | None = None
    next_retrain: datetime | None = None
    polls: int = 0
    poll_failures: int = 0
    signals: int = 0
    retrains: int = 0
    retrain_running: bool = False
    last_error: str = ""
    bars: int = 0

    def to_dict(self) -> dict[str, Any]:
        def s(v):
            return v.isoformat() if isinstance(v, datetime) else v
        return {k: s(v) for k, v in self.__dict__.items()}


class Autopilot:
    """The supervised loop. Start it, leave it running."""

    def __init__(
        self,
        cfg: Config,
        engine,
        auto: AutopilotConfig | None = None,
    ):
        self.cfg = cfg
        self.engine = engine
        self.auto = auto or AutopilotConfig()
        self.state = AutopilotState()
        self.journal = PaperJournal(ARTIFACT_DIR / self.auto.journal_name)

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._retrain_thread: threading.Thread | None = None
        self._seen_closed = 0

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.state.started_at = datetime.now(timezone.utc)
        self.state.next_retrain = self.state.started_at + timedelta(
            hours=self.auto.retrain_hours
        )
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="autopilot", daemon=True)
        self._thread.start()
        log.info(
            "autopilot started: poll %ds, signal %ds, retrain every %.0fh",
            self.auto.poll_seconds, self.auto.signal_seconds, self.auto.retrain_hours,
        )

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        try:
            self.engine.store.flush()
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            log.warning("could not flush the bar store: %s", exc)
        log.info("autopilot stopped")

    # ------------------------------------------------------------- main loop

    def _run(self) -> None:
        if self.auto.retrain_on_start:
            self._maybe_retrain(force=True)

        while not self._stop.is_set():
            cycle_started = time.monotonic()
            try:
                self._poll()
                self._maybe_signal()
                self._collect_closed_trades()
                self._maybe_retrain()
            except Exception as exc:  # noqa: BLE001 - the loop must not die
                # A crash here would silently end autonomous operation while
                # the dashboard kept serving a frozen snapshot.
                self.state.last_error = f"{type(exc).__name__}: {exc}"
                log.exception("autopilot cycle failed; continuing")

            elapsed = time.monotonic() - cycle_started
            self._stop.wait(max(1.0, self.auto.poll_seconds - elapsed))

    # ----------------------------------------------------------------- steps

    def _poll(self) -> None:
        """Fetch recent bars and feed them to the engine."""
        from .data.yahoo import download, drop_maintenance

        now = datetime.now(timezone.utc)
        try:
            bars = download(self.cfg.data.symbol, "1m", POLL_LOOKBACK, self.cfg.data.tz)
        except Exception as exc:  # noqa: BLE001 - a failed poll is routine
            self.state.poll_failures += 1
            self.state.last_error = f"poll: {exc}"
            log.warning("poll failed (%s); will retry next cycle", exc)
            return

        if self.cfg.data.drop_maintenance_break:
            bars = drop_maintenance(bars)
        if bars.empty:
            return

        # Feed only bars the store has not seen, so trade management runs once
        # per bar rather than replaying the whole window every poll.
        last = self.engine.store.last_bar_time()
        fresh = bars[bars.index > last] if last is not None else bars
        for ts, row in fresh.iterrows():
            self.engine.on_minute_bar(
                ts.to_pydatetime(),
                {
                    "open": float(row["open"]), "high": float(row["high"]),
                    "low": float(row["low"]), "close": float(row["close"]),
                    "volume": float(row.get("volume", 0.0)),
                },
            )

        self.engine.store.flush()
        self.state.polls += 1
        self.state.last_poll = now
        self.state.bars = len(self.engine.store)
        if len(fresh):
            log.info("polled: %d new bars (%d stored)", len(fresh), self.state.bars)

    def _maybe_signal(self) -> None:
        now = datetime.now(timezone.utc)
        due = (
            self.state.last_signal_check is None
            or (now - self.state.last_signal_check).total_seconds()
            >= self.auto.signal_seconds
        )
        if not due:
            return
        self.state.last_signal_check = now
        try:
            signal = self.engine.evaluate()
        except Exception as exc:  # noqa: BLE001
            self.state.last_error = f"evaluate: {exc}"
            log.warning("signal evaluation failed: %s", exc)
            return
        if signal is not None:
            self.state.signals += 1
            log.info("signal: %s", signal)

    def _collect_closed_trades(self) -> None:
        """Journal any trade the engine has closed since the last cycle."""
        closed = list(self.engine.closed_trades)
        for trade in closed[self._seen_closed:]:
            try:
                self.journal.record(trade_to_row(trade))
            except Exception as exc:  # noqa: BLE001 - never lose the loop to I/O
                log.warning("could not journal a trade: %s", exc)
        self._seen_closed = len(closed)

    def _maybe_retrain(self, force: bool = False) -> None:
        if not self.auto.enabled_retrain and not force:
            return
        if self.state.retrain_running:
            return
        now = datetime.now(timezone.utc)
        if not force and self.state.next_retrain and now < self.state.next_retrain:
            return

        self.state.retrain_running = True
        self._retrain_thread = threading.Thread(
            target=self._retrain, name="autopilot-retrain", daemon=True
        )
        self._retrain_thread.start()

    def _retrain(self) -> None:
        """Refit on the pooled instruments, then swap the models in.

        Runs off the main loop so polling and trade management continue; a
        pooled retrain takes minutes and freezing position management for that
        long would be worse than a slightly stale model.
        """
        started = datetime.now(timezone.utc)
        log.info("retrain: starting (pooled)")
        try:
            from .models.pooled import pooled_walk_forward
            from .models.train import load_bundle

            # Predictions feed the dashboard's calibration table; training the
            # production bundle is a separate, cheaper step done by `train`.
            _, preds, metrics = pooled_walk_forward(
                self.cfg, n_folds=self.cfg.model.wf_folds, refresh=True
            )
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            preds.to_csv(ARTIFACT_DIR / "wf_predictions.csv")

            try:
                # Atomic from the engine's point of view: either the new bundle
                # loads cleanly and replaces the old one, or nothing changes.
                bundle = load_bundle(self.cfg)
                self.engine.bundle = bundle
                log.info("retrain: models reloaded")
            except FileNotFoundError:
                log.info("retrain: predictions refreshed; no saved bundle to reload")

            self.state.retrains += 1
            self.state.last_retrain = started
            log.info(
                "retrain: done in %.1f min (long AUC %.4f, short %.4f)",
                (datetime.now(timezone.utc) - started).total_seconds() / 60,
                metrics.get("long_auc") or float("nan"),
                metrics.get("short_auc") or float("nan"),
            )
        except Exception as exc:  # noqa: BLE001 - keep serving the old model
            self.state.last_error = f"retrain: {exc}"
            log.exception("retrain failed; continuing with the existing model")
        finally:
            self.state.retrain_running = False
            self.state.next_retrain = datetime.now(timezone.utc) + timedelta(
                hours=self.auto.retrain_hours
            )

    # ---------------------------------------------------------------- status

    def status(self) -> dict[str, Any]:
        """Everything the dashboard shows about autonomous operation."""
        live = self.journal.stats()
        baseline = baseline_from_artifacts(ARTIFACT_DIR)
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "mode": "paper",           # never anything else without a broker
            "state": self.state.to_dict(),
            "config": {
                "poll_seconds": self.auto.poll_seconds,
                "signal_seconds": self.auto.signal_seconds,
                "retrain_hours": self.auto.retrain_hours,
            },
            "paper": live,
            "drift": compare_to_backtest(live, baseline),
        }
