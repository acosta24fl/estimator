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
from .journal import DecisionLog, NullLog
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
    #: Watch-only when False: poll, score, and write down the decision, but
    #: never open even a simulated position. This is what the plain dashboard
    #: (option 7) runs, so the page shows live price and a live read without
    #: quietly starting to accumulate paper trades behind the user's back.
    trade_enabled: bool = True

    @property
    def mode(self) -> str:
        return "paper" if self.trade_enabled else "watch"


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
    #: Timestamp of the newest bar held, and where it came from. The dashboard
    #: shows both: "polling" is not the same claim as "holding fresh data", and
    #: a stale feed that keeps polling successfully is the failure mode that
    #: looks healthiest.
    last_bar_time: datetime | None = None
    source: str = ""
    consecutive_failures: int = 0

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
        decisions: DecisionLog | None = None,
    ):
        self.cfg = cfg
        self.engine = engine
        self.auto = auto or AutopilotConfig()
        self.state = AutopilotState()
        self.journal = PaperJournal(ARTIFACT_DIR / self.auto.journal_name)
        # Share the engine's log by default so poll, score and trade events
        # land in one ordered stream rather than two files to interleave later.
        self.decisions: DecisionLog = (
            decisions or getattr(engine, "journal", None) or NullLog()
        )

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
            "%s started: poll %ds, signal %ds, retrain every %.0fh",
            self.auto.mode, self.auto.poll_seconds, self.auto.signal_seconds,
            self.auto.retrain_hours,
        )
        self.decisions.record(
            "system",
            (
                f"Live feed started in {self.auto.mode} mode. Pulling "
                f"{self.cfg.data.symbol} 1-minute bars from Yahoo Finance every "
                f"{self.auto.poll_seconds}s and re-reading the market every "
                f"{self.auto.signal_seconds}s."
                + ("" if self.auto.trade_enabled else
                   " Watch-only: it will say what it would do and open nothing.")
            ),
            mode=self.auto.mode, poll_seconds=self.auto.poll_seconds,
            signal_seconds=self.auto.signal_seconds, symbol=self.cfg.data.symbol,
        )

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self.state.started_at is not None:
            self.decisions.record(
                "system", f"Live feed stopped after {self.state.polls} polls.",
                polls=self.state.polls, signals=self.state.signals,
            )
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
        self.state.source = f"Yahoo Finance · {self.cfg.data.symbol} · 1m"
        try:
            bars = download(self.cfg.data.symbol, "1m", POLL_LOOKBACK, self.cfg.data.tz)
        except Exception as exc:  # noqa: BLE001 - a failed poll is routine
            self.state.poll_failures += 1
            self.state.consecutive_failures += 1
            self.state.last_error = f"poll: {exc}"
            log.warning("poll failed (%s); will retry next cycle", exc)
            self.decisions.record(
                "error",
                f"Could not reach the price feed ({type(exc).__name__}). "
                f"Failure {self.state.consecutive_failures} in a row; retrying "
                f"in {self.auto.poll_seconds}s. Prices on screen are the last "
                f"good ones, not current.",
                detail=str(exc)[:300],
                consecutive_failures=self.state.consecutive_failures,
            )
            return

        self.state.consecutive_failures = 0
        if self.cfg.data.drop_maintenance_break:
            bars = drop_maintenance(bars)
        if bars.empty:
            self.state.polls += 1
            self.state.last_poll = now
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
        self.state.last_bar_time = self.engine.store.last_bar_time()
        if len(fresh):
            log.info("polled: %d new bars (%d stored)", len(fresh), self.state.bars)
            price = float(fresh["close"].iloc[-1])
            self.decisions.record(
                "poll",
                f"Pulled {len(fresh)} new 1-minute bar(s) from Yahoo Finance. "
                f"Last price {price:,.2f}. {self.state.bars:,} bars held.",
                new_bars=int(len(fresh)), price=price, bars=self.state.bars,
                source="yahoo", last_bar=self.state.last_bar_time,
            )
        elif self.state.polls % 12 == 1:
            # A quiet market produces no new bars. Say so occasionally rather
            # than on every poll, so the feed does not drown in "nothing yet".
            self.decisions.record(
                "poll",
                "Checked the feed: no new bars since the last poll. The market "
                "is closed or not printing.",
                new_bars=0, bars=self.state.bars, source="yahoo",
                last_bar=self.state.last_bar_time,
            )

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
            if self.auto.trade_enabled:
                signal = self.engine.evaluate()
            else:
                # Same reasoning, same thresholds, no position. The decision
                # still gets written down - that is the whole point of the mode.
                self.engine.observe()
                signal = None
        except Exception as exc:  # noqa: BLE001
            self.state.last_error = f"evaluate: {exc}"
            log.warning("signal evaluation failed: %s", exc)
            self.decisions.record(
                "error", f"The read failed with {type(exc).__name__}; the loop "
                         f"continues and will try again next cycle.",
                detail=str(exc)[:300],
            )
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
        self.decisions.record(
            "retrain",
            "Started refitting the models on the pooled instruments. This takes "
            "several minutes; polling and position management keep running.",
            phase="start",
        )
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
            self.decisions.record(
                "retrain",
                f"Finished refitting in "
                f"{(datetime.now(timezone.utc) - started).total_seconds() / 60:.1f} "
                f"minutes. Long accuracy {metrics.get('long_auc') or float('nan'):.4f}, "
                f"short {metrics.get('short_auc') or float('nan'):.4f} "
                f"(0.50 would be a coin flip).",
                phase="done", long_auc=metrics.get("long_auc"),
                short_auc=metrics.get("short_auc"),
                minutes=(datetime.now(timezone.utc) - started).total_seconds() / 60,
            )
            log.info(
                "retrain: done in %.1f min (long AUC %.4f, short %.4f)",
                (datetime.now(timezone.utc) - started).total_seconds() / 60,
                metrics.get("long_auc") or float("nan"),
                metrics.get("short_auc") or float("nan"),
            )
        except Exception as exc:  # noqa: BLE001 - keep serving the old model
            self.state.last_error = f"retrain: {exc}"
            log.exception("retrain failed; continuing with the existing model")
            self.decisions.record(
                "error",
                f"The refit failed ({type(exc).__name__}). The previous model is "
                f"still in use, so nothing was lost.",
                phase="failed", detail=str(exc)[:300],
            )
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
            "mode": self.auto.mode,    # "paper" or "watch"; never live orders
            "trading": self.auto.trade_enabled,
            "state": self.state.to_dict(),
            "config": {
                "poll_seconds": self.auto.poll_seconds,
                "signal_seconds": self.auto.signal_seconds,
                "retrain_hours": self.auto.retrain_hours,
                "retrain_enabled": self.auto.enabled_retrain,
            },
            "feed": self.feed_status(),
            "paper": live,
            "drift": compare_to_backtest(live, baseline),
        }

    def feed_status(self) -> dict[str, Any]:
        """Is data actually arriving, and when is the next attempt?

        Separated out because "the process is running" and "the data is fresh"
        are different claims, and the page has to be able to tell them apart. A
        feed that polls successfully against a closed market is healthy; one
        that polls successfully and gets stale bars is not, and both look the
        same from the poll counter alone.
        """
        now = datetime.now(timezone.utc)
        running = bool(self._thread and self._thread.is_alive())

        def age(when: datetime | None) -> float | None:
            return None if when is None else max(0.0, (now - when).total_seconds())

        poll_age = age(self.state.last_poll)
        next_poll = (
            None if not running or poll_age is None
            else max(0.0, self.auto.poll_seconds - poll_age)
        )
        bar_age = age(self.state.last_bar_time)

        if not running:
            health, detail = "stopped", "Nothing is pulling prices."
        elif self.state.consecutive_failures:
            health, detail = "failing", (
                f"{self.state.consecutive_failures} failed poll(s) in a row — "
                f"the prices shown are the last ones that arrived."
            )
        elif bar_age is not None and bar_age > 15 * 60:
            health, detail = "stale", (
                f"Connected, but the newest bar is {bar_age / 60:.0f} minutes "
                f"old. The market is most likely closed."
            )
        elif poll_age is None:
            health, detail = "starting", "First poll has not completed yet."
        else:
            health, detail = "live", "Prices are arriving on schedule."

        return {
            "running": running,
            "mode": self.auto.mode,
            "health": health,
            "detail": detail,
            "source": self.state.source or f"Yahoo Finance · {self.cfg.data.symbol} · 1m",
            "poll_seconds": self.auto.poll_seconds,
            "signal_seconds": self.auto.signal_seconds,
            "seconds_since_poll": poll_age,
            "seconds_until_poll": next_poll,
            "seconds_since_bar": bar_age,
            "last_bar": (
                self.state.last_bar_time.isoformat()
                if self.state.last_bar_time else None
            ),
            "polls": self.state.polls,
            "poll_failures": self.state.poll_failures,
        }
