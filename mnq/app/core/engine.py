"""The polling engine and snapshot builder.

Responsibilities, in order:

1. poll the feed for 1-minute bars and merge them into the store (which logs
   every bar the moment it closes),
2. periodically refresh the daily series used for market structure,
3. fold minutes into whichever timeframe is asked for, and
4. run every registered indicator and package the result for the client.

Everything the API and the WebSocket serve comes from :meth:`Engine.snapshot`,
so REST and streaming can never disagree.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from ..config import Settings
from ..feed import PriceFeed
from ..indicators import IndicatorContext, all_indicators
from ..models import Bar, Quote
from . import timeframes
from .aggregator import AggregationCache, aggregate
from .forecast import compute_forecast
from .outlook import build_outlook
from .paper import PaperTrader
from .prediction_log import PredictionLog
from .store import BarStore

log = logging.getLogger(__name__)


class Engine:
    def __init__(self, settings: Settings, feed: PriceFeed, store: BarStore) -> None:
        self.settings = settings
        self.feed = feed
        self.store = store
        self._cache = AggregationCache()
        self.predictions = PredictionLog(settings.data_dir / "predictions.jsonl")
        self.paper = PaperTrader(
            settings.data_dir / "trades.jsonl", settings.paper_cost_points
        )
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()
        self._listeners: set[Callable[[], Any]] = set()

        self.quote: Quote | None = None
        self.last_poll_ts: float | None = None
        self.last_daily_ts: float = 0.0
        self.last_error: str | None = None
        self.poll_count = 0
        self.error_count = 0
        self.consecutive_errors = 0

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        self.store.ensure_dirs()
        self.store.load()
        self.predictions.load()
        if self.settings.paper_trading:
            log.info("paper trading enabled: %d trades loaded", self.paper.load())
        await self._poll_once(include_daily=True)
        self._task = asyncio.create_task(self._loop(), name="mnq-poll")

    async def stop(self) -> None:
        self._stopping.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.feed.aclose()

    #: Longest gap between polls once the feed is failing.
    max_backoff_seconds = 300.0

    def next_delay(self) -> float:
        """Poll interval, stretched while the feed keeps failing.

        Retrying a rate-limited endpoint on the normal cadence only deepens the
        throttling, so each consecutive failure doubles the wait up to a cap.
        One success resets it.
        """
        if not self.consecutive_errors:
            return self.settings.poll_seconds
        stretched = self.settings.poll_seconds * (2**self.consecutive_errors)
        return min(stretched, self.max_backoff_seconds)

    async def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self.next_delay())
                return  # stop requested
            except asyncio.TimeoutError:
                pass
            await self._poll_once()

    # -- polling -----------------------------------------------------------

    async def _poll_once(self, include_daily: bool = False) -> None:
        now = time.time()
        try:
            snap = await self.feed.fetch_intraday()
            self.store.upsert_minute(snap.bars)
            if snap.quote is not None:
                self.quote = snap.quote
            self.last_poll_ts = now
            self.poll_count += 1
            self.last_error = None
            self.consecutive_errors = 0
        except Exception as exc:
            self.error_count += 1
            self.consecutive_errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning(
                "intraday poll failed (%d in a row, next try in %.0fs): %s",
                self.consecutive_errors,
                self.next_delay(),
                self.last_error,
            )

        try:
            self.record_prediction()
        except Exception:  # a forecasting fault must not stop data collection
            log.exception("failed to record prediction")

        if self.settings.paper_trading:
            try:
                self.update_paper_trades(now)
            except Exception:  # simulated trading must not stop data collection
                log.exception("failed to update paper trades")

        due = (now - self.last_daily_ts) >= self.settings.daily_refresh_seconds
        if include_daily or due:
            try:
                daily = await self.feed.fetch_daily()
                self.store.upsert_daily(self._normalise_daily(daily.bars))
                self.last_daily_ts = now
            except Exception as exc:
                self.error_count += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.warning("daily poll failed: %s", self.last_error)

        await self._notify()

    @staticmethod
    def _normalise_daily(bars: list[Bar]) -> list[Bar]:
        """Re-key feed daily bars onto our session boundaries.

        Feeds timestamp daily bars inconsistently (midnight, session open,
        cash open).  Mapping each through the session bucket puts them all on
        the same trade date the rest of the app uses.
        """
        session = timeframes.session_bucket()
        out: dict[int, Bar] = {}
        for bar in bars:
            key = session.start(bar.ts)
            existing = out.get(key)
            if existing is None:
                merged = bar.copy()
                merged.ts = key
                out[key] = merged
            else:
                existing.high = max(existing.high, bar.high)
                existing.low = min(existing.low, bar.low)
                existing.close = bar.close
                existing.volume += bar.volume
        return [out[k] for k in sorted(out)]

    # -- listeners ---------------------------------------------------------

    def add_listener(self, cb: Callable[[], Any]) -> None:
        self._listeners.add(cb)

    def remove_listener(self, cb: Callable[[], Any]) -> None:
        self._listeners.discard(cb)

    async def _notify(self) -> None:
        for cb in list(self._listeners):
            try:
                await cb()
            except Exception:  # pragma: no cover - listener hygiene
                log.exception("listener failed")

    # -- series ------------------------------------------------------------

    def daily_series(self) -> list[Bar]:
        """Feed daily history, with sessions we have full minute data for
        recomputed from those minutes so today's bar is live."""
        merged: dict[int, Bar] = {b.ts: b for b in self.store.daily_series()}
        minutes = self.store.minute_series()
        if minutes:
            first_ts = minutes[0].ts
            for bar in aggregate(minutes, timeframes.get("1d")):
                # A session that *began* inside our minute coverage is fully
                # represented by those minutes; earlier ones are not.
                if bar.ts >= first_ts:
                    merged[bar.ts] = bar
        return [merged[k] for k in sorted(merged)]

    def current_forecast(self):
        """Projection for the 5-minute bar currently forming."""
        return compute_forecast(
            self.bars_for(timeframes.get("5m")),
            self.daily_series(),
            timeframes.session_bucket(),
            strength=self.settings.forecast_strength,
            ridge_lambda=self.settings.forecast_ridge_lambda,
            min_fit_samples=self.settings.forecast_min_samples,
        )

    def signal_timeframe(self):
        """The registered timeframe matching the headline call's horizon."""
        wanted = self.settings.signal_horizon_minutes * 60
        for tf in timeframes.ordered():
            if tf.nominal_seconds == wanted:
                return tf
        return timeframes.get("5m")

    def current_outlook(self):
        """Bullish / bearish / no-call for the configured horizon.

        Fitted on the timeframe that matches the horizon, so a 10-minute call
        is a projection of the next 10-minute bar rather than a 5-minute one
        stretched to fit.
        """
        tf = self.signal_timeframe()
        forecast = compute_forecast(
            self.bars_for(tf),
            self.daily_series(),
            timeframes.session_bucket(),
            horizon_seconds=tf.nominal_seconds,
            strength=self.settings.forecast_strength,
            ridge_lambda=self.settings.forecast_ridge_lambda,
            min_fit_samples=self.settings.forecast_min_samples,
        )
        accuracy = self.predictions.accuracy(self.bars_for(timeframes.get("5m")))
        return build_outlook(
            forecast,
            horizon_minutes=tf.nominal_seconds // 60,
            skill=accuracy.get("skill_score"),
            min_ratio=self.settings.signal_min_ratio,
        )

    def update_paper_trades(self, now: float):
        """Advance the simulated position using the current call."""
        tf = self.signal_timeframe()
        return self.paper.update(self.bars_for(tf), self.current_outlook(), now)

    def record_prediction(self):
        """Lock the current projection, once per 5-minute bar."""
        return self.predictions.observe(self.current_forecast())

    def bars_for(self, tf: timeframes.Timeframe) -> list[Bar]:
        if tf.key == "1d":
            return self.daily_series()
        return self._cache.get(self.store.minute_series(), tf)

    # -- snapshot ----------------------------------------------------------

    def snapshot(self, tf_key: str, limit: int | None = None) -> dict[str, Any]:
        tf = timeframes.get(tf_key)
        limit = limit or tf.default_limit
        now = time.time()

        full = self.bars_for(tf)
        visible = full[-limit:] if limit else full
        cutoff = visible[0].ts if visible else 0

        ctx = IndicatorContext(
            timeframe=tf,
            bars=full,
            minute_bars=self.store.minute_series(),
            daily_bars=self.daily_series(),
            session=timeframes.session_bucket(),
            settings=self.settings,
            now=now,
            bars_5m=self.bars_for(timeframes.get("5m")),
            predictions=self.predictions,
            paper=self.paper if self.settings.paper_trading else None,
        )

        indicators: dict[str, Any] = {}
        for indicator in all_indicators():
            try:
                result = indicator.compute(ctx)
            except Exception as exc:  # one bad indicator must not break the page
                log.exception("indicator %s failed", indicator.key)
                indicators[indicator.key] = {
                    "series": {},
                    "markers": [],
                    "stats": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
                continue
            payload = result.as_dict()
            payload["series"] = {
                name: [p for p in pts if p["time"] >= cutoff]
                for name, pts in payload["series"].items()
            }
            payload["markers"] = [m for m in payload["markers"] if m["time"] >= cutoff]
            indicators[indicator.key] = payload

        last = visible[-1] if visible else None
        price = self.quote.price if self.quote else (last.close if last else None)
        prev_close = self._previous_session_close()
        change = None if (price is None or prev_close is None) else price - prev_close
        change_pct = (
            None if (change is None or not prev_close) else change / prev_close * 100.0
        )

        return {
            "timeframe": tf.key,
            "generated_at": now,
            "symbol": self.settings.symbol,
            "display_name": self.settings.display_name,
            "price": price,
            "prev_close": prev_close,
            "change": None if change is None else round(change, 2),
            "change_pct": None if change_pct is None else round(change_pct, 3),
            "bars": [b.as_chart_dict() for b in visible],
            "outlook": self._outlook_payload(),
            "paper": self.paper.summary(price) if self.settings.paper_trading else None,
            "indicators": indicators,
            "status": self.status(),
        }

    def _outlook_payload(self) -> dict[str, Any]:
        try:
            return self.current_outlook().as_dict()
        except Exception:  # a bad call must not break the chart
            log.exception("failed to build outlook")
            from .outlook import Outlook

            return Outlook(reason="unavailable").as_dict()

    def _previous_session_close(self) -> float | None:
        daily = self.daily_series()
        if len(daily) >= 2:
            return daily[-2].close
        if self.quote and self.quote.prev_close:
            return self.quote.prev_close
        return None

    def status(self) -> dict[str, Any]:
        return {
            "feed": self.feed.key,
            "feed_name": self.feed.name,
            "symbol": self.settings.symbol,
            "poll_seconds": self.settings.poll_seconds,
            "last_poll": self.last_poll_ts,
            "last_error": self.last_error,
            "poll_count": self.poll_count,
            "error_count": self.error_count,
            "consecutive_errors": self.consecutive_errors,
            "next_poll_seconds": round(self.next_delay(), 1),
            "minute_bars": self.store.minute_count,
            "daily_bars": self.store.daily_count,
            "bars_logged": self.store.logged_count,
            "last_logged_ts": self.store.last_logged_ts,
            "data_dir": str(self.settings.data_dir),
        }
