"""Liveness: is the system actually moving, and can the page tell?

A dashboard that renders a fresh timestamp while nothing behind it has changed
is worse than one that renders nothing, because it manufactures confidence in
stale prices. The tests here defend the distinctions that make the difference
visible:

* **running** is not the same claim as **fresh** - a feed that polls
  successfully against a closed market is healthy, and one that polls
  successfully and receives nothing new for an hour is not, yet both increment
  the same counter;
* **watch-only** must reason exactly like paper mode and open nothing, because
  a mode that quietly skipped the reasoning would leave the page blank for a
  reason nobody could see;
* a blank projection must name its own cause, since "no model", "no data" and
  "cold feed" all render as the same empty panel.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from mnq.autopilot import Autopilot, AutopilotConfig
from mnq.config import Config
from mnq.journal import DecisionLog
from mnq.server import dashboard


@pytest.fixture
def cfg():
    return Config()


class _Store:
    def __init__(self):
        self.bars: list = []

    def __len__(self):
        return len(self.bars)

    def last_bar_time(self):
        return self.bars[-1] if self.bars else None

    def flush(self):
        pass


class _Engine:
    """Enough of LiveEngine for the loop, with the calls it made recorded."""

    def __init__(self, journal=None):
        self.store = _Store()
        self.closed_trades = []
        self.bundle = object()
        self.journal = journal
        self.evaluated = 0
        self.observed = 0

    def on_minute_bar(self, ts, ohlcv):
        self.store.bars.append(ts)
        return {}

    def evaluate(self, force=False):
        self.evaluated += 1
        return None

    def observe(self):
        self.observed += 1
        return None

    def status(self):
        return {"bars": len(self.store)}


def _bars(n=5, start="2026-01-02 10:00"):
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"open": 21_000.0, "high": 21_010.0, "low": 20_990.0,
         "close": 21_005.0, "volume": 100.0},
        index=idx,
    )


def _pilot(cfg, tmp_path, **kw):
    journal = DecisionLog(tmp_path / "decisions.jsonl")
    engine = _Engine(journal)
    pilot = Autopilot(cfg, engine, AutopilotConfig(**kw), decisions=journal)
    from mnq.paper import PaperJournal

    pilot.journal = PaperJournal(tmp_path / "paper.csv")
    return pilot, journal


class TestWatchOnlyMode:
    def test_watch_mode_reads_the_market_but_opens_nothing(self, cfg, tmp_path):
        """The dashboard's own feed must not start accumulating paper trades."""
        pilot, _ = _pilot(cfg, tmp_path, trade_enabled=False)
        pilot._maybe_signal()
        assert pilot.engine.observed == 1
        assert pilot.engine.evaluated == 0, "watch mode must not open positions"

    def test_paper_mode_evaluates_for_real(self, cfg, tmp_path):
        pilot, _ = _pilot(cfg, tmp_path, trade_enabled=True)
        pilot._maybe_signal()
        assert pilot.engine.evaluated == 1 and pilot.engine.observed == 0

    def test_the_mode_is_named_in_the_status(self, cfg, tmp_path):
        watch, _ = _pilot(cfg, tmp_path, trade_enabled=False)
        paper, _ = _pilot(cfg, tmp_path, trade_enabled=True)
        assert watch.status()["mode"] == "watch"
        assert paper.status()["mode"] == "paper"
        # Whatever the mode, it is never live order routing.
        assert not watch.status()["trading"]
        assert paper.status()["mode"] != "live"


class TestFeedStatus:
    def test_a_stopped_feed_says_so(self, cfg, tmp_path):
        pilot, _ = _pilot(cfg, tmp_path)
        feed = pilot.feed_status()
        assert feed["running"] is False and feed["health"] == "stopped"
        assert "Nothing is pulling prices" in feed["detail"]

    def test_a_countdown_to_the_next_poll_is_reported(self, cfg, tmp_path):
        pilot, _ = _pilot(cfg, tmp_path, poll_seconds=60)
        pilot._thread = _AliveThread()
        pilot.state.last_poll = datetime.now(timezone.utc) - timedelta(seconds=20)
        pilot.state.last_bar_time = datetime.now(timezone.utc)

        feed = pilot.feed_status()
        assert feed["health"] == "live"
        assert 15 <= feed["seconds_since_poll"] <= 25
        assert 35 <= feed["seconds_until_poll"] <= 45

    def test_successful_polls_against_a_dead_feed_are_not_called_live(
        self, cfg, tmp_path
    ):
        """The failure mode that looks healthiest: polling fine, data frozen."""
        pilot, _ = _pilot(cfg, tmp_path, poll_seconds=60)
        pilot._thread = _AliveThread()
        pilot.state.last_poll = datetime.now(timezone.utc)
        pilot.state.last_bar_time = datetime.now(timezone.utc) - timedelta(hours=3)

        feed = pilot.feed_status()
        assert feed["health"] == "stale"
        assert "closed" in feed["detail"]

    def test_consecutive_failures_are_surfaced(self, cfg, tmp_path, monkeypatch):
        pilot, journal = _pilot(cfg, tmp_path)
        pilot._thread = _AliveThread()

        def boom(*a, **k):
            raise RuntimeError("Yahoo is down")

        monkeypatch.setattr("mnq.data.yahoo.download", boom)
        pilot._poll()
        pilot._poll()

        feed = pilot.feed_status()
        assert feed["health"] == "failing"
        assert pilot.state.consecutive_failures == 2
        assert "last good ones" in journal.tail(1)[0]["summary"]

    def test_a_recovered_poll_clears_the_failure_run(
        self, cfg, tmp_path, monkeypatch
    ):
        pilot, _ = _pilot(cfg, tmp_path)
        monkeypatch.setattr("mnq.data.yahoo.download",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
        pilot._poll()
        assert pilot.state.consecutive_failures == 1

        monkeypatch.setattr("mnq.data.yahoo.download", lambda *a, **k: _bars(3))
        monkeypatch.setattr("mnq.data.yahoo.drop_maintenance", lambda df: df)
        pilot._poll()
        assert pilot.state.consecutive_failures == 0
        assert pilot.state.last_bar_time is not None


class TestPollsAreLogged:
    def test_new_bars_produce_a_readable_line(self, cfg, tmp_path, monkeypatch):
        pilot, journal = _pilot(cfg, tmp_path)
        monkeypatch.setattr("mnq.data.yahoo.download", lambda *a, **k: _bars(4))
        monkeypatch.setattr("mnq.data.yahoo.drop_maintenance", lambda df: df)

        pilot._poll()
        entry = journal.tail(1, kinds=["poll"])[0]
        assert entry["new_bars"] == 4
        assert "Yahoo" in entry["summary"] and "21,005" in entry["summary"]

    def test_a_quiet_market_does_not_flood_the_log(self, cfg, tmp_path, monkeypatch):
        """No new bars is the common case; it must not drown the feed."""
        pilot, journal = _pilot(cfg, tmp_path)
        monkeypatch.setattr("mnq.data.yahoo.download", lambda *a, **k: _bars(3))
        monkeypatch.setattr("mnq.data.yahoo.drop_maintenance", lambda df: df)

        for _ in range(15):
            pilot._poll()
        quiet = [e for e in journal.tail(200, kinds=["poll"]) if e["new_bars"] == 0]
        assert 0 < len(quiet) <= 3, f"expected occasional notes, got {len(quiet)}"


class _AliveThread:
    def is_alive(self):
        return True


class TestReadiness:
    def _engine_status(self, bars=20_000, models=True):
        return {"bars": bars, "models_loaded": models}

    def test_a_missing_model_is_named_with_its_fix(self, cfg):
        got = dashboard.readiness(
            cfg, self._engine_status(models=False), None, [], None
        )
        model = [c for c in got["checks"] if c["name"] == "Trained model"][0]
        assert not model["ok"]
        assert "option 3" in model["fix"]
        assert not got["ready"]

    def test_thin_history_is_named_separately_from_a_missing_model(self, cfg):
        got = dashboard.readiness(cfg, self._engine_status(bars=50), None, [], None)
        history = [c for c in got["checks"] if c["name"] == "Price history"][0]
        assert not history["ok"] and "option 1" in history["fix"]

    def test_a_healthy_system_reports_ready(self, cfg):
        feed = {"running": True, "health": "live", "detail": "fine"}
        got = dashboard.readiness(
            cfg, self._engine_status(), {"close": 21_000.0},
            [{"has_skill": True}, {"has_skill": False}], feed,
        )
        assert got["ready"] and got["blocking"] == 0
        assert "in place" in got["headline"]

    def test_horizons_report_how_many_are_usable(self, cfg):
        feed = {"running": True, "health": "live", "detail": "fine"}
        got = dashboard.readiness(
            cfg, self._engine_status(), {"close": 1.0},
            [{"has_skill": False}] * 7, feed,
        )
        hz = [c for c in got["checks"] if c["name"] == "Forecast horizons"][0]
        assert hz["state"] == "0/7 usable"

    def test_no_feed_is_reported_rather_than_assumed_fine(self, cfg):
        got = dashboard.readiness(cfg, self._engine_status(), None, [], None)
        live = got["checks"][0]
        assert not live["ok"] and "option 7" in live["fix"]


class TestEveryDecisionIsWrittenDown:
    """The rejections are the point.

    "Why did it skip that move?" is the question that improves a trading
    system, and it can only be answered from a written record of the skips.
    Each test below pins one reason the system stands aside and asserts the
    reason is legible without knowing the codebase.
    """

    def _engine(self, cfg, tmp_path, **over):
        from mnq.data.store import BarStore
        from mnq.server.engine import LiveEngine

        for key, value in over.items():
            setattr(cfg.trade, key, value)
        journal = DecisionLog(tmp_path / "decisions.jsonl")
        engine = LiveEngine(
            cfg, store=BarStore(tmp_path / "bars.csv", 5_000),
            load_models=False, journal=journal,
        )
        return engine, journal

    def test_an_injected_empty_store_is_not_swapped_for_the_live_one(
        self, cfg, tmp_path
    ):
        """BarStore defines __len__, so an empty one is falsy.

        With ``store or BarStore(...)`` these tests silently read and wrote the
        real bar cache in artifacts/, and so would anyone injecting a scratch
        store. The engine must take the store it was handed.
        """
        engine, _ = self._engine(cfg, tmp_path)
        assert len(engine.store) == 0
        assert engine.store.path == tmp_path / "bars.csv"

    def _scored(self, p_long=0.51, p_short=0.44, atr=40.0):
        return {
            "timestamp": pd.Timestamp("2026-07-29T10:00:00Z"),
            "close": 21_000.0, "atr": atr, "fwd_pred": 0.4,
            "p_long": p_long, "p_short": p_short,
            "long_components": {"p_xgb": p_long, "p_lgbm": p_long, "p_meta": p_long},
            "short_components": {"p_xgb": p_short, "p_lgbm": p_short, "p_meta": p_short},
        }

    def test_a_missing_model_is_logged_not_silently_skipped(self, cfg, tmp_path):
        engine, journal = self._engine(cfg, tmp_path)
        assert engine.observe() is None
        entry = journal.tail(1, kinds=["score"])[0]
        assert entry["reason"] == "no_model"
        assert "option 3" in entry["summary"]

    def test_standing_aside_is_recorded_with_the_shortfall(self, cfg, tmp_path):
        engine, journal = self._engine(cfg, tmp_path, min_probability=0.58)
        engine.bundle = object()
        engine.score_latest = lambda: self._scored(p_long=0.51, p_short=0.44)

        engine.observe()
        gate = journal.tail(1, kinds=["gate"])[0]
        assert gate["decision"] == "no_setup"
        assert "51.0%" in gate["summary"] and "58%" in gate["summary"]
        assert "Standing aside is the correct action" in gate["summary"]

    def test_a_move_too_small_to_pay_for_itself_says_so(self, cfg, tmp_path):
        engine, journal = self._engine(
            cfg, tmp_path, min_probability=0.55, min_edge_points=25.0
        )
        engine.bundle = object()
        # A 1-point ATR cannot produce a 25-point target.
        engine.score_latest = lambda: self._scored(p_long=0.70, atr=1.0)

        assert engine.evaluate() is None
        gate = journal.tail(1, kinds=["gate"])[0]
        assert gate["reason"] == "target_too_small"
        assert "too quiet" in gate["summary"]
        assert gate["min_edge_points"] == 25.0

    def test_the_score_itself_is_recorded_in_plain_words(self, cfg, tmp_path):
        engine, journal = self._engine(cfg, tmp_path, min_probability=0.58)
        engine.bundle = object()
        engine.score_latest = lambda: self._scored(p_long=0.62, p_short=0.31)

        engine.observe()
        score = journal.tail(1, kinds=["score"])[0]
        assert "62.0% confidence a long works" in score["summary"]
        assert score["p_long"] == pytest.approx(0.62)

    def test_watch_only_says_it_would_have_traded(self, cfg, tmp_path):
        """A setup that clears the bar but opens nothing must still be visible."""
        engine, journal = self._engine(cfg, tmp_path, min_probability=0.58)
        engine.bundle = object()
        engine.score_latest = lambda: self._scored(p_long=0.71, p_short=0.20)

        engine.observe()
        gate = journal.tail(1, kinds=["gate"])[0]
        assert gate["decision"] == "setup" and gate["side"] == "LONG"
        assert gate["watch_only"] is True
        assert "opened" in gate["summary"]

    def test_a_cooldown_is_reported_rather_than_being_invisible(self, cfg, tmp_path):
        engine, journal = self._engine(cfg, tmp_path)
        engine.bundle = object()
        engine.last_signal_time = datetime.now(timezone.utc)

        assert engine.evaluate() is None
        gate = journal.tail(1, kinds=["gate"])[0]
        assert gate["decision"] == "skipped" and "cooldown" in gate["reason"]


class TestSnapshotCarriesLiveness:
    def test_a_snapshot_without_a_feed_says_it_is_static(self, cfg):
        payload = dashboard.build_snapshot(cfg, None, None, None)
        assert payload["feed"]["health"] == "stopped"
        assert payload["feed"]["mode"] == "static"
        assert "not pulling prices" in payload["feed"]["detail"]

    def test_the_feed_block_is_lifted_out_of_the_autopilot_status(self, cfg):
        auto = {"feed": {"running": True, "health": "live", "detail": "ok",
                         "source": "Yahoo Finance · MNQ=F · 1m"}}
        payload = dashboard.build_snapshot(cfg, None, None, None, autopilot=auto)
        assert payload["feed"]["source"].startswith("Yahoo")
        assert payload["readiness"]["checks"][0]["ok"]

    def test_activity_rides_along_and_stays_serialisable(self, cfg):
        import json

        events = [{"t": "2026-07-29T10:00:00+00:00", "kind": "poll",
                   "summary": "Pulled 2 new 1-minute bar(s)."}]
        payload = dashboard.build_snapshot(cfg, None, None, None, activity=events)
        assert payload["activity"] == events
        assert "NaN" not in json.dumps(payload)
