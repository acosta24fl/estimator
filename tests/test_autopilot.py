"""Autonomous operation and the paper-trade journal.

Two properties matter more than the rest. The loop must never die - a crash in
one cycle would silently end autonomous operation while the dashboard kept
serving a frozen snapshot, which looks identical to a quiet market. And the
drift check must distinguish "no evidence of a problem" from "too early to
tell", because at realistic trade rates the second state lasts weeks and
reporting it as the first manufactures false confidence.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from mnq.autopilot import Autopilot, AutopilotConfig
from mnq.config import Config
from mnq.paper import (
    MIN_FOR_INFERENCE,
    PaperJournal,
    baseline_from_artifacts,
    compare_to_backtest,
)


@pytest.fixture
def cfg():
    return Config()


def _row(net, when="2026-01-02T10:00:00Z", direction="LONG"):
    return {
        "entry_time": when, "exit_time": when, "direction": direction,
        "entry_price": 21_000.0, "exit_price": 21_000.0 + net / 2,
        "stop_price": 20_950.0, "target_price": 21_100.0,
        "realised_points": net / 2, "net_usd": net,
        "exit_reason": "take_profit" if net > 0 else "stop_loss",
        "probability": 0.61, "bars_held": 12,
    }


class TestJournal:
    def test_empty_journal_reads_cleanly(self, tmp_path):
        j = PaperJournal(tmp_path / "p.csv")
        assert j.frame().empty
        assert j.stats()["n"] == 0

    def test_records_and_aggregates(self, tmp_path):
        j = PaperJournal(tmp_path / "p.csv")
        for net in (100.0, -50.0, 80.0, -50.0):
            j.record(_row(net))
        s = j.stats()
        assert s["n"] == 4 and s["wins"] == 2
        assert s["win_rate"] == pytest.approx(0.5)
        assert s["net_usd"] == pytest.approx(80.0)
        assert s["expectancy_usd"] == pytest.approx(20.0)

    def test_appends_across_restarts(self, tmp_path):
        path = tmp_path / "p.csv"
        PaperJournal(path).record(_row(100.0))
        PaperJournal(path).record(_row(-50.0))     # a fresh object, as after a restart
        assert PaperJournal(path).stats()["n"] == 2

    def test_a_corrupt_journal_does_not_raise(self, tmp_path):
        path = tmp_path / "p.csv"
        path.write_text("this is not,a valid\ncsv\x00\x00\n")
        assert PaperJournal(path).stats()["n"] == 0

    def test_confidence_interval_once_there_are_enough(self, tmp_path):
        j = PaperJournal(tmp_path / "p.csv")
        for i in range(40):
            j.record(_row(100.0 if i % 3 == 0 else -40.0))
        s = j.stats()
        assert s["ci95_low_usd"] < s["expectancy_usd"] < s["ci95_high_usd"]


class TestDrift:
    def _live(self, n, wins):
        return {"n": n, "wins": wins, "win_rate": wins / n if n else None}

    def test_no_baseline_is_reported_not_assumed(self):
        got = compare_to_backtest(self._live(50, 20), None)
        assert got["verdict"] == "no baseline"

    def test_no_trades(self):
        got = compare_to_backtest({"n": 0}, 0.375)
        assert got["verdict"] == "no trades yet"

    def test_small_samples_are_called_too_early(self):
        """Five losses in a row is normal at a 37.5% win rate."""
        got = compare_to_backtest(self._live(5, 0), 0.375)
        assert got["verdict"] == "too early"
        assert str(MIN_FOR_INFERENCE) in got["detail"]

    def test_a_matching_rate_is_consistent(self):
        got = compare_to_backtest(self._live(100, 38), 0.375)
        assert got["verdict"] == "consistent"
        assert got["p_value"] > 0.05

    def test_a_genuine_collapse_is_flagged(self):
        got = compare_to_backtest(self._live(200, 30), 0.375)   # 15% vs 37.5%
        assert got["verdict"] == "broken"
        assert got["p_value"] < 0.05
        assert "not a losing streak" in got["detail"]

    def test_outperformance_is_flagged_but_not_celebrated(self):
        got = compare_to_backtest(self._live(200, 130), 0.375)  # 65% vs 37.5%
        assert got["verdict"] == "outperforming"
        assert "treat it as luck" in got["detail"]

    def test_reports_a_confidence_interval(self):
        got = compare_to_backtest(self._live(60, 22), 0.375)
        lo, hi = got["ci95"]
        assert 0.0 <= lo < hi <= 1.0

    def test_baseline_reads_what_backtest_wrote(self, tmp_path):
        import json

        (tmp_path / "profitability.json").write_text(
            json.dumps({"expectancy": {"win_rate": 0.375, "n_trades": 688}})
        )
        assert baseline_from_artifacts(tmp_path) == pytest.approx(0.375)

    def test_missing_baseline_file_is_none(self, tmp_path):
        assert baseline_from_artifacts(tmp_path) is None


class _FakeStore:
    def __init__(self):
        self.bars = []
        self.flushed = 0

    def last_bar_time(self):
        return self.bars[-1] if self.bars else None

    def flush(self):
        self.flushed += 1

    def __len__(self):
        return len(self.bars)


class _FakeEngine:
    """Records what the loop asks of it, and can be told to misbehave."""

    def __init__(self, fail_on=None):
        self.store = _FakeStore()
        self.closed_trades = []
        self.bundle = object()
        self.fail_on = fail_on or set()
        self.evaluated = 0

    def on_minute_bar(self, ts, ohlcv):
        self.store.bars.append(ts)
        return {}

    def refresh_models(self):
        return False

    def evaluate(self, force=False):
        if "evaluate" in self.fail_on:
            raise RuntimeError("evaluate exploded")
        self.evaluated += 1
        return None

    def status(self):
        return {"bars": len(self.store)}


def _bars(n=5, start="2026-01-02 10:00"):
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"open": 21000.0, "high": 21005.0, "low": 20995.0,
         "close": 21000.0, "volume": 10.0},
        index=idx,
    )


class TestAutopilotLoop:
    def _pilot(self, cfg, tmp_path, engine=None, **kw):
        from mnq import autopilot as ap

        pilot = Autopilot(cfg, engine or _FakeEngine(), AutopilotConfig(**kw))
        pilot.journal = PaperJournal(tmp_path / "p.csv")
        return pilot, ap

    def test_poll_feeds_only_new_bars(self, cfg, tmp_path, monkeypatch):
        from mnq import autopilot as ap

        pilot, mod = self._pilot(cfg, tmp_path)
        monkeypatch.setattr("mnq.data.yahoo.download", lambda *a, **k: _bars(5))
        monkeypatch.setattr("mnq.data.yahoo.drop_maintenance", lambda df: df)

        pilot._poll()
        assert len(pilot.engine.store) == 5
        pilot._poll()   # same window again
        assert len(pilot.engine.store) == 5, "already-seen bars must not replay"

    def test_a_failed_poll_is_counted_not_fatal(self, cfg, tmp_path, monkeypatch):
        pilot, _ = self._pilot(cfg, tmp_path)

        def boom(*a, **k):
            raise RuntimeError("Yahoo is down")

        monkeypatch.setattr("mnq.data.yahoo.download", boom)
        pilot._poll()
        assert pilot.state.poll_failures == 1
        assert "Yahoo is down" in pilot.state.last_error

    def test_signal_respects_its_interval(self, cfg, tmp_path):
        pilot, _ = self._pilot(cfg, tmp_path, signal_seconds=600)
        pilot._maybe_signal()
        assert pilot.engine.evaluated == 1
        pilot._maybe_signal()          # immediately again
        assert pilot.engine.evaluated == 1, "must wait for the interval"

        pilot.state.last_signal_check -= timedelta(seconds=601)
        pilot._maybe_signal()
        assert pilot.engine.evaluated == 2

    def test_an_exploding_evaluate_does_not_stop_the_loop(self, cfg, tmp_path):
        pilot, _ = self._pilot(cfg, tmp_path, engine=_FakeEngine({"evaluate"}))
        pilot._maybe_signal()          # must not raise
        assert "exploded" in pilot.state.last_error

    def test_closed_trades_are_journalled_once(self, cfg, tmp_path):
        from mnq.labeling import LONG

        class T:
            direction = LONG
            entry_time = datetime(2026, 1, 2, 10, tzinfo=timezone.utc)
            exit_time = datetime(2026, 1, 2, 11, tzinfo=timezone.utc)
            entry_price, exit_price = 21_000.0, 21_050.0
            stop_price, target_price = 20_950.0, 21_100.0
            realised_points = 50.0
            exit_reason = "take_profit"

            def net_usd(self, tcfg):
                return 98.26

        pilot, _ = self._pilot(cfg, tmp_path)
        pilot.engine.closed_trades = [T()]
        pilot._collect_closed_trades()
        pilot._collect_closed_trades()          # a second cycle
        assert pilot.journal.stats()["n"] == 1, "each trade is journalled once"

    def test_retrain_is_skipped_before_it_is_due(self, cfg, tmp_path):
        pilot, _ = self._pilot(cfg, tmp_path, retrain_hours=168)
        pilot.state.next_retrain = datetime.now(timezone.utc) + timedelta(hours=100)
        pilot._maybe_retrain()
        assert not pilot.state.retrain_running

    def test_retrain_can_be_disabled(self, cfg, tmp_path):
        pilot, _ = self._pilot(cfg, tmp_path, enabled_retrain=False)
        pilot.state.next_retrain = datetime.now(timezone.utc) - timedelta(hours=1)
        pilot._maybe_retrain()
        assert not pilot.state.retrain_running

    def test_a_failed_retrain_keeps_the_existing_model(self, cfg, tmp_path, monkeypatch):
        pilot, _ = self._pilot(cfg, tmp_path)
        original = pilot.engine.bundle
        monkeypatch.setattr(
            "mnq.models.pooled.pooled_walk_forward",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("retrain blew up")),
        )
        pilot._retrain()
        assert pilot.engine.bundle is original
        assert "blew up" in pilot.state.last_error
        assert not pilot.state.retrain_running

    def test_status_is_paper_and_serialisable(self, cfg, tmp_path):
        import json

        pilot, _ = self._pilot(cfg, tmp_path)
        st = pilot.status()
        assert st["mode"] == "paper"
        assert "drift" in st and "paper" in st
        json.dumps(st)      # must survive the dashboard's serialisation
