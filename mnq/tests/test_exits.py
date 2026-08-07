import random

import pytest

from app.core.exits import (
    LONG,
    SHORT,
    STOP,
    TARGET,
    TIMEOUT,
    break_even_win_rate,
    resolve,
    run,
    theoretical_win_rate,
)
from app.models import Bar

BASE = 1_700_000_000 // 60 * 60


def minute(i, o, h, l, c):
    return Bar(ts=BASE + i * 60, open=o, high=h, low=l, close=c,
               volume=10, count=1, complete=True)


def flat_walk(n, tick=0.9, ticks=60, seed=3, start=19000.0):
    rng = random.Random(seed)
    price = start
    out = []
    for i in range(n):
        o = price
        hi = lo = price
        for _ in range(ticks):
            price += rng.gauss(0.0, tick)
            hi, lo = max(hi, price), min(lo, price)
        out.append(minute(i, o, hi, lo, price))
    return out


class TestArithmetic:
    def test_a_1_to_5_bracket_breaks_even_at_one_in_six(self):
        assert theoretical_win_rate(5.0, 25.0) == pytest.approx(1 / 6)

    def test_costs_raise_the_bar(self):
        free = break_even_win_rate(5.0, 25.0, 0.0)
        paid = break_even_win_rate(5.0, 25.0, 0.75)
        assert paid > free
        assert free == pytest.approx(1 / 6)

    def test_a_target_inside_the_cost_can_never_break_even(self):
        assert break_even_win_rate(5.0, 0.5, 0.75) is None

    def test_a_symmetric_bracket_needs_a_coin_flip(self):
        assert theoretical_win_rate(10.0, 10.0) == pytest.approx(0.5)


class TestResolution:
    def test_a_stop_is_taken(self):
        path = [minute(0, 100.0, 100.5, 94.0, 95.0)]
        t = resolve(path, LONG, 100.0, 5.0, 25.0, cost_points=0.0)
        assert t.exit_reason == STOP
        assert t.net_points == pytest.approx(-5.0)

    def test_a_target_is_taken(self):
        path = [minute(0, 100.0, 126.0, 99.5, 125.0)]
        t = resolve(path, LONG, 100.0, 5.0, 25.0, cost_points=0.0)
        assert t.exit_reason == TARGET
        assert t.net_points == pytest.approx(25.0)

    def test_the_stop_wins_a_tie_inside_one_minute(self):
        """1-minute data cannot say which came first, and the pessimistic
        reading is the one that will not flatter a strategy."""
        path = [minute(0, 100.0, 130.0, 90.0, 100.0)]
        t = resolve(path, LONG, 100.0, 5.0, 25.0, cost_points=0.0)
        assert t.exit_reason == STOP

    def test_a_gap_through_the_stop_fills_worse_than_the_stop(self):
        """Price traded past the order. Filling at the level would be a fill
        nobody got, and it always flatters the result."""
        path = [minute(0, 100.0, 100.0, 100.0, 100.0),
                minute(1, 88.0, 89.0, 87.0, 88.0)]
        t = resolve(path, LONG, 100.0, 5.0, 25.0, cost_points=0.0)
        assert t.exit_reason == STOP
        assert t.exit == 88.0
        assert t.net_points == pytest.approx(-12.0)

    def test_a_gap_through_the_target_fills_better(self):
        path = [minute(0, 100.0, 100.0, 100.0, 100.0),
                minute(1, 130.0, 131.0, 129.0, 130.0)]
        t = resolve(path, LONG, 100.0, 5.0, 25.0, cost_points=0.0)
        assert t.exit == 130.0

    def test_an_unresolved_trade_times_out_at_the_last_close(self):
        path = [minute(i, 100.0, 101.0, 99.0, 100.5) for i in range(5)]
        t = resolve(path, LONG, 100.0, 5.0, 25.0, cost_points=0.0)
        assert t.exit_reason == TIMEOUT
        assert t.exit == 100.5

    def test_a_short_stops_out_upward(self):
        path = [minute(0, 100.0, 106.0, 99.0, 105.0)]
        t = resolve(path, SHORT, 100.0, 5.0, 25.0, cost_points=0.0)
        assert t.exit_reason == STOP
        assert t.net_points == pytest.approx(-5.0)

    def test_a_short_targets_downward(self):
        path = [minute(0, 100.0, 100.5, 74.0, 75.0)]
        t = resolve(path, SHORT, 100.0, 5.0, 25.0, cost_points=0.0)
        assert t.exit_reason == TARGET
        assert t.net_points == pytest.approx(25.0)

    def test_costs_are_charged_once(self):
        path = [minute(0, 100.0, 126.0, 99.5, 125.0)]
        t = resolve(path, LONG, 100.0, 5.0, 25.0, cost_points=0.75)
        assert t.net_points == pytest.approx(25.0 - 0.75)

    def test_mfe_and_mae_track_the_extremes(self):
        path = [minute(0, 100.0, 108.0, 97.0, 100.0),
                minute(1, 100.0, 103.0, 96.0, 100.0)]
        t = resolve(path, LONG, 100.0, 0.0, 0.0, cost_points=0.0)
        assert t.mfe == pytest.approx(8.0)
        assert t.mae == pytest.approx(-4.0)

    def test_no_levels_means_hold_to_the_end(self):
        path = [minute(i, 100.0, 140.0, 60.0, 110.0) for i in range(3)]
        t = resolve(path, LONG, 100.0, 0.0, 0.0, cost_points=0.0)
        assert t.exit_reason == TIMEOUT
        assert t.net_points == pytest.approx(10.0)

    def test_an_empty_path_does_not_crash(self):
        t = resolve([], LONG, 100.0, 5.0, 25.0)
        assert t.exit == 100.0 and t.minutes_held == 0


class TestNoFreeLunch:
    """A bracket cannot manufacture an edge, and this is what proves the code
    does not accidentally hand it one."""

    @classmethod
    @pytest.fixture(scope="class")
    def minutes(cls):
        return flat_walk(6000, seed=5)

    @pytest.mark.parametrize("stop,target", [(5.0, 25.0), (10.0, 10.0), (20.0, 5.0)])
    def test_a_driftless_walk_pays_nothing(self, minutes, stop, target):
        signals = [(minutes[i].ts, LONG, minutes[i].open) for i in range(0, 5000, 20)]
        result = run(signals, minutes, stop, target, cost_points=0.0,
                     max_hold_minutes=400, allow_overlap=True)
        per_trade = result.stats.net_points / result.stats.trades
        # Zero within noise: one trade's spread is ~the target size.
        assert abs(per_trade) < 0.15 * (stop + target)

    def test_the_win_rate_matches_optional_stopping(self, minutes):
        signals = [(minutes[i].ts, LONG, minutes[i].open) for i in range(0, 5000, 20)]
        result = run(signals, minutes, 5.0, 25.0, cost_points=0.0,
                     max_hold_minutes=800, allow_overlap=True)
        reasons = result.by_reason()
        resolved = reasons[TARGET] + reasons[STOP]
        rate = reasons[TARGET] / resolved
        assert abs(rate - theoretical_win_rate(5.0, 25.0)) < 0.08

    def test_shorts_are_symmetric(self, minutes):
        signals = [(minutes[i].ts, SHORT, minutes[i].open) for i in range(0, 5000, 20)]
        result = run(signals, minutes, 10.0, 10.0, cost_points=0.0,
                     max_hold_minutes=400, allow_overlap=True)
        assert abs(result.stats.net_points / result.stats.trades) < 2.0


class TestRun:
    def test_overlapping_signals_are_skipped_by_default(self):
        """One position at a time, like the live trader. Counting them all
        assumes unlimited capital and double-counts the same market move."""
        minutes = flat_walk(600, seed=1)
        signals = [(m.ts, LONG, m.open) for m in minutes[:50]]
        result = run(signals, minutes, 0.0, 0.0, cost_points=0.0,
                     max_hold_minutes=10)
        assert result.skipped_overlap > 0
        assert len(result.trades) < len(signals)

    def test_overlap_can_be_allowed(self):
        minutes = flat_walk(600, seed=1)
        signals = [(m.ts, LONG, m.open) for m in minutes[:50]]
        result = run(signals, minutes, 0.0, 0.0, cost_points=0.0,
                     max_hold_minutes=10, allow_overlap=True)
        assert result.skipped_overlap == 0
        assert len(result.trades) == 50

    def test_a_signal_past_the_end_of_the_data_is_dropped(self):
        minutes = flat_walk(100)
        result = run([(minutes[-1].ts + 99_999, LONG, 100.0)], minutes, 5.0, 25.0)
        assert result.trades == []

    def test_the_hold_is_bounded(self):
        minutes = [minute(i, 100.0, 100.1, 99.9, 100.0) for i in range(500)]
        result = run([(minutes[0].ts, LONG, 100.0)], minutes, 50.0, 50.0,
                     cost_points=0.0, max_hold_minutes=30)
        assert result.trades[0].minutes_held == 30
        assert result.trades[0].exit_reason == TIMEOUT

    def test_exit_reasons_are_counted(self):
        minutes = flat_walk(2000, seed=9)
        signals = [(minutes[i].ts, LONG, minutes[i].open) for i in range(0, 1500, 50)]
        reasons = run(signals, minutes, 5.0, 25.0, cost_points=0.0,
                      max_hold_minutes=200, allow_overlap=True).by_reason()
        assert sum(reasons.values()) == len(signals)
        assert reasons[STOP] > reasons[TARGET]  # 1:5 geometry, by construction

    def test_no_minutes_yields_nothing(self):
        assert run([(BASE, LONG, 100.0)], [], 5.0, 25.0).trades == []
