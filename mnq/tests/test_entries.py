import random

import pytest

from app.core.candle import shapes
from app.core.entries import (
    LONG,
    SHORT,
    SUSTAINED,
    TOUCH,
    disagree,
    fill_profile,
    minutes_within,
    offset_stats,
    profiles,
    recommend,
    simulate_entry,
    sustained_bars,
)
from app.models import Bar

STEP = 300
BASE = 1_700_000_000 // STEP * STEP


def bar(ts, o, h, l, c, complete=True):
    return Bar(ts=ts, open=o, high=h, low=l, close=c, volume=10, count=5,
               complete=complete)


def walk(n, tick_sigma=0.35, seed=11, ticks=120, drift=0.0, start=19000.0):
    rng = random.Random(seed)
    price = start
    out = []
    for i in range(n):
        o = price
        hi = lo = price
        for _ in range(ticks):
            price += rng.gauss(drift / ticks, tick_sigma)
            hi, lo = max(hi, price), min(lo, price)
        out.append(bar(BASE + i * STEP, o, hi, lo, price))
    return out


class TestFillMechanics:
    def test_market_entry_always_fills(self):
        p = fill_profile(walk(200), offsets=[0.0])[0]
        assert p.fills == p.attempts

    def test_deeper_limits_fill_less_often(self):
        rates = [p.fill_rate for p in fill_profile(walk(400))]
        assert rates == sorted(rates, reverse=True)

    def test_a_long_fills_on_the_down_excursion(self):
        """Long limits sit below the open, so a bar with no lower wick at all
        cannot fill one."""
        bars = [bar(BASE + i * STEP, 100.0, 105.0, 100.0, 104.0) for i in range(80)]
        p = fill_profile(bars, offsets=[0.5], direction=LONG)[0]
        assert p.fills == 0

    def test_a_short_fills_on_the_up_excursion(self):
        bars = [bar(BASE + i * STEP, 100.0, 100.0, 95.0, 96.0) for i in range(80)]
        assert fill_profile(bars, offsets=[0.5], direction=SHORT)[0].fills == 0

    def test_unfilled_attempts_still_count(self):
        """Skipping them is what makes 'per fill' a mirage."""
        p = fill_profile(walk(400), offsets=[1.0])[0]
        assert p.fills < p.attempts
        assert p.per_attempt == pytest.approx(
            p.per_fill * p.fills / p.attempts, abs=1e-9
        )


class TestNoFreeLunch:
    """The control that keeps this module honest.

    A driftless random walk has, by optional stopping, exactly zero expectancy
    for "buy when price touches L, sell at the close" — the walk restarts at L
    with no drift. Any edge reported here is a bug in the arithmetic, not an
    effect in the data, and this test is what would catch that.
    """

    @classmethod
    @pytest.fixture(scope="class")
    def history(cls):
        return shapes(walk(3000, seed=7))

    @pytest.mark.parametrize("offset", [0.25, 0.5, 1.0])
    def test_a_random_walk_yields_no_edge(self, history, offset):
        p = offset_stats(history, offset, LONG, cost_points=0.0, sigma=1.0)
        assert p.fills > 200
        assert abs(p.per_fill) < 0.06  # in sigma units

    @pytest.mark.parametrize("offset", [0.25, 0.5, 1.0])
    def test_adverse_selection_cancels_the_price_improvement(self, history, offset):
        """This is the whole reason waiting for a better price is not free."""
        p = offset_stats(history, offset, LONG, cost_points=0.0, sigma=1.0)
        assert p.adverse_selection == pytest.approx(-offset, abs=0.08)

    def test_costs_come_straight_off_the_result(self, history):
        free = offset_stats(history, 0.5, LONG, cost_points=0.0, sigma=2.0)
        paid = offset_stats(history, 0.5, LONG, cost_points=1.0, sigma=2.0)
        # 1.0 points of cost is 0.5 sigma when sigma is 2.0
        assert paid.per_fill == pytest.approx(free.per_fill - 0.5, abs=1e-9)

    def test_shorts_are_symmetric(self, history):
        long_ = offset_stats(history, 0.5, LONG, cost_points=0.0, sigma=1.0)
        short = offset_stats(history, 0.5, SHORT, cost_points=0.0, sigma=1.0)
        assert abs(long_.per_fill - short.per_fill) < 0.08


class TestSustainedFills:
    def _bars_and_minutes(self, spike_low):
        """One 5-minute bar whose low is a single-minute spike.

        The parent opens above every minute close so the re-cut low comes from
        the minute closes rather than from the open, which always bounds it.
        """
        parent = bar(BASE, 100.6, 102.0, spike_low, 101.0)
        minutes = [
            bar(BASE + 0, 100.6, 100.7, 100.4, 100.4),
            bar(BASE + 60, 100.4, 100.4, spike_low, 100.2),  # spike, recovers
            bar(BASE + 120, 100.2, 100.6, 100.1, 100.5),
            bar(BASE + 180, 100.5, 101.2, 100.4, 101.0),
            bar(BASE + 240, 101.0, 102.0, 100.9, 101.0),
        ]
        return parent, minutes

    def test_a_wick_low_is_not_a_sustained_low(self):
        parent, minutes = self._bars_and_minutes(spike_low=95.0)
        out = sustained_bars([parent], minutes, STEP)[0]
        assert parent.low == 95.0  # a single tick nobody could rely on
        assert out.low == 100.2  # the lowest minute *close*, not the spike

    def test_open_and_close_still_bound_the_bar(self):
        """Those two prices certainly traded, so they can never be cut away."""
        parent = bar(BASE, 100.0, 110.0, 90.0, 108.0)
        minutes = [bar(BASE + 60 * i, 100.0, 100.1, 99.9, 100.0) for i in range(5)]
        out = sustained_bars([parent], minutes, STEP)[0]
        assert out.low <= 100.0 and out.high >= 108.0

    def test_the_minute_series_itself_is_untouched(self):
        """A 1-minute bar has no sub-minute path; re-cutting it to its own close
        deletes both wicks and yields a different instrument, not a stricter
        fill model."""
        bars = walk(50)
        assert sustained_bars(bars, bars, 60) == bars

    def test_no_conservative_profile_below_minute_resolution(self):
        bars = walk(120)
        assert SUSTAINED not in profiles(bars, minutes=bars, step_seconds=60)

    def test_both_models_are_reported_when_possible(self):
        bars = walk(120)
        minutes = walk(600, ticks=24, seed=11)
        both = profiles(bars, minutes=minutes, step_seconds=STEP)
        assert set(both) == {TOUCH, SUSTAINED}
        assert all(p.fill_model == SUSTAINED for p in both[SUSTAINED])

    def test_sign_disagreements_are_surfaced(self):
        optimistic = [offset_stats([], 0.5, LONG)]
        optimistic[0].per_attempt = 2.0
        conservative = [offset_stats([], 0.5, LONG)]
        conservative[0].per_attempt = -1.0
        assert disagree({TOUCH: optimistic, SUSTAINED: conservative}) == [0.5]

    def test_agreement_is_not_flagged(self):
        a = [offset_stats([], 0.5, LONG)]
        a[0].per_attempt = 2.0
        b = [offset_stats([], 0.5, LONG)]
        b[0].per_attempt = 1.0
        assert disagree({TOUCH: a, SUSTAINED: b}) == []


class TestRecommendation:
    def test_an_edgeless_series_recommends_nothing(self):
        rec = recommend(walk(1500, seed=13), cost_points=0.75)
        assert rec.best is None
        assert "no entry depth has positive expectancy" in rec.reason

    def test_it_names_the_signal_not_the_entry(self):
        """Entry placement cannot rescue a call with no edge, and saying so is
        more useful than ranking the least-bad depth."""
        rec = recommend(walk(1500, seed=13), cost_points=0.75)
        assert "direction call is what needs fixing" in rec.reason

    def test_thin_depths_are_never_selected(self):
        rec = recommend(walk(400, seed=2), offsets=[0.0, 6.0], min_fills=30)
        assert rec.best is None or rec.best.offset == 0.0

    def test_it_prefers_the_conservative_model(self):
        rec = recommend(walk(300), minutes=walk(1500, ticks=24, seed=11),
                        step_seconds=STEP)
        assert rec.fill_model == SUSTAINED

    def test_it_falls_back_when_no_minutes_are_given(self):
        assert recommend(walk(300)).fill_model == TOUCH

    def test_touch_only_is_never_trustworthy(self):
        """Without a finer series there is nothing to check the fills against."""
        assert not recommend(walk(300)).trustworthy

    def test_a_trending_series_favours_the_open(self):
        """When price runs, waiting for a pullback misses the move entirely."""
        rec = recommend(walk(1200, seed=6, drift=6.0), cost_points=0.0)
        assert rec.best is not None
        assert rec.best.offset == 0.0


class TestPathResolution:
    def _minutes(self, prices):
        return [
            bar(BASE + 60 * i, p, p + 0.5, p - 0.5, p)
            for i, p in enumerate(prices)
        ]

    def test_minutes_are_selected_by_the_parent_window(self):
        mins = self._minutes([100.0] * 10)
        inner = minutes_within(mins, BASE, BASE + STEP)
        assert len(inner) == 5

    def test_an_unreached_limit_does_not_fill(self):
        parent = bar(BASE, 100.0, 101.0, 99.5, 100.5)
        out = simulate_entry(parent, self._minutes([100.0] * 5), LONG,
                             offset_points=10.0)
        assert not out.filled and out.exit_reason == "unfilled"
        assert out.net_points == 0.0

    def test_a_market_order_fills_at_the_open(self):
        parent = bar(BASE, 100.0, 102.0, 99.0, 101.0)
        out = simulate_entry(parent, self._minutes([100.0, 100.5, 101.0, 101.0, 101.0]),
                             LONG, offset_points=0.0, cost_points=0.0)
        assert out.entry == 100.0 and out.exit_reason == "close"
        assert out.net_points == pytest.approx(1.0)

    def test_a_stop_is_taken_before_a_target_in_the_same_minute(self):
        """1-minute data cannot say which came first, and the pessimistic
        reading is the one that will not flatter a strategy."""
        parent = bar(BASE, 100.0, 105.0, 95.0, 100.0)
        minutes = [bar(BASE, 100.0, 105.0, 95.0, 100.0)]
        out = simulate_entry(parent, minutes, LONG, offset_points=0.0,
                             stop_points=2.0, target_points=2.0, cost_points=0.0)
        assert out.exit_reason == "stop"
        assert out.net_points == pytest.approx(-2.0)

    def test_a_target_is_taken_when_no_stop_is_hit(self):
        parent = bar(BASE, 100.0, 105.0, 99.8, 100.0)
        minutes = self._minutes([100.0, 101.0, 103.0, 100.5, 100.0])
        out = simulate_entry(parent, minutes, LONG, offset_points=0.0,
                             stop_points=5.0, target_points=2.0, cost_points=0.0)
        assert out.exit_reason == "target"
        assert out.net_points == pytest.approx(2.0)

    def test_a_gap_through_the_limit_fills_at_the_open(self):
        """Price traded past your resting order; that is the fill you get."""
        parent = bar(BASE, 100.0, 100.5, 90.0, 99.0)
        minutes = self._minutes([100.0, 92.0, 95.0, 98.0, 99.0])
        out = simulate_entry(parent, minutes, LONG, offset_points=5.0,
                             cost_points=0.0)
        assert out.entry == 92.0  # not 95.0, the limit

    def test_a_short_stops_out_upward(self):
        parent = bar(BASE, 100.0, 106.0, 99.0, 105.0)
        minutes = self._minutes([100.0, 101.0, 103.0, 105.0, 105.0])
        out = simulate_entry(parent, minutes, SHORT, offset_points=0.0,
                             stop_points=2.0, cost_points=0.0)
        assert out.exit_reason == "stop"
        assert out.net_points == pytest.approx(-2.0)

    def test_costs_are_charged_once(self):
        parent = bar(BASE, 100.0, 102.0, 99.0, 101.0)
        minutes = self._minutes([100.0, 100.5, 101.0, 101.0, 101.0])
        out = simulate_entry(parent, minutes, LONG, offset_points=0.0,
                             cost_points=0.75)
        assert out.net_points == pytest.approx(1.0 - 0.75)

    def test_holding_time_counts_from_the_fill(self):
        parent = bar(BASE, 100.0, 102.0, 97.0, 101.0)
        minutes = self._minutes([100.0, 100.0, 97.5, 100.0, 101.0])
        out = simulate_entry(parent, minutes, LONG, offset_points=2.0,
                             cost_points=0.0)
        assert out.filled and out.minutes_held == 3  # filled on minute 3 of 5

    def test_it_falls_back_to_the_parent_bar_without_a_path(self):
        parent = bar(BASE, 100.0, 102.0, 98.0, 101.0)
        out = simulate_entry(parent, [], LONG, offset_points=1.0, cost_points=0.0)
        assert out.filled and out.entry == 99.0
        assert out.net_points == pytest.approx(2.0)
