import random

from app.core.profiling import (
    acceleration_profile,
    continuation_profile,
    hourly_profile,
    returns,
    scaling_profile,
    volatility_profile,
    wick_profile,
)
from app.core.stats import ols_slope, significant, wilson_interval
from app.models import Bar


def walk(n=2000, sd=3.0, persist=0.0, seed=1, wick_mult=0.6, step=60):
    random.seed(seed)
    price, prev, out = 20_000.0, 0.0, []
    for i in range(n):
        r = persist * prev + random.gauss(0, sd)
        o, c = price, price + r
        w = abs(random.gauss(0, sd * wick_mult))
        out.append(Bar(ts=i * step, open=o, high=max(o, c) + w, low=min(o, c) - w,
                       close=c, volume=100, complete=True))
        price, prev = c, r
    return out


def resample(bars, k):
    out = []
    for i in range(0, len(bars) - k + 1, k):
        g = bars[i:i + k]
        out.append(Bar(ts=g[0].ts, open=g[0].open, high=max(b.high for b in g),
                       low=min(b.low for b in g), close=g[-1].close,
                       volume=0, complete=True))
    return out


class TestStatsHelpers:
    def test_wilson_brackets_the_estimate(self):
        low, high = wilson_interval(60, 100)
        assert low < 0.6 < high

    def test_significance_needs_the_interval_to_clear_the_reference(self):
        assert not significant((0.45, 0.55))
        assert significant((0.55, 0.65))

    def test_ols_slope_recovers_a_known_line(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        assert abs(ols_slope(xs, [3 * x + 7 for x in xs]) - 3.0) < 1e-9


class TestContinuation:
    def test_a_random_walk_is_a_coin_flip(self):
        c = continuation_profile(walk(4000, seed=3))
        assert c.ci[0] <= 0.5 <= c.ci[1]

    def test_a_trending_series_is_detected(self):
        c = continuation_profile(walk(4000, persist=0.3, seed=3))
        assert c.same_direction_rate > 0.55
        assert c.return_autocorr > 0.15
        assert c.significant

    def test_a_mean_reverting_series_is_detected(self):
        c = continuation_profile(walk(4000, persist=-0.3, seed=3))
        assert c.same_direction_rate < 0.45
        assert c.return_autocorr < -0.15
        assert c.significant

    def test_too_little_data_reports_nothing(self):
        assert continuation_profile(walk(10)).same_direction_rate is None


class TestScaling:
    def test_a_random_walk_has_a_square_root_exponent(self):
        base = walk(20000, seed=5)
        s = scaling_profile({k: resample(base, k) for k in (1, 5, 10, 15, 30, 60)})
        assert 0.45 <= s.exponent <= 0.55
        assert "random walk" in s.interpretation

    def test_a_trending_series_scales_faster(self):
        base = walk(20000, persist=0.3, seed=5)
        s = scaling_profile({k: resample(base, k) for k in (1, 5, 10, 15, 30, 60)})
        assert s.exponent > 0.55

    def test_a_mean_reverting_series_scales_slower(self):
        base = walk(20000, persist=-0.3, seed=5)
        s = scaling_profile({k: resample(base, k) for k in (1, 5, 10, 15, 30, 60)})
        assert s.exponent < 0.45

    def test_too_few_timeframes(self):
        assert scaling_profile({1: walk(100)}).exponent == 0.0


class TestAcceleration:
    def test_constant_volatility_shows_no_expansion_signal(self):
        a = acceleration_profile(walk(4000, seed=7))
        assert a.big_then_big_ci[0] <= 0.5 <= a.big_then_big_ci[1]

    def test_clustered_volatility_is_detected(self):
        random.seed(11)
        price, out = 20_000.0, []
        for i in range(4000):
            sd = 12.0 if (i // 50) % 2 else 2.0
            r = random.gauss(0, sd)
            o, c = price, price + r
            w = abs(random.gauss(0, sd * 0.6))
            out.append(Bar(ts=i * 60, open=o, high=max(o, c) + w, low=min(o, c) - w,
                           close=c, volume=100, complete=True))
            price = c
        a = acceleration_profile(out)
        assert a.big_then_big_rate > 0.55
        assert a.significant


class TestWick:
    def test_shape_fractions_sum_to_one(self):
        w = wick_profile(walk(2000, seed=13))
        total = w.body_fraction + w.upper_fraction + w.lower_fraction
        assert abs(total - 1.0) < 1e-9

    def test_symmetric_bars_have_balanced_wicks(self):
        w = wick_profile(walk(4000, seed=13))
        assert abs(w.upper_fraction - w.lower_fraction) < 0.02

    def test_no_signal_on_a_random_walk(self):
        w = wick_profile(walk(4000, seed=13, wick_mult=2.5), dominance=0.4)
        assert w.lower_samples > 20, "the test must actually exercise the predictor"
        assert not w.after_long_lower_significant
        assert not w.after_long_upper_significant

    def test_the_two_wick_tests_are_reported_separately(self):
        """OR-ing them would double the false-positive rate."""
        w = wick_profile(walk(4000, seed=13, wick_mult=2.5), dominance=0.4)
        assert w.after_long_lower_ci is not None
        assert w.after_long_upper_ci is not None

    def test_too_little_data(self):
        assert wick_profile(walk(5)).bars < 30


class TestVolatilityProfile:
    def test_reports_ordered_percentiles(self):
        v = volatility_profile(walk(2000, seed=17))
        assert v.range_p10 <= v.median_range <= v.range_p90

    def test_a_wilder_series_measures_wider(self):
        assert (volatility_profile(walk(2000, sd=10.0, seed=17)).mean_range
                > volatility_profile(walk(2000, sd=1.0, seed=17)).mean_range)


class TestHourly:
    def test_buckets_cover_the_clock(self):
        # 60-second bars over 3000 minutes span more than a full day.
        hours = hourly_profile(walk(3000, seed=19), "America/New_York")
        assert len(hours) == 24
        assert all(h.bars > 0 for h in hours)

    def test_drift_is_not_significant_on_a_random_walk(self):
        hours = hourly_profile(walk(3000, seed=19), "America/New_York")
        assert sum(h.drift_significant for h in hours) <= 3  # ~5% of 24 buckets

    def test_returns_helper(self):
        bars = walk(5, seed=1)
        assert len(returns(bars)) == 4


class TestGapAwareness:
    """A weekend gap is not a bar-sized move and must never be counted as one."""

    def test_contiguous_runs_split_on_a_gap(self):
        from app.core.profiling import contiguous_runs

        bars = walk(50, seed=21)
        jumped = bars[:25] + [
            Bar(ts=b.ts + 100_000, open=b.open, high=b.high, low=b.low,
                close=b.close, volume=b.volume, complete=True)
            for b in bars[25:]
        ]
        runs = contiguous_runs(jumped)
        assert len(runs) == 2
        assert len(runs[0]) == 25

    def test_returns_exclude_the_gap(self):
        bars = walk(50, seed=21)
        jumped = bars[:25] + [
            Bar(ts=b.ts + 100_000, open=b.open, high=b.high, low=b.low,
                close=b.close, volume=b.volume, complete=True)
            for b in bars[25:]
        ]
        # 50 bars in two runs of 25 give 24 + 24 returns, not 49.
        assert len(returns(jumped)) == 48

    def test_hourly_drift_is_not_inflated_by_cross_day_diffs(self):
        """The bug this guards: bucketing bars by hour, then diffing inside the
        bucket, subtracts yesterday's close from today's."""
        bars = walk(3000, seed=23, step=300)  # 5-minute bars over ~10 days
        true_mean = sum(returns(bars)) / len(returns(bars))
        hours = hourly_profile(bars, "America/New_York")
        for h in hours:
            assert abs(h.mean_move) < abs(true_mean) + 5 * h.move_stderr + 1.0
