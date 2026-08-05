import math
import random

from app.core.candle import (
    DEFAULT_QUANTILE,
    calibrate,
    evaluate,
    gap_estimate,
    predict,
    shapes,
)
from app.config import load_settings
from app.core.timeframes import SessionBucket, get
from app.indicators.base import IndicatorContext
from app.indicators.next_candle import NextCandle
from app.models import Bar

STEP = 300
BASE = 1_700_000_000 // STEP * STEP


def bar(i, o, h, l, c, complete=True):
    return Bar(ts=BASE + i * STEP, open=o, high=h, low=l, close=c,
               volume=100, count=5, complete=complete)


def walk(n, tick_sigma=0.35, seed=11, ticks=120, start=19000.0):
    """OHLC sampled from an actual driftless random walk."""
    rng = random.Random(seed)
    price = start
    out = []
    for i in range(n):
        o = price
        hi = lo = price
        for _ in range(ticks):
            price += rng.gauss(0.0, tick_sigma)
            hi, lo = max(hi, price), min(lo, price)
        out.append(bar(i, o, hi, lo, price))
    return out


class TestShapes:
    def test_the_first_bar_is_dropped(self):
        """Standardising a bar by its own range would leak its range into its
        own scale, so there is nothing to compare it against."""
        assert len(shapes(walk(20))) == 19

    def test_excursions_are_non_negative(self):
        for s in shapes(walk(60)):
            assert s.up >= 0 and s.down >= 0

    def test_uses_only_prior_volatility(self):
        """A shape must not change when a *later* bar is appended."""
        bars = walk(80)
        early = shapes(bars[:60])
        late = shapes(bars)
        assert [round(s.up, 9) for s in early] == [round(s.up, 9) for s in late[:59]]

    def test_incomplete_bars_are_excluded(self):
        bars = walk(30)
        bars.append(bar(30, 100.0, 110.0, 90.0, 105.0, complete=False))
        assert all(s.ts != bars[-1].ts for s in shapes(bars))

    def test_a_flat_series_yields_nothing(self):
        flat = [bar(i, 100.0, 100.0, 100.0, 100.0) for i in range(30)]
        assert shapes(flat) == []  # zero volatility, nothing to divide by


class TestGap:
    def test_contiguous_bars_have_no_gap(self):
        bars = [bar(i, 100.0 + i, 101.0 + i, 99.0 + i, 100.0 + i) for i in range(20)]
        # each bar opens exactly where the last closed
        chained = []
        prev = 100.0
        for i in range(20):
            chained.append(bar(i, prev, prev + 2, prev - 2, prev + 1))
            prev += 1
        assert gap_estimate(chained) == 0.0

    def test_a_persistent_gap_is_measured(self):
        bars = []
        prev = 100.0
        for i in range(20):
            o = prev + 3.0  # every bar opens 3 points above the last close
            bars.append(bar(i, o, o + 1, o - 1, o))
            prev = o
        assert gap_estimate(bars) == 3.0


class TestPredict:
    def test_needs_history(self):
        f = predict(walk(10))
        assert not f.valid and "standardised bars" in f.reason

    def test_envelope_brackets_the_open(self):
        f = predict(walk(300))
        assert f.valid
        assert f.low < f.open < f.high

    def test_a_wider_quantile_gives_a_wider_envelope(self):
        bars = walk(300)
        assert predict(bars, quantile=0.95).predicted_range > \
               predict(bars, quantile=0.60).predicted_range

    def test_the_typical_range_is_inside_the_envelope(self):
        f = predict(walk(300), quantile=0.9)
        assert f.high > f.typical_high and f.low < f.typical_low

    def test_drift_moves_the_close_not_the_envelope(self):
        bars = walk(300)
        flat = predict(bars)
        pushed = predict(bars, drift=5.0)
        assert pushed.close > flat.close
        assert (pushed.high, pushed.low) == (flat.high, flat.low)

    def test_the_close_never_leaves_its_own_band(self):
        """A drift larger than the envelope would draw a close outside the band
        it is supposed to live in."""
        f = predict(walk(300), drift=100_000.0)
        assert f.low <= f.close <= f.high

    def test_the_predicted_bar_is_one_step_ahead(self):
        f = predict(walk(300), step_seconds=STEP)
        assert f.ts == BASE + 299 * STEP + STEP

    def test_volatility_scales_the_envelope(self):
        calm = predict(walk(300, tick_sigma=0.2, seed=3))
        wild = predict(walk(300, tick_sigma=1.0, seed=3))
        assert wild.predicted_range > 2 * calm.predicted_range


class TestCalibration:
    """The envelope's whole claim is its coverage, so that is what is tested."""

    def test_coverage_tracks_the_requested_quantile(self):
        acc = evaluate(walk(1200, seed=5), quantile=0.80)
        assert acc.bars > 500
        assert abs(acc.high_coverage - 0.80) < 0.08
        assert abs(acc.low_coverage - 0.80) < 0.08

    def test_a_tight_quantile_covers_less(self):
        bars = walk(1200, seed=5)
        assert evaluate(bars, quantile=0.5).contained < evaluate(bars, quantile=0.9).contained

    def test_coverage_is_monotone_in_the_quantile(self):
        rows = calibrate(walk(1200, seed=9), candidates=(0.5, 0.7, 0.9))
        covers = [r.high_coverage for r in rows]
        assert covers == sorted(covers)

    def test_it_beats_predicting_the_last_bars_range(self):
        """Volatility clusters, but a single previous bar is a noisy estimate of
        it; an EWMA over many should do better."""
        acc = evaluate(walk(1500, seed=21), quantile=0.5)
        assert acc.naive_skill > 0

    def test_too_little_history_scores_nothing(self):
        assert evaluate(walk(30)).bars == 0

    def test_evaluation_is_walk_forward(self):
        """Scoring must not change when bars *after* the scored window arrive."""
        bars = walk(900, seed=4)
        short = evaluate(bars[:600], quantile=0.8)
        long = evaluate(bars, quantile=0.8)
        assert long.bars > short.bars  # more bars scored, same early ones
        assert short.bars > 0


class TestChartRendering:
    """The indicator that draws the envelope on whichever timeframe is shown."""

    @staticmethod
    def _ctx(bars, tf_key="5m", outlook=None, minutes=()):
        return IndicatorContext(
            timeframe=get(tf_key),
            bars=list(bars),
            minute_bars=list(minutes),
            daily_bars=[],
            session=SessionBucket("America/New_York", 18),
            settings=load_settings(),
            now=bars[-1].ts + 1 if bars else 0,
            outlook=outlook,
        )

    def test_the_envelope_brackets_the_candles(self):
        result = NextCandle().compute(self._ctx(walk(300)))
        highs = [p["value"] for p in result.series["next_high"]]
        lows = [p["value"] for p in result.series["next_low"]]
        assert min(highs) > max(lows)

    def test_each_level_is_drawn_flat(self):
        """A level, not a series with shape of its own."""
        result = NextCandle().compute(self._ctx(walk(300)))
        assert len(set(p["value"] for p in result.series["next_high"])) == 1

    def test_it_extends_one_bar_into_the_future(self):
        bars = walk(300)
        result = NextCandle().compute(self._ctx(bars))
        assert result.series["next_high"][-1]["time"] == bars[-1].ts + 300

    def test_thin_history_reports_rather_than_crashes(self):
        result = NextCandle().compute(self._ctx(walk(10)))
        assert result.series["next_high"] == []
        assert result.stats[0].value == "—"

    def test_no_call_means_no_entry_advice(self):
        stats = {s.key: s.value for s in NextCandle().compute(self._ctx(walk(300))).stats}
        assert stats["next_entry"] == "no call"

    def test_a_bullish_call_produces_an_entry(self):
        class Call:
            direction = "bullish"

        stats = {s.key: s.value
                 for s in NextCandle().compute(self._ctx(walk(300), outlook=Call())).stats}
        assert "next_entry" in stats and stats["next_entry"] != "no call"
        assert "next_stop" in stats and "next_target" in stats

    def test_an_unconfirmed_entry_is_flagged(self):
        """Without a finer series to check fills against, the advice is a fill
        assumption and has to say so."""
        class Call:
            direction = "bullish"

        stats = {s.key: s.value
                 for s in NextCandle().compute(self._ctx(walk(300), outlook=Call())).stats}
        assert stats.get("next_entry_warning") == "not confirmed"

    def test_the_recommendation_is_cached_per_bar(self):
        class Call:
            direction = "bullish"

        bars = walk(300)
        indicator = NextCandle()
        ctx = self._ctx(bars, outlook=Call())
        first = indicator._recommendation(ctx, 1, 300)
        assert indicator._recommendation(ctx, 1, 300) is first

    def test_a_new_bar_invalidates_the_cache(self):
        class Call:
            direction = "bullish"

        bars = walk(301)
        indicator = NextCandle()
        first = indicator._recommendation(self._ctx(bars[:-1], outlook=Call()), 1, 300)
        second = indicator._recommendation(self._ctx(bars, outlook=Call()), 1, 300)
        assert second is not first
