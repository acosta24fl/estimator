import pytest

from app.core.aggregator import AggregationCache, aggregate
from app.core.timeframes import get
from app.models import Bar

MINUTE = 60
# Aligned to an hour so every intraday timeframe starts on a bucket boundary.
BASE = 1_700_000_000 // 3600 * 3600


def minute_bars(count, start=BASE, price=20_000.0, complete=True):
    bars = []
    for i in range(count):
        open_ = price + i
        bars.append(
            Bar(
                ts=start + i * MINUTE,
                open=open_,
                high=open_ + 2,
                low=open_ - 1,
                close=open_ + 1,
                volume=10.0,
                count=1,
                complete=complete,
            )
        )
    return bars


class TestAggregate:
    def test_five_minutes_fold_into_one_bar(self):
        out = aggregate(minute_bars(5), get("5m"), now=BASE + 10 * MINUTE)
        assert len(out) == 1
        bar = out[0]
        assert bar.ts == BASE
        assert bar.open == 20_000.0  # first bar's open
        assert bar.close == 20_005.0  # last bar's close
        assert bar.high == 20_006.0  # max of highs
        assert bar.low == 19_999.0  # min of lows
        assert bar.volume == 50.0
        assert bar.count == 5
        assert bar.complete

    def test_partial_bucket_is_incomplete(self):
        out = aggregate(minute_bars(7), get("5m"), now=BASE + 7 * MINUTE)
        assert len(out) == 2
        assert out[0].complete
        assert not out[1].complete
        assert out[1].count == 2

    def test_incomplete_source_bar_keeps_bucket_open(self):
        bars = minute_bars(5)
        bars[-1].complete = False
        out = aggregate(bars, get("5m"), now=BASE + 10 * MINUTE)
        assert not out[0].complete

    def test_gaps_do_not_create_empty_bars(self):
        bars = minute_bars(2) + minute_bars(2, start=BASE + 30 * MINUTE)
        out = aggregate(bars, get("5m"), now=BASE + 60 * MINUTE)
        assert len(out) == 2  # only buckets with data
        assert out[1].ts == BASE + 30 * MINUTE

    def test_empty_input(self):
        assert aggregate([], get("5m")) == []

    @pytest.mark.parametrize("tf_key,expected", [("5m", 12), ("10m", 6), ("15m", 4), ("30m", 2)])
    def test_bucket_counts_over_an_hour(self, tf_key, expected):
        out = aggregate(minute_bars(60), get(tf_key), now=BASE + 120 * MINUTE)
        assert len(out) == expected

    def test_volume_and_count_are_conserved(self):
        bars = minute_bars(60)
        for tf_key in ("5m", "15m", "1h"):
            out = aggregate(bars, get(tf_key), now=BASE + 120 * MINUTE)
            assert sum(b.volume for b in out) == sum(b.volume for b in bars)
            assert sum(b.count for b in out) == 60

    def test_higher_timeframe_bounds_contain_lower(self):
        bars = minute_bars(120)
        fine = aggregate(bars, get("5m"), now=BASE + 200 * MINUTE)
        coarse = aggregate(bars, get("30m"), now=BASE + 200 * MINUTE)
        assert max(b.high for b in coarse) == max(b.high for b in fine)
        assert min(b.low for b in coarse) == min(b.low for b in fine)


class TestAggregationCache:
    def test_repeat_call_returns_identical_object(self):
        cache = AggregationCache()
        bars = minute_bars(30)
        first = cache.get(bars, get("5m"), now=BASE + 100 * MINUTE)
        second = cache.get(bars, get("5m"), now=BASE + 100 * MINUTE)
        assert first is second

    def test_changed_last_bar_invalidates(self):
        cache = AggregationCache()
        bars = minute_bars(30)
        first = cache.get(bars, get("5m"), now=BASE + 100 * MINUTE)
        bars[-1].close += 5
        second = cache.get(bars, get("5m"), now=BASE + 100 * MINUTE)
        assert first is not second
        assert second[-1].close == bars[-1].close

    def test_new_bar_invalidates(self):
        cache = AggregationCache()
        bars = minute_bars(30)
        cache.get(bars, get("5m"), now=BASE + 100 * MINUTE)
        bars.extend(minute_bars(1, start=BASE + 30 * MINUTE))
        out = cache.get(bars, get("5m"), now=BASE + 100 * MINUTE)
        assert sum(b.count for b in out) == 31
