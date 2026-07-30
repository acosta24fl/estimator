"""Feed tests.

The Yahoo tests drive the parser with recorded response shapes rather than the
network, so the awkward cases (null gaps, missing legs, empty results) are
covered deterministically and offline.
"""

import asyncio
import time

import pytest

from app.config import Settings, load_settings
from app.feed import available, create
from app.feed.synthetic import SyntheticFeed
from app.feed.yahoo import YahooError, YahooFeed


_DEFAULT_META = {"regularMarketPrice": 20_000.0, "currency": "USD"}


def yahoo_payload(timestamps, opens, highs, lows, closes, volumes, meta=None):
    return {
        # `meta={}` must stay empty — it is the "feed sent no quote" case.
        "meta": _DEFAULT_META if meta is None else meta,
        "timestamp": timestamps,
        "indicators": {
            "quote": [
                {
                    "open": opens,
                    "high": highs,
                    "low": lows,
                    "close": closes,
                    "volume": volumes,
                }
            ]
        },
    }


class TestRegistry:
    def test_both_feeds_registered(self):
        assert set(available()) >= {"yahoo", "synthetic"}

    def test_create_selects_by_key(self):
        feed = create(Settings(feed="synthetic"))
        assert isinstance(feed, SyntheticFeed)

    def test_unknown_feed_is_rejected(self):
        with pytest.raises(KeyError):
            create(Settings(feed="nope"))


class TestYahooParsing:
    feed = YahooFeed(load_settings())

    def test_parses_a_normal_response(self):
        now = int(time.time()) // 60 * 60
        ts = [now - 180, now - 120, now - 60]
        payload = yahoo_payload(
            ts, [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [0.5, 1.5, 2.5],
            [1.2, 2.2, 3.2], [10, 20, 30],
        )
        snap = self.feed._parse(payload, bar_seconds=60)
        assert [b.ts for b in snap.bars] == ts
        assert snap.bars[0].open == 1.0
        assert snap.bars[-1].close == 3.2
        assert snap.bars[1].volume == 20

    def test_null_close_rows_are_dropped(self):
        now = int(time.time()) // 60 * 60
        ts = [now - 180, now - 120, now - 60]
        payload = yahoo_payload(
            ts, [1.0, None, 3.0], [1.5, None, 3.5], [0.5, None, 2.5],
            [1.2, None, 3.2], [10, None, 30],
        )
        snap = self.feed._parse(payload, bar_seconds=60)
        assert [b.ts for b in snap.bars] == [ts[0], ts[2]]

    def test_missing_open_falls_back_to_close(self):
        now = int(time.time()) // 60 * 60
        payload = yahoo_payload([now - 60], [None], [None], [None], [5.0], [7])
        bar = self.feed._parse(payload, bar_seconds=60).bars[0]
        assert bar.open == bar.high == bar.low == bar.close == 5.0

    def test_high_low_always_bracket_open_and_close(self):
        now = int(time.time()) // 60 * 60
        # A feed glitch where the reported high is below the close.
        payload = yahoo_payload([now - 60], [10.0], [9.0], [11.0], [12.0], [1])
        bar = self.feed._parse(payload, bar_seconds=60).bars[0]
        assert bar.high >= max(bar.open, bar.close)
        assert bar.low <= min(bar.open, bar.close)

    def test_null_volume_becomes_zero(self):
        now = int(time.time()) // 60 * 60
        payload = yahoo_payload([now - 60], [1.0], [1.0], [1.0], [1.0], [None])
        assert self.feed._parse(payload, bar_seconds=60).bars[0].volume == 0.0

    def test_last_bar_is_incomplete_until_its_period_elapses(self):
        now = int(time.time())
        forming = now // 60 * 60  # started this minute
        closed = forming - 60
        payload = yahoo_payload(
            [closed, forming], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1, 1]
        )
        bars = self.feed._parse(payload, bar_seconds=60).bars
        assert bars[0].complete
        assert not bars[1].complete

    def test_bars_are_sorted(self):
        now = int(time.time()) // 60 * 60
        payload = yahoo_payload(
            [now - 60, now - 180], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1, 1]
        )
        bars = self.feed._parse(payload, bar_seconds=60).bars
        assert [b.ts for b in bars] == sorted(b.ts for b in bars)

    def test_quote_prefers_meta_price(self):
        now = int(time.time()) // 60 * 60
        payload = yahoo_payload(
            [now - 60], [1.0], [1.0], [1.0], [1.0], [1],
            meta={"regularMarketPrice": 21_000.5, "previousClose": 20_900.0,
                  "currency": "USD", "fullExchangeName": "CME"},
        )
        quote = self.feed._parse(payload, bar_seconds=60).quote
        assert quote.price == 21_000.5
        assert quote.prev_close == 20_900.0
        assert quote.exchange == "CME"

    def test_quote_falls_back_to_last_close(self):
        now = int(time.time()) // 60 * 60
        payload = yahoo_payload([now - 60], [1.0], [1.0], [1.0], [4.5], [1], meta={})
        assert self.feed._parse(payload, bar_seconds=60).quote.price == 4.5

    def test_empty_payload_yields_nothing(self):
        snap = self.feed._parse({"meta": {}, "timestamp": [], "indicators": {}}, 60)
        assert snap.bars == []
        assert snap.quote is None


class TestSynthetic:
    feed = SyntheticFeed(load_settings())

    def test_repeated_polls_return_identical_closed_bars(self):
        """The append-only log depends on closed bars never being revised."""
        first = asyncio.run(self.feed.fetch_intraday()).bars
        second = asyncio.run(self.feed.fetch_intraday()).bars
        second_by_ts = {b.ts: b for b in second}
        for bar in first:
            other = second_by_ts.get(bar.ts)
            if other is None or not bar.complete:
                continue
            assert (bar.open, bar.high, bar.low, bar.close, bar.volume) == (
                other.open, other.high, other.low, other.close, other.volume
            )

    def test_series_is_continuous(self):
        bars = asyncio.run(self.feed.fetch_intraday()).bars
        assert all(bars[i].close == bars[i + 1].open for i in range(len(bars) - 1))

    def test_bars_are_one_minute_apart(self):
        bars = asyncio.run(self.feed.fetch_intraday()).bars
        assert all(bars[i + 1].ts - bars[i].ts == 60 for i in range(len(bars) - 1))

    def test_ohlc_is_internally_consistent(self):
        bars = asyncio.run(self.feed.fetch_intraday()).bars
        for bar in bars:
            assert bar.high >= max(bar.open, bar.close)
            assert bar.low <= min(bar.open, bar.close)

    def test_prices_are_on_the_quarter_point_tick(self):
        bars = asyncio.run(self.feed.fetch_intraday()).bars[:200]
        for bar in bars:
            for price in (bar.open, bar.high, bar.low, bar.close):
                assert abs(price * 4 - round(price * 4)) < 1e-9

    def test_daily_returns_a_long_history(self):
        bars = asyncio.run(self.feed.fetch_daily()).bars
        assert len(bars) > 300
