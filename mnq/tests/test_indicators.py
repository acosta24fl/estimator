import math

from app.config import load_settings
from app.core.timeframes import SessionBucket, get
from app.indicators import all_indicators
from app.indicators.base import IndicatorContext, ema
from app.indicators.bar_size import BarSize
from app.indicators.macd import Macd
from app.indicators.swings import DailyStructure, find_pivots
from app.indicators.vwap import SessionVwap
from app.models import Bar

MINUTE = 60
SESSION = SessionBucket("America/New_York", 18)
SETTINGS = load_settings()


def ctx_for(bars, tf_key="5m", daily=None, minutes=None):
    return IndicatorContext(
        timeframe=get(tf_key),
        bars=bars,
        minute_bars=minutes or [],
        daily_bars=daily or [],
        session=SESSION,
        settings=SETTINGS,
        now=bars[-1].ts + 1 if bars else 0,
    )


def make_bars(closes, start=1_700_000_000, step=300, volume=100.0):
    start = start // step * step
    return [
        Bar(
            ts=start + i * step,
            open=c,
            high=c + 5,
            low=c - 5,
            close=c,
            volume=volume,
            complete=True,
        )
        for i, c in enumerate(closes)
    ]


class TestEma:
    def test_seeds_with_sma(self):
        values = [1, 2, 3, 4, 5]
        out = ema(values, 3)
        assert out[:2] == [None, None]
        assert out[2] == 2.0  # (1+2+3)/3

    def test_recurrence(self):
        values = [1, 2, 3, 4, 5]
        out = ema(values, 3)
        alpha = 2 / 4
        assert math.isclose(out[3], (4 - 2.0) * alpha + 2.0)

    def test_too_short_returns_all_none(self):
        assert ema([1, 2], 5) == [None, None]

    def test_constant_series_is_flat(self):
        out = ema([7.0] * 20, 5)
        assert all(v == 7.0 for v in out[4:])


class TestVwap:
    def test_equals_typical_price_when_volume_is_uniform(self):
        bars = make_bars([100.0, 110.0, 120.0])
        result = SessionVwap().compute(ctx_for(bars))
        typicals = [b.typical for b in bars]
        expected = sum(typicals) / len(typicals)
        assert math.isclose(result.series["vwap"][-1]["value"], round(expected, 2), abs_tol=0.01)

    def test_weights_by_volume(self):
        bars = make_bars([100.0, 200.0])
        bars[0].volume, bars[1].volume = 1.0, 9.0
        result = SessionVwap().compute(ctx_for(bars))
        expected = (bars[0].typical * 1 + bars[1].typical * 9) / 10
        assert math.isclose(result.series["vwap"][-1]["value"], round(expected, 2), abs_tol=0.01)

    def test_resets_at_session_boundary(self):
        # Two bars before the 18:00 ET roll, two after.
        open_ts = SESSION.start(1_700_000_000)
        roll = SESSION.end(open_ts)
        bars = [
            Bar(ts=roll - 600, open=100, high=100, low=100, close=100, volume=10, complete=True),
            Bar(ts=roll - 300, open=100, high=100, low=100, close=100, volume=10, complete=True),
            Bar(ts=roll, open=500, high=500, low=500, close=500, volume=10, complete=True),
        ]
        result = SessionVwap().compute(ctx_for(bars))
        # The new session's first bar restarts the accumulation at its own price.
        assert math.isclose(result.series["vwap"][-1]["value"], 500.0, abs_tol=0.01)

    def test_distance_stat_matches_close_minus_vwap(self):
        bars = make_bars([100.0, 110.0, 120.0])
        result = SessionVwap().compute(ctx_for(bars))
        vwap = result.series["vwap"][-1]["value"]
        distance = next(s for s in result.stats if s.key == "vwap_distance")
        assert math.isclose(distance.value, round(bars[-1].close - vwap, 2), abs_tol=0.02)

    def test_not_applicable_on_daily(self):
        bars = make_bars([100.0, 110.0], step=86400)
        result = SessionVwap().compute(ctx_for(bars, tf_key="1d"))
        assert result.series["vwap"] == []
        assert result.stats[0].value == "n/a on this timeframe"

    def test_zero_volume_falls_back_to_typical_mean(self):
        bars = make_bars([100.0, 200.0], volume=0.0)
        result = SessionVwap().compute(ctx_for(bars))
        expected = (bars[0].typical + bars[1].typical) / 2
        assert math.isclose(result.series["vwap"][-1]["value"], round(expected, 2), abs_tol=0.01)


class TestMacd:
    def test_matches_manual_ema_difference(self):
        closes = [float(100 + i) for i in range(60)]
        bars = make_bars(closes)
        result = Macd().compute(ctx_for(bars))
        fast = ema(closes, 12)
        slow = ema(closes, 26)
        assert math.isclose(
            result.series["macd"][-1]["value"], round(fast[-1] - slow[-1], 3), abs_tol=0.002
        )

    def test_histogram_is_macd_minus_signal(self):
        bars = make_bars([float(100 + (i % 7) * 3) for i in range(80)])
        result = Macd().compute(ctx_for(bars))
        macd = result.series["macd"][-1]["value"]
        signal = result.series["signal"][-1]["value"]
        hist = result.series["histogram"][-1]["value"]
        assert math.isclose(hist, round(macd - signal, 3), abs_tol=0.002)

    def test_breakout_higher_is_bullish(self):
        bars = make_bars([100.0] * 50 + [100.0 + i * 3 for i in range(1, 16)])
        state = next(s for s in Macd().compute(ctx_for(bars)).stats if s.key == "macd_state")
        assert state.value in ("Bullish", "Bullish cross")

    def test_breakdown_lower_is_bearish(self):
        bars = make_bars([100.0] * 50 + [100.0 - i * 3 for i in range(1, 16)])
        state = next(s for s in Macd().compute(ctx_for(bars)).stats if s.key == "macd_state")
        assert state.value in ("Bearish", "Bearish cross")

    def test_perfectly_linear_series_flattens_the_histogram(self):
        """A constant-slope ramp drives MACD and its signal to the same value."""
        bars = make_bars([float(100 + i * 2) for i in range(80)])
        result = Macd().compute(ctx_for(bars))
        assert abs(result.series["histogram"][-1]["value"]) < 0.01
        state = next(s for s in result.stats if s.key == "macd_state")
        assert state.value == "Flat"

    def test_short_series_reports_warming_up(self):
        bars = make_bars([100.0] * 10)
        state = next(s for s in Macd().compute(ctx_for(bars)).stats if s.key == "macd_state")
        assert state.value == "Warming up"

    def test_signal_not_yet_available_is_warming_up(self):
        # Enough bars for the MACD line (26) but not for the signal EMA.
        bars = make_bars([float(100 + i) for i in range(30)])
        result = Macd().compute(ctx_for(bars))
        state = next(s for s in result.stats if s.key == "macd_state")
        assert state.value == "Warming up"


class TestBarSize:
    def test_range_series_matches_high_minus_low(self):
        bars = make_bars([100.0] * 30)
        result = BarSize().compute(ctx_for(bars))
        assert result.series["range"][-1]["value"] == round(bars[-1].range, 2)

    def test_average_starts_after_period(self):
        bars = make_bars([100.0] * 25)
        result = BarSize().compute(ctx_for(bars))
        assert len(result.series["average"]) == 25 - 20 + 1

    def test_expansion_is_flagged(self):
        bars = make_bars([100.0] * 30)
        bars[-1].high = bars[-1].close + 100  # a much larger bar
        bars[-1].low = bars[-1].close - 100
        ratio = next(s for s in BarSize().compute(ctx_for(bars)).stats if s.key == "bar_size_ratio")
        assert ratio.value > 1.5
        assert ratio.tone == "up"

    def test_contraction_is_flagged(self):
        bars = make_bars([100.0] * 30)
        bars[-1].high = bars[-1].close + 0.25
        bars[-1].low = bars[-1].close - 0.25
        ratio = next(s for s in BarSize().compute(ctx_for(bars)).stats if s.key == "bar_size_ratio")
        assert ratio.tone == "down"


class TestSwings:
    # Zigzag price paths. Each level becomes a bar with high=level+1, low=level-1,
    # so pivots land where the path turns.
    #                 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14
    UPTREND = [20, 15, 10, 15, 20, 25, 30, 25, 20, 25, 30, 35, 40, 35, 30]
    DOWNTREND = [20, 25, 30, 25, 20, 15, 10, 15, 20, 15, 10, 5, 0, 5, 10]

    @staticmethod
    def daily(levels):
        return [
            Bar(ts=1_700_000_000 + i * 86400, open=l, high=l + 1, low=l - 1, close=l, complete=True)
            for i, l in enumerate(levels)
        ]

    def test_finds_obvious_pivot(self):
        bars = self.daily([1, 2, 5, 2, 1])
        highs, _ = find_pivots(bars, 2)
        assert highs == [2]

    def test_finds_obvious_trough(self):
        bars = self.daily([5, 4, 1, 4, 5])
        _, lows = find_pivots(bars, 2)
        assert lows == [2]

    def test_uptrend_is_higher_highs_and_higher_lows(self):
        daily = self.daily(self.UPTREND)
        result = DailyStructure().compute(ctx_for(daily, "1d", daily=daily))
        structure = next(s for s in result.stats if s.key == "structure")
        assert structure.value == "Uptrend"
        assert structure.tone == "up"

    def test_downtrend_is_lower_highs_and_lower_lows(self):
        daily = self.daily(self.DOWNTREND)
        result = DailyStructure().compute(ctx_for(daily, "1d", daily=daily))
        structure = next(s for s in result.stats if s.key == "structure")
        assert structure.value == "Downtrend"
        assert structure.tone == "down"

    def test_reports_the_latest_swing_levels(self):
        daily = self.daily(self.UPTREND)
        result = DailyStructure().compute(ctx_for(daily, "1d", daily=daily))
        high = next(s for s in result.stats if s.key == "last_daily_high")
        low = next(s for s in result.stats if s.key == "last_daily_low")
        assert high.value == 41  # the 40 level's high
        assert low.value == 19  # the 20 level's low

    def test_streak_counts_consecutive_higher_highs(self):
        daily = self.daily(self.UPTREND)
        result = DailyStructure().compute(ctx_for(daily, "1d", daily=daily))
        streak = next(s for s in result.stats if s.key == "structure_streak")
        assert streak.value.startswith("1 HH")

    def test_markers_only_on_daily_timeframe(self):
        daily = self.daily(self.UPTREND)
        on_daily = DailyStructure().compute(ctx_for(daily, "1d", daily=daily))
        on_intraday = DailyStructure().compute(ctx_for(make_bars([1.0] * 9), "5m", daily=daily))
        assert on_daily.markers
        assert on_intraday.markers == []

    def test_marker_labels_are_structure_labels(self):
        daily = self.daily(self.UPTREND)
        result = DailyStructure().compute(ctx_for(daily, "1d", daily=daily))
        assert {m.text for m in result.markers} == {"H", "HH", "L", "HL"}

    def test_swing_lines_render_on_intraday_timeframes(self):
        daily = self.daily(self.UPTREND)
        # Intraday bars must sit *after* the daily history for the confirmed
        # levels to have been known by then.
        after = daily[-1].ts + 3600
        result = DailyStructure().compute(
            ctx_for(make_bars([1.0] * 30, start=after), "5m", daily=daily)
        )
        assert {p["value"] for p in result.series["swing_high"]} == {41}
        assert {p["value"] for p in result.series["swing_low"]} == {19}

    def test_too_little_history_warms_up(self):
        daily = self.daily([1, 2])
        result = DailyStructure().compute(ctx_for(daily, "1d", daily=daily))
        assert result.stats[0].value == "Warming up"

    def test_swing_lines_never_lead_the_pivot(self):
        """A level may only appear once its pivot is confirmed."""
        daily = self.daily(self.UPTREND)
        result = DailyStructure().compute(ctx_for(daily, "1d", daily=daily))
        pivot_ts = daily[6].ts  # the 30 level: a swing high confirmed 2 bars later
        early = [p for p in result.series["swing_high"] if p["time"] <= pivot_ts]
        assert all(p["value"] != 31 for p in early)
        later = [p for p in result.series["swing_high"] if p["time"] > pivot_ts]
        assert later and later[0]["value"] == 31


class TestRegistry:
    def test_every_indicator_survives_an_empty_series(self):
        for indicator in all_indicators():
            result = indicator.compute(ctx_for([]))
            assert result.as_dict()["series"] is not None

    def test_render_specs_are_serialisable(self):
        for indicator in all_indicators():
            payload = indicator.as_dict()
            assert payload["key"] and payload["render"]["pane"] in ("price", "own")
            for spec in payload["render"]["series"]:
                assert spec["type"] in ("line", "histogram")
