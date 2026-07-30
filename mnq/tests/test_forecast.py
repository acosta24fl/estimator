import math

from app.core.features import (
    average_range,
    close_to_close_moves,
    macd,
    percentile,
    session_vwap,
)
from app.core.forecast import MAX_MOVE_SIGMA, MIN_BARS, compute_forecast
from app.core.timeframes import SessionBucket
from app.indicators.base import ema
from app.models import Bar

SESSION = SessionBucket("America/New_York", 18)
STEP = 300
BASE_TS = 1_700_000_000 // STEP * STEP


def bars(closes, spread=4.0, volume=100.0, complete=True):
    out = []
    for i, c in enumerate(closes):
        out.append(
            Bar(
                ts=BASE_TS + i * STEP,
                open=c,
                high=c + spread,
                low=c - spread,
                close=c,
                volume=volume,
                complete=complete,
            )
        )
    return out


def flat(n=60, price=20_000.0):
    return bars([price] * n)


class TestFeatureHelpers:
    def test_percentile_endpoints(self):
        values = [1, 2, 3, 4, 5]
        assert percentile(values, 0.0) == 1
        assert percentile(values, 1.0) == 5

    def test_percentile_interpolates(self):
        assert percentile([0, 10], 0.5) == 5

    def test_percentile_of_empty_is_zero(self):
        assert percentile([], 0.5) == 0.0

    def test_average_range(self):
        assert average_range(bars([100.0] * 10, spread=3.0), 5) == 6.0

    def test_close_to_close_moves(self):
        assert close_to_close_moves(bars([100.0, 105.0, 102.0])) == [5.0, 3.0]

    def test_macd_matches_the_ema_definition(self):
        closes = [float(100 + i) for i in range(60)]
        line, signal, hist = macd(closes)
        fast, slow = ema(closes, 12), ema(closes, 26)
        assert math.isclose(line[-1], fast[-1] - slow[-1], abs_tol=1e-9)
        assert math.isclose(hist[-1], line[-1] - signal[-1], abs_tol=1e-9)

    def test_session_vwap_resets(self):
        open_ts = SESSION.start(BASE_TS)
        roll = SESSION.end(open_ts)
        series = [
            Bar(ts=roll - 600, open=100, high=100, low=100, close=100, volume=10, complete=True),
            Bar(ts=roll, open=500, high=500, low=500, close=500, volume=10, complete=True),
        ]
        assert math.isclose(session_vwap(series, SESSION)[-1], 500.0)


class TestForecastGuards:
    def test_too_little_history_is_invalid(self):
        forecast = compute_forecast(flat(10), "Range", SESSION)
        assert not forecast.valid
        assert str(MIN_BARS) in forecast.reason

    def test_incomplete_bars_do_not_count(self):
        series = flat(60)
        for bar in series:
            bar.complete = False
        assert not compute_forecast(series, "Range", SESSION).valid

    def test_zero_volatility_is_invalid(self):
        series = bars([20_000.0] * 60, spread=0.0)
        forecast = compute_forecast(series, "Range", SESSION)
        assert not forecast.valid
        assert "volatility" in forecast.reason

    def test_the_forming_bar_is_never_used(self):
        """The projection must not see the bar it is predicting."""
        series = flat(60)
        forming = Bar(
            ts=series[-1].ts + STEP, open=20_000, high=99_999, low=20_000,
            close=99_999, volume=1, complete=False,
        )
        forecast = compute_forecast(series + [forming], "Range", SESSION)
        # Anchored on the last completed close, not the wild forming bar.
        assert forecast.anchor_price == 20_000.0
        assert forecast.target < 21_000


class TestForecastDirection:
    def test_anchor_and_target_timestamps_span_the_horizon(self):
        forecast = compute_forecast(flat(60), "Range", SESSION)
        assert forecast.target_ts - forecast.anchor_ts == 300
        assert forecast.anchor_ts == BASE_TS + 59 * STEP + 300

    def test_uptrend_structure_pushes_the_target_up(self):
        up = compute_forecast(flat(60), "Uptrend", SESSION)
        down = compute_forecast(flat(60), "Downtrend", SESSION)
        assert up.target > down.target

    def test_rally_gives_upward_momentum(self):
        series = bars([20_000.0] * 45 + [20_000.0 + i * 6 for i in range(1, 16)])
        momentum = next(
            f for f in compute_forecast(series, "Range", SESSION).factors
            if f.key == "momentum"
        )
        assert momentum.score > 0

    def test_selloff_gives_downward_momentum(self):
        series = bars([20_000.0] * 45 + [20_000.0 - i * 6 for i in range(1, 16)])
        momentum = next(
            f for f in compute_forecast(series, "Range", SESSION).factors
            if f.key == "momentum"
        )
        assert momentum.score < 0

    def test_price_far_above_vwap_pulls_down(self):
        # A long flat base then a spike leaves price stretched above VWAP.
        series = bars([20_000.0] * 55 + [20_400.0] * 5)
        vwap = next(
            f for f in compute_forecast(series, "Range", SESSION).factors
            if f.key == "vwap"
        )
        assert vwap.score < 0, "stretched above VWAP should pull back down"

    def test_price_far_below_vwap_pulls_up(self):
        series = bars([20_000.0] * 55 + [19_600.0] * 5)
        vwap = next(
            f for f in compute_forecast(series, "Range", SESSION).factors
            if f.key == "vwap"
        )
        assert vwap.score > 0


class TestForecastBounds:
    def test_move_is_capped_at_the_volatility_limit(self):
        series = bars([20_000.0] * 30 + [20_000.0 + i * 40 for i in range(1, 31)])
        forecast = compute_forecast(series, "Uptrend", SESSION)
        assert abs(forecast.expected_move) <= MAX_MOVE_SIGMA * forecast.volatility + 1e-9

    def test_band_brackets_the_target(self):
        forecast = compute_forecast(flat(60), "Range", SESSION)
        assert forecast.low <= forecast.target <= forecast.high

    def test_factor_points_sum_to_the_expected_move(self):
        """The breakdown shown to the user must add up to the total."""
        series = bars([20_000.0] * 40 + [20_000.0 + i * 5 for i in range(1, 21)])
        forecast = compute_forecast(series, "Uptrend", SESSION)
        total = sum(f.points for f in forecast.factors)
        assert math.isclose(total, forecast.expected_move, abs_tol=1e-6)

    def test_all_three_factors_are_reported(self):
        forecast = compute_forecast(flat(60), "Range", SESSION)
        assert {f.key for f in forecast.factors} == {"momentum", "vwap", "structure"}

    def test_weights_sum_to_one(self):
        forecast = compute_forecast(flat(60), "Range", SESSION)
        assert math.isclose(sum(f.weight for f in forecast.factors), 1.0)

    def test_agreement_counts_factors_matching_the_direction(self):
        series = bars([20_000.0] * 40 + [20_000.0 + i * 5 for i in range(1, 21)])
        forecast = compute_forecast(series, "Uptrend", SESSION)
        agreeing = sum(1 for f in forecast.factors if f.direction == forecast.direction and f.points)
        assert forecast.agreement == agreeing

    def test_serialises(self):
        payload = compute_forecast(flat(60), "Range", SESSION).as_dict()
        assert payload["valid"] is True
        assert len(payload["factors"]) == 3
