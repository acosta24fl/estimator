import math

from app.core.features import (
    average_range,
    close_to_close_moves,
    macd,
    percentile,
    session_vwap,
)
from app.core.forecast import (
    MAX_MOVE_SIGMA,
    MIN_BARS,
    MIN_FIT_SAMPLES,
    build_features,
    compute_forecast,
    ridge_fit,
)
from app.core.timeframes import SessionBucket
from app.indicators.base import ema
from app.models import Bar

SESSION = SessionBucket("America/New_York", 18)
STEP = 300
BASE_TS = 1_700_000_000 // STEP * STEP
#: Enough completed bars to clear both the feature warm-up and the fit.
ENOUGH = MIN_BARS + MIN_FIT_SAMPLES + 20


def bars(closes, spread=4.0, volume=100.0, complete=True):
    return [
        Bar(
            ts=BASE_TS + i * STEP,
            open=c,
            high=c + spread,
            low=c - spread,
            close=c,
            volume=volume,
            complete=complete,
        )
        for i, c in enumerate(closes)
    ]


def wave(n=ENOUGH, base=20_000.0, amp=30.0, period=40):
    """A gently oscillating series — enough variation to fit against."""
    return bars([base + amp * math.sin(2 * math.pi * i / period) for i in range(n)])


class TestFeatureHelpers:
    def test_percentile_endpoints(self):
        assert percentile([1, 2, 3, 4, 5], 0.0) == 1
        assert percentile([1, 2, 3, 4, 5], 1.0) == 5

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


class TestRidge:
    """The solver is the heart of the model, so it gets known-answer tests."""

    def test_recovers_exact_coefficients_without_penalty(self):
        X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]
        y = [3.0 * a + (-2.0) * b for a, b in X]
        beta = ridge_fit(X, y, lam=0.0)
        assert math.isclose(beta[0], 3.0, abs_tol=1e-6)
        assert math.isclose(beta[1], -2.0, abs_tol=1e-6)

    def test_penalty_shrinks_toward_zero(self):
        X = [[1.0], [2.0], [3.0], [4.0]]
        y = [10.0, 20.0, 30.0, 40.0]
        unpenalised = ridge_fit(X, y, lam=0.0)[0]
        shrunk = ridge_fit(X, y, lam=50.0)[0]
        assert 0 < shrunk < unpenalised

    def test_huge_penalty_collapses_to_no_signal(self):
        X = [[1.0], [2.0], [3.0]]
        y = [10.0, 20.0, 30.0]
        assert abs(ridge_fit(X, y, lam=1e12)[0]) < 1e-6

    def test_pure_noise_gives_near_zero_coefficients(self):
        """The safety property: no signal must degrade to 'no move'."""
        X = [[1.0 if i % 2 else -1.0] for i in range(200)]
        y = [0.0] * 200
        assert abs(ridge_fit(X, y, lam=10.0)[0]) < 1e-9

    def test_singular_system_returns_zeros_not_an_explosion(self):
        X = [[0.0, 0.0], [0.0, 0.0]]
        assert ridge_fit(X, [1.0, 2.0], lam=0.0) == [0.0, 0.0]

    def test_empty_input(self):
        assert ridge_fit([], [], lam=1.0) == []


class TestVwapScaling:
    """Regression test for the saturation defect.

    The old normaliser divided session-scale VWAP distance by a 5-minute bar
    range, which drove tanh into saturation on roughly half of all bars and
    collapsed the feature to a bare sign bit.
    """

    def test_a_constant_offset_is_scale_free(self):
        """Distance normalised by its own magnitude must not depend on size."""
        _, small, _ = build_features(wave(), [], SESSION)
        # Same shape, 100x further from VWAP: the score must not saturate.
        big_bars = wave(base=20_000.0, amp=3000.0)
        _, big, _ = build_features(big_bars, [], SESSION)
        assert abs(small[-1][1]) < 0.999
        assert abs(big[-1][1]) < 0.999

    def test_scores_stay_inside_the_unit_range(self):
        _, rows, _ = build_features(wave(), [], SESSION)
        for row in rows:
            for value in row:
                assert -1.0 <= value <= 1.0

    def test_saturation_is_rare(self):
        _, rows, _ = build_features(wave(), [], SESSION)
        vwap_scores = [abs(r[1]) for r in rows[MIN_BARS:]]
        saturated = sum(1 for v in vwap_scores if v > 0.99) / len(vwap_scores)
        assert saturated < 0.05, f"vwap feature saturating on {saturated:.0%} of bars"


class TestGuards:
    def test_too_little_history_is_invalid(self):
        forecast = compute_forecast(wave(10), [], SESSION)
        assert not forecast.valid
        assert str(MIN_BARS) in forecast.reason

    def test_not_enough_fitted_samples_reports_learning(self):
        forecast = compute_forecast(wave(MIN_BARS + 30), [], SESSION)
        assert not forecast.valid
        assert "learning" in forecast.reason
        assert not forecast.fit.ready

    def test_incomplete_bars_do_not_count(self):
        series = wave()
        for bar in series:
            bar.complete = False
        assert not compute_forecast(series, [], SESSION).valid

    def test_zero_volatility_is_invalid(self):
        series = bars([20_000.0] * ENOUGH, spread=0.0)
        forecast = compute_forecast(series, [], SESSION)
        assert not forecast.valid
        assert "volatility" in forecast.reason


class TestNoLookahead:
    def test_the_forming_bar_is_never_used(self):
        series = wave()
        forming = Bar(
            ts=series[-1].ts + STEP, open=20_000, high=99_999, low=20_000,
            close=99_999, volume=1, complete=False,
        )
        forecast = compute_forecast(series + [forming], [], SESSION)
        assert forecast.anchor_price == series[-1].close
        assert forecast.target < 30_000

    def test_a_wild_forming_bar_does_not_move_the_projection(self):
        series = wave()
        clean = compute_forecast(series, [], SESSION)
        forming = Bar(
            ts=series[-1].ts + STEP, open=20_000, high=50_000, low=1.0,
            close=50_000, volume=1, complete=False,
        )
        dirty = compute_forecast(series + [forming], [], SESSION)
        assert math.isclose(clean.target, dirty.target, abs_tol=1e-9)

    def test_anchor_and_target_span_the_horizon(self):
        forecast = compute_forecast(wave(), [], SESSION)
        assert forecast.target_ts - forecast.anchor_ts == 300
        assert forecast.anchor_ts == BASE_TS + (ENOUGH - 1) * STEP + 300


class TestFittedForecast:
    def test_a_fitted_forecast_is_produced(self):
        forecast = compute_forecast(wave(), [], SESSION)
        assert forecast.valid
        assert forecast.fit.ready
        assert forecast.fit.samples >= MIN_FIT_SAMPLES

    def test_all_three_factors_are_reported_with_coefficients(self):
        forecast = compute_forecast(wave(), [], SESSION)
        assert {f.key for f in forecast.factors} == {"momentum", "vwap", "structure"}
        assert set(forecast.fit.coefficients) == {"momentum", "vwap", "structure"}

    def test_contributions_sum_to_the_projection(self):
        """What the panel shows must add up to what the chart draws."""
        forecast = compute_forecast(wave(), [], SESSION)
        total = sum(f.points for f in forecast.factors)
        assert math.isclose(total, forecast.expected_move, abs_tol=1e-6)

    def test_contributions_still_sum_when_the_move_is_clamped(self):
        series = wave(amp=400.0, period=12)  # violent, likely to clamp
        forecast = compute_forecast(series, [], SESSION)
        total = sum(f.points for f in forecast.factors)
        assert math.isclose(total, forecast.expected_move, abs_tol=1e-6)

    def test_move_is_capped_at_the_volatility_limit(self):
        forecast = compute_forecast(wave(amp=400.0, period=12), [], SESSION)
        assert abs(forecast.expected_move) <= MAX_MOVE_SIGMA * forecast.volatility + 1e-9

    def test_band_brackets_the_target(self):
        forecast = compute_forecast(wave(), [], SESSION)
        assert forecast.low <= forecast.target <= forecast.high

    def test_strength_scales_the_projection(self):
        full = compute_forecast(wave(), [], SESSION, strength=1.0)
        half = compute_forecast(wave(), [], SESSION, strength=0.5)
        if abs(full.expected_move) > 1e-6:
            assert abs(half.expected_move) < abs(full.expected_move)

    def test_zero_strength_projects_no_move(self):
        forecast = compute_forecast(wave(), [], SESSION, strength=0.0)
        assert math.isclose(forecast.target, forecast.anchor_price, abs_tol=1e-9)

    def test_a_huge_ridge_penalty_degrades_to_no_move(self):
        """Heavy shrinkage must collapse toward the baseline, not misbehave."""
        forecast = compute_forecast(wave(), [], SESSION, ridge_lambda=1e12)
        assert abs(forecast.expected_move) < 0.01

    def test_serialises(self):
        payload = compute_forecast(wave(), [], SESSION).as_dict()
        assert payload["valid"] is True
        assert len(payload["factors"]) == 3
        assert payload["fit"]["ready"] is True
        assert "coefficients" in payload["fit"]
