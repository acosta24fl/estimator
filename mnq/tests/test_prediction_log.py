from app.core.forecast import Forecast
from app.core.prediction_log import PredictionLog
from app.models import Bar

STEP = 300
BASE_TS = 1_700_000_000 // STEP * STEP


def forecast_at(anchor_ts, anchor_price=20_000.0, target=20_010.0, band=8.0):
    return Forecast(
        valid=True,
        anchor_ts=anchor_ts,
        target_ts=anchor_ts + STEP,
        anchor_price=anchor_price,
        target=target,
        low=target - band,
        high=target + band,
        expected_move=target - anchor_price,
        volatility=10.0,
        band_half=band,
        direction="up" if target > anchor_price else "down",
    )


def bar_at(ts, close, complete=True):
    return Bar(ts=ts, open=close, high=close, low=close, close=close, complete=complete)


class TestLocking:
    def test_a_valid_forecast_is_recorded(self, tmp_path):
        logbook = PredictionLog(tmp_path / "p.jsonl")
        assert logbook.observe(forecast_at(BASE_TS)) is not None
        assert logbook.count == 1

    def test_an_invalid_forecast_is_ignored(self, tmp_path):
        logbook = PredictionLog(tmp_path / "p.jsonl")
        assert logbook.observe(Forecast(valid=False, reason="warming up")) is None
        assert logbook.count == 0

    def test_a_bar_is_only_predicted_once(self, tmp_path):
        """Locking once is what stops the projection being revised mid-bar."""
        logbook = PredictionLog(tmp_path / "p.jsonl")
        logbook.observe(forecast_at(BASE_TS, target=20_010.0))
        assert logbook.observe(forecast_at(BASE_TS, target=20_999.0)) is None
        assert logbook.count == 1

    def test_each_new_bar_gets_its_own_prediction(self, tmp_path):
        logbook = PredictionLog(tmp_path / "p.jsonl")
        for i in range(3):
            logbook.observe(forecast_at(BASE_TS + i * STEP))
        assert logbook.count == 3

    def test_predictions_survive_a_restart(self, tmp_path):
        path = tmp_path / "p.jsonl"
        first = PredictionLog(path)
        first.observe(forecast_at(BASE_TS))
        reloaded = PredictionLog(path)
        assert reloaded.load() == 1
        assert reloaded.count == 1

    def test_a_corrupt_line_is_skipped(self, tmp_path):
        path = tmp_path / "p.jsonl"
        logbook = PredictionLog(path)
        logbook.observe(forecast_at(BASE_TS))
        with path.open("a") as fh:
            fh.write("{broken\n")
        assert PredictionLog(path).load() == 1


class TestScoring:
    def test_an_unmatured_prediction_is_not_scored(self, tmp_path):
        logbook = PredictionLog(tmp_path / "p.jsonl")
        logbook.observe(forecast_at(BASE_TS))
        assert logbook.scored([bar_at(BASE_TS, 20_010.0, complete=False)]) == []

    def test_error_is_actual_minus_target(self, tmp_path):
        logbook = PredictionLog(tmp_path / "p.jsonl")
        logbook.observe(forecast_at(BASE_TS, target=20_010.0))
        scored = logbook.scored([bar_at(BASE_TS, 20_004.0)])[0]
        assert scored.actual == 20_004.0
        assert scored.error == -6.0

    def test_direction_is_correct_when_price_moves_the_predicted_way(self, tmp_path):
        logbook = PredictionLog(tmp_path / "p.jsonl")
        logbook.observe(forecast_at(BASE_TS, anchor_price=20_000.0, target=20_010.0))
        scored = logbook.scored([bar_at(BASE_TS, 20_003.0)])[0]
        assert scored.direction_correct  # predicted up, went up

    def test_direction_is_wrong_when_price_moves_the_other_way(self, tmp_path):
        logbook = PredictionLog(tmp_path / "p.jsonl")
        logbook.observe(forecast_at(BASE_TS, anchor_price=20_000.0, target=20_010.0))
        scored = logbook.scored([bar_at(BASE_TS, 19_995.0)])[0]
        assert not scored.direction_correct

    def test_within_band_is_tracked(self, tmp_path):
        logbook = PredictionLog(tmp_path / "p.jsonl")
        logbook.observe(forecast_at(BASE_TS, target=20_010.0, band=8.0))
        assert logbook.scored([bar_at(BASE_TS, 20_015.0)])[0].within_band
        logbook2 = PredictionLog(tmp_path / "q.jsonl")
        logbook2.observe(forecast_at(BASE_TS, target=20_010.0, band=8.0))
        assert not logbook2.scored([bar_at(BASE_TS, 20_100.0)])[0].within_band


class TestAccuracy:
    def test_empty_accuracy_is_reported_as_no_data(self, tmp_path):
        assert PredictionLog(tmp_path / "p.jsonl").accuracy([])["count"] == 0

    def test_perfect_predictions_score_100_percent(self, tmp_path):
        logbook = PredictionLog(tmp_path / "p.jsonl")
        actual_bars = []
        for i in range(5):
            ts = BASE_TS + i * STEP
            logbook.observe(forecast_at(ts, anchor_price=20_000.0, target=20_010.0))
            actual_bars.append(bar_at(ts, 20_010.0))
        acc = logbook.accuracy(actual_bars)
        assert acc["count"] == 5
        assert acc["direction_rate"] == 1.0
        assert acc["mean_abs_error"] == 0.0

    def test_baseline_is_the_actual_distance_travelled(self, tmp_path):
        """The no-move baseline is what the model must beat to be useful."""
        logbook = PredictionLog(tmp_path / "p.jsonl")
        logbook.observe(forecast_at(BASE_TS, anchor_price=20_000.0, target=20_010.0))
        acc = logbook.accuracy([bar_at(BASE_TS, 20_006.0)])
        assert acc["baseline_abs_error"] == 6.0  # |20006 - 20000|
        assert acc["mean_abs_error"] == 4.0  # |20006 - 20010|

    def test_window_limits_how_far_back_accuracy_looks(self, tmp_path):
        logbook = PredictionLog(tmp_path / "p.jsonl")
        actual_bars = []
        for i in range(10):
            ts = BASE_TS + i * STEP
            logbook.observe(forecast_at(ts))
            actual_bars.append(bar_at(ts, 20_010.0))
        assert logbook.accuracy(actual_bars, window=4)["count"] == 4


class TestDirectionMetricIntegrity:
    """A flat projection makes no directional claim and must not be scored as one."""

    def test_flat_predictions_are_excluded_from_the_direction_rate(self, tmp_path):
        logbook = PredictionLog(tmp_path / "p.jsonl")
        actual_bars = []
        for i in range(4):
            ts = BASE_TS + i * STEP
            flat = forecast_at(ts, anchor_price=20_000.0, target=20_000.0)
            flat.direction = "flat"
            logbook.observe(flat)
            actual_bars.append(bar_at(ts, 20_000.5))
        acc = logbook.accuracy(actual_bars)
        assert acc["count"] == 4
        assert acc["directional_count"] == 0
        assert acc["direction_rate"] is None

    def test_direction_rate_uses_only_directional_calls(self, tmp_path):
        logbook = PredictionLog(tmp_path / "p.jsonl")
        actual_bars = []
        # Two flat calls, then two correct upward calls.
        for i in range(2):
            ts = BASE_TS + i * STEP
            flat = forecast_at(ts, anchor_price=20_000.0, target=20_000.0)
            flat.direction = "flat"
            logbook.observe(flat)
            actual_bars.append(bar_at(ts, 20_000.0))
        for i in range(2, 4):
            ts = BASE_TS + i * STEP
            logbook.observe(forecast_at(ts, anchor_price=20_000.0, target=20_010.0))
            actual_bars.append(bar_at(ts, 20_008.0))
        acc = logbook.accuracy(actual_bars)
        assert acc["directional_count"] == 2
        assert acc["direction_rate"] == 1.0

    def test_zero_drift_error_equals_the_baseline_exactly(self, tmp_path):
        """A no-move projection must score identically to the baseline."""
        logbook = PredictionLog(tmp_path / "p.jsonl")
        actual_bars = []
        for i in range(5):
            ts = BASE_TS + i * STEP
            flat = forecast_at(ts, anchor_price=20_000.0, target=20_000.0)
            flat.direction = "flat"
            logbook.observe(flat)
            actual_bars.append(bar_at(ts, 20_000.0 + i * 3))
        acc = logbook.accuracy(actual_bars)
        assert acc["mean_abs_error"] == acc["baseline_abs_error"]
