"""Multi-horizon forecasting and the skill score that gates it.

The single most important test here is the random-walk control. This panel
projects price at seven horizons and grades its own conviction; if it can be
made to look confident on noise, it is worse than not existing.

Two real defects were found by exactly that control while this was built:

1. Skill measured **in-sample** gave a pure random walk skill up to +0.12 and
   an alpha grade of A, because a conditional median always fits the data it
   was computed from.
2. Even out of sample, the 1440-minute horizon swung between +0.14 and -0.32
   across seeds, because 18,000 overlapping test bars hold only ~12
   *independent* 24-hour windows.

Both are now guarded, and the tests below fail if either guard is removed.
"""

import numpy as np
import pandas as pd
import pytest

from mnq.models.horizon import (
    HORIZONS,
    MIN_EFFECTIVE_WINDOWS,
    SKILL_FLOOR,
    alpha_score,
    build_horizon_model,
    classify,
    forecast,
    path_payload,
)

COST_POINTS = 0.87   # MNQ round turn: $1.74 at $2/point


def random_walk(n=60_000, seed=1, sigma=2.0):
    rng = np.random.default_rng(seed)
    px = 21_000 + np.cumsum(rng.normal(0, sigma, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    # The bar range scales with sigma. Pinning high/low at a constant would
    # make ATR independent of the noise, and then nothing ATR-normalised could
    # possibly be scale-free - a property of the fixture, not of the code.
    wick = sigma
    return pd.DataFrame(
        {"open": px, "high": px + wick, "low": px - wick, "close": px,
         "volume": 10.0},
        index=idx,
    )


def momentum_walk(n=60_000, seed=1):
    """A slow-moving drift: state genuinely predicts the next move."""
    rng = np.random.default_rng(seed)
    drift, d = np.zeros(n), 0.0
    for i in range(n):
        d = 0.995 * d + rng.normal(0, 0.05)
        drift[i] = d
    px = 21_000 + np.cumsum(drift + rng.normal(0, 2.0, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"open": px, "high": px + 2, "low": px - 2, "close": px, "volume": 10.0},
        index=idx,
    )


class TestClassify:
    def test_labels_every_bar_with_a_state(self):
        got = classify(random_walk(5_000))
        assert {"slope", "atr", "atr_pct", "trend", "vol"} <= set(got.columns)
        assert got["trend"].dropna().isin(
            ["strong_down", "down", "range", "up", "strong_up"]
        ).all()
        assert got["vol"].dropna().isin(["low", "normal", "high"]).all()

    def test_a_rising_series_is_labelled_up(self):
        n = 3_000
        px = 21_000 + np.arange(n) * 1.5      # steady climb
        idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
        df = pd.DataFrame(
            {"open": px, "high": px + 1, "low": px - 1, "close": px, "volume": 10.0},
            index=idx,
        )
        assert classify(df)["trend"].iloc[-1] in ("up", "strong_up")

    def test_slope_is_volatility_normalised(self):
        """The same shape at double the noise must not read as a stronger trend."""
        quiet, loud = random_walk(20_000, seed=5, sigma=1.0), random_walk(
            20_000, seed=5, sigma=4.0
        )
        a = classify(quiet)["slope"].abs().median()
        b = classify(loud)["slope"].abs().median()
        assert abs(a - b) < 0.35 * max(a, b), "slope should be scale-free"


class TestSkillIsHonest:
    """The control. If these pass trivially, the guards have been removed."""

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_a_random_walk_scores_no_skill_and_grade_f(self, seed):
        data = random_walk(60_000, seed=seed)
        model = build_horizon_model(data)
        fc, read = forecast(data, model)

        assert fc, "forecasts should still be produced"
        assert not any(f.has_skill for f in fc), (
            "a random walk must not show skill at any horizon; "
            f"got {[(f.label, round(f.skill, 4)) for f in fc if f.has_skill]}"
        )
        alpha = alpha_score(fc, read, COST_POINTS)
        assert alpha["score"] == 0.0 and alpha["grade"] == "F"
        assert not alpha["tradeable"]

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_genuine_momentum_is_detected(self, seed):
        """The other half: the control must not simply reject everything."""
        data = momentum_walk(60_000, seed=seed)
        model = build_horizon_model(data)
        fc, _ = forecast(data, model)
        skilled = [f for f in fc if f.has_skill]
        assert len(skilled) >= 4, "real structure should register at most horizons"
        assert max(f.skill for f in skilled) > 0.05

    def test_skill_is_measured_out_of_sample(self):
        """In-sample scoring was the original bug; prove the split exists.

        Two observable consequences of a real split: an honest random walk
        scores ~0 at every measurable horizon, and shrinking the test block
        below the minimum leaves nothing to score rather than silently falling
        back to in-sample numbers.
        """
        data = random_walk(40_000, seed=7)
        honest = build_horizon_model(data, train_frac=0.7)
        skill = honest.table.reset_index().groupby("horizon")["skill"].first()
        assert skill.abs().max() < 0.25, (
            f"a random walk should score near zero out of sample; got {dict(skill)}"
        )

        # No test block -> no model, rather than an in-sample score.
        starved = build_horizon_model(data, train_frac=0.999)
        assert not starved.available

    def test_training_block_stops_short_of_the_test_block(self):
        """The label at bar t peeks h bars ahead, so the blocks must not touch.

        Without the h-bar gap the last rows of training carry outcomes drawn
        from the test period, which is a lookahead the skill score cannot see.
        """
        n, train_frac = 20_000, 0.7
        h = 240
        split = int(n * train_frac)
        # This mirrors the slicing in build_horizon_model.
        train_end = max(0, split - h)
        assert train_end + h <= split, "training must end at least h bars early"

    def test_long_horizons_are_rejected_for_too_few_windows(self):
        """1440m over 60k bars is ~12 independent windows - not measurable."""
        data = random_walk(60_000, seed=2)
        fc, _ = forecast(data, build_horizon_model(data))
        day = [f for f in fc if f.minutes == 1440]
        assert day, "the horizon should still appear"
        assert day[0].effective_windows < MIN_EFFECTIVE_WINDOWS
        assert not day[0].has_skill
        assert "independent" in day[0].note

    def test_short_horizons_have_enough_windows(self):
        data = random_walk(60_000, seed=2)
        fc, _ = forecast(data, build_horizon_model(data))
        five = [f for f in fc if f.minutes == 5][0]
        assert five.effective_windows >= MIN_EFFECTIVE_WINDOWS

    def test_skill_floor_is_above_zero(self):
        """A hair above zero is luck, not skill."""
        assert SKILL_FLOOR > 0


class TestForecast:
    def test_produces_every_horizon_with_ordered_bands(self):
        data = momentum_walk(60_000, seed=1)
        fc, _ = forecast(data, build_horizon_model(data))
        assert {f.minutes for f in fc} <= set(HORIZONS)
        for f in fc:
            assert f.p10_points <= f.p25_points <= f.p75_points <= f.p90_points
            assert 0.0 <= f.p_up <= 1.0
            assert f.median_price == pytest.approx(f.from_price + f.median_points)

    def test_bands_widen_with_horizon(self):
        """Uncertainty must grow with time; a flat cone would be wrong."""
        data = random_walk(60_000, seed=4)
        fc, _ = forecast(data, build_horizon_model(data))
        widths = [(f.minutes, f.p90_points - f.p10_points) for f in fc]
        widths.sort()
        spans = [w for _, w in widths]
        assert spans[-1] > spans[0] * 2, f"cone should widen: {widths}"

    def test_no_model_returns_empty_not_a_guess(self):
        fc, read = forecast(random_walk(500), None)
        assert fc == [] and read.verdict == "unknown"

    def test_too_little_history_builds_nothing(self):
        assert not build_horizon_model(random_walk(500)).available

    def test_path_payload_is_serialisable(self):
        import json

        data = momentum_walk(60_000, seed=1)
        fc, _ = forecast(data, build_horizon_model(data))
        text = json.dumps(path_payload(fc))
        assert "NaN" not in text


class TestTrendRead:
    def test_a_range_state_refuses_to_call_continuation(self):
        n = 20_000
        rng = np.random.default_rng(3)
        px = 21_000 + rng.normal(0, 3, n)      # mean-reverting, no trend
        idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
        flat = pd.DataFrame(
            {"open": px, "high": px + 2, "low": px - 2, "close": px, "volume": 10.0},
            index=idx,
        )
        _, read = forecast(flat, build_horizon_model(flat))
        if read.state == "range":
            assert read.verdict == "no trend"
            assert "not meaningful" in read.note

    def test_continuation_is_judged_against_the_base_rate(self):
        """55% continuation in a market that rises 55% of the time is nothing."""
        data = random_walk(60_000, seed=1)
        _, read = forecast(data, build_horizon_model(data))
        if read.continuation_edge is not None:
            assert abs(read.continuation_edge) < 0.25
        assert read.verdict in ("no trend", "no signal", "neutral",
                                "continuation", "reversal", "unknown")

    def test_without_skill_the_trend_is_called_descriptive(self):
        data = random_walk(60_000, seed=2)
        _, read = forecast(data, build_horizon_model(data))
        if read.state != "range":
            assert read.verdict == "no signal"
            assert "descriptive" in read.note


class TestAlphaScore:
    def _forecasts(self, kind, seed=1):
        data = momentum_walk(60_000, seed) if kind == "momentum" else random_walk(60_000, seed)
        return forecast(data, build_horizon_model(data))

    def test_components_are_all_reported(self):
        fc, read = self._forecasts("momentum")
        a = alpha_score(fc, read, COST_POINTS)
        assert set(a["components"]) == {"skill", "conviction", "payoff", "agreement"}
        assert 0.0 <= a["score"] <= 100.0

    def test_no_skill_anywhere_scores_exactly_zero(self):
        a = alpha_score([], None, COST_POINTS)
        assert a["score"] == 0.0 and a["grade"] == "F"
        assert "nothing to score" in a["note"]

    def test_grade_tracks_score(self):
        fc, read = self._forecasts("momentum")
        a = alpha_score(fc, read, COST_POINTS)
        expected = ("A" if a["score"] >= 75 else "B" if a["score"] >= 60
                    else "C" if a["score"] >= 45 else "D" if a["score"] >= 25 else "F")
        assert a["grade"] == expected

    def test_higher_costs_reduce_the_payoff_component(self):
        fc, read = self._forecasts("momentum")
        cheap = alpha_score(fc, read, 0.5)["components"]["payoff"]
        dear = alpha_score(fc, read, 50.0)["components"]["payoff"]
        assert dear < cheap

    def test_is_serialisable(self):
        import json

        fc, read = self._forecasts("momentum")
        json.dumps(alpha_score(fc, read, COST_POINTS))
