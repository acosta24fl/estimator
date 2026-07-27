"""Dashboard payloads and the calibrated projection.

Two things matter here. The payload must be JSON-serialisable - NaN is
pervasive in a feature matrix and would produce a page that silently fails to
parse. And the projection must not overstate: a squared-error regressor shrinks
toward zero, so reporting its output as "expected points" would understate
every real move, while reporting a calibrated median without its spread would
overstate the certainty.
"""

import json

import numpy as np
import pandas as pd
import pytest

from mnq.config import Config
from mnq.models.projection import (
    MIN_BUCKET,
    Calibration,
    build_calibration,
    project,
)
from mnq.server import dashboard


@pytest.fixture
def cfg():
    c = Config()
    c.trade.min_probability = 0.58
    return c


def _matrix(n=400, start=15_000.0, drift=0.0, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-02", periods=n, freq="1h", tz="UTC")
    close = start + np.cumsum(rng.normal(drift, 20, n))
    return pd.DataFrame(
        {
            "open": close, "high": close + 25, "low": close - 25,
            "close": close, "volume": 1_000.0, "atr": 40.0,
            "tf1h_rsi": 55.0, "tf1h_adx": 22.0, "tf1h_atr_pct": 0.3,
            "tf1h_macd_hist": 0.1, "tf1h_slope20": 0.02,
            "tf1h_trend_strength_signed": 0.05, "tf1h_rel_volume": 1.1,
            "tf1h_dist_ema21": 0.5, "tf1h_dist_ema50": 0.9,
            "tf1h_dist_ema200": 1.4,
        },
        index=idx,
    )


def _predictions(matrix, p_long=0.62, p_short=0.40):
    return pd.DataFrame(
        {"long_p_meta": p_long, "short_p_meta": p_short}, index=matrix.index
    )


class TestBarsPayload:
    def test_shapes_and_limit(self):
        got = dashboard.bars_payload(_matrix(300), limit=120)
        assert got["count"] == 120 and len(got["bars"]) == 120
        assert set(got["bars"][0]) == {"t", "o", "h", "l", "c", "v"}

    def test_empty_matrix(self):
        got = dashboard.bars_payload(pd.DataFrame())
        assert got["bars"] == [] and got["count"] == 0

    def test_emas_are_recovered_from_the_feature_matrix(self):
        """The chart must not recompute indicators independently.

        A separately computed EMA would eventually disagree with the one the
        model saw, and the disagreement would be invisible.
        """
        m = _matrix(200)
        got = dashboard.bars_payload(m)
        assert "ema21" in got["emas"]
        expected = m["close"] - m["tf1h_dist_ema21"] * m["atr"]
        assert got["emas"]["ema21"][-1] == pytest.approx(expected.iloc[-1])

    def test_nan_becomes_null_not_nan(self):
        """json.dumps emits bare NaN, which no browser will parse."""
        m = _matrix(50)
        m.loc[m.index[10], "close"] = np.nan
        payload = dashboard.bars_payload(m)
        text = json.dumps(payload)
        assert "NaN" not in text
        assert json.loads(text)["bars"][10]["c"] is None


class TestSnapshot:
    def test_is_json_serialisable(self, cfg):
        m = _matrix(200)
        snap = dashboard.build_snapshot(
            cfg, m, {"p_long": 0.6, "p_short": 0.3, "close": 15_000.0,
                     "atr": 40.0, "fwd_pred": 0.05},
            project(None, None, cfg),
        )
        text = json.dumps(snap)
        assert "NaN" not in text and "Infinity" not in text
        assert json.loads(text)["symbol"] == cfg.data.symbol

    def test_levels_use_the_configured_atr_multiples(self, cfg):
        snap = dashboard.build_snapshot(
            cfg, _matrix(100),
            {"p_long": 0.6, "p_short": 0.3, "close": 15_000.0, "atr": 40.0},
            None,
        )
        lv = snap["levels"]
        assert lv["long_target"] == pytest.approx(15_000 + 2.0 * 40)
        assert lv["long_stop"] == pytest.approx(15_000 - 1.0 * 40)
        assert lv["short_target"] == pytest.approx(15_000 - 2.0 * 40)

    def test_survives_a_missing_engine_and_score(self, cfg):
        snap = dashboard.build_snapshot(cfg, _matrix(100), None, None)
        assert snap["chart"]["bars"]
        assert snap["projection"] is None
        assert "levels" not in snap

    def test_trend_reports_points_and_direction(self, cfg):
        m = _matrix(300, drift=5.0, seed=1)   # strong uptrend
        trend = dashboard.trend_summary(m)
        assert trend["last_24"]["direction"] == "up"
        assert trend["last_24"]["points"] > 0

    def test_indicators_pick_the_available_timeframe(self, cfg):
        ind = dashboard.indicator_payload(_matrix(100))
        assert ind["rsi"] == pytest.approx(55.0)
        assert ind["atr"] == pytest.approx(40.0)


class TestCalibration:
    def test_builds_buckets_from_out_of_sample_predictions(self, cfg):
        m = _matrix(600, drift=3.0, seed=5)
        cal = build_calibration(_predictions(m), m, cfg)
        assert cal.available
        rec = cal.to_records()
        assert {"side", "n", "median_points", "p25_points", "hit_rate"} <= set(rec[0])
        assert all(r["n"] >= MIN_BUCKET for r in rec)

    def test_short_side_is_signed_so_positive_means_favourable(self, cfg):
        """A short is right when price falls; the table must reflect that."""
        down = _matrix(600, drift=-3.0, seed=7)
        cal = build_calibration(_predictions(down, 0.40, 0.62), down, cfg)
        shorts = [r for r in cal.to_records() if r["side"] == "short"]
        assert shorts and shorts[0]["median_points"] > 0

    def test_too_little_data_yields_nothing_rather_than_a_guess(self, cfg):
        m = _matrix(20)
        assert not build_calibration(_predictions(m), m, cfg).available

    def test_empty_inputs_are_safe(self, cfg):
        assert not build_calibration(pd.DataFrame(), pd.DataFrame(), cfg).available


class TestProjection:
    def test_no_score_is_reported_as_unavailable(self, cfg):
        p = project(None, None, cfg)
        assert p.direction == "flat" and p.basis == "unavailable"

    def test_direction_follows_the_stronger_side(self, cfg):
        up = project({"p_long": 0.7, "p_short": 0.2, "close": 15_000,
                      "fwd_pred": 0.1}, None, cfg)
        down = project({"p_long": 0.2, "p_short": 0.7, "close": 15_000,
                        "fwd_pred": 0.1}, None, cfg)
        assert up.direction == "up" and down.direction == "down"

    def test_below_the_entry_gate_is_flat(self, cfg):
        p = project({"p_long": 0.52, "p_short": 0.3, "close": 15_000,
                     "fwd_pred": 0.1}, None, cfg)
        assert p.direction == "flat"
        assert "below the" in p.note

    def test_without_calibration_it_says_so(self, cfg):
        p = project({"p_long": 0.7, "p_short": 0.2, "close": 15_000,
                     "fwd_pred": 0.1}, None, cfg)
        assert p.basis == "model"
        assert "shrunk toward zero" in p.note

    def test_calibrated_projection_uses_realised_moves(self, cfg):
        m = _matrix(600, drift=3.0, seed=5)
        cal = build_calibration(_predictions(m), m, cfg)
        p = project({"p_long": 0.62, "p_short": 0.30, "close": float(m["close"].iloc[-1]),
                     "fwd_pred": 0.01}, cal, cfg)
        assert p.basis == "calibrated"
        assert p.n_analogues >= MIN_BUCKET
        assert p.low_points <= p.expected_points <= p.high_points

    def test_the_calibrated_estimate_beats_the_shrunk_regressor(self, cfg):
        """The reason this module exists.

        A squared-error regressor on noisy returns predicts near zero. Taking
        that literally would project moves far smaller than anything tradeable.
        """
        m = _matrix(600, drift=3.0, seed=5)
        cal = build_calibration(_predictions(m), m, cfg)
        score = {"p_long": 0.62, "p_short": 0.30,
                 "close": float(m["close"].iloc[-1]),
                 "fwd_pred": 0.004}      # a typical shrunk output
        p = project(score, cal, cfg)
        assert abs(p.model_points) < 5           # the regressor says ~nothing
        assert abs(p.expected_points) > abs(p.model_points)

    def test_a_range_spanning_zero_is_flagged(self, cfg):
        """Honest uncertainty: a wide band must not read as a confident call."""
        m = _matrix(600, drift=0.0, seed=11)     # no drift -> band spans zero
        cal = build_calibration(_predictions(m), m, cfg)
        p = project({"p_long": 0.62, "p_short": 0.30,
                     "close": float(m["close"].iloc[-1]), "fwd_pred": 0.01}, cal, cfg)
        if p.low_points < 0:
            assert "losing outcome" in p.note

    def test_projection_is_serialisable(self, cfg):
        p = project({"p_long": 0.7, "p_short": 0.2, "close": 15_000,
                     "fwd_pred": 0.1}, None, cfg)
        assert "NaN" not in json.dumps(p.to_dict())


class TestPage:
    def test_page_is_self_contained(self):
        """No CDN, no external scripts - it has to work offline."""
        html = dashboard.read_page()
        assert "<canvas" in html and "/api/snapshot" in html
        for marker in ("http://", "https://"):
            assert f'src="{marker}' not in html
            assert f'href="{marker}' not in html
