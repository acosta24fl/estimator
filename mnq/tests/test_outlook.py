from app.core.forecast import Forecast
from app.core.outlook import BEARISH, BULLISH, NEUTRAL, build_outlook


def forecast(move, band=20.0, valid=True, reason=""):
    return Forecast(
        valid=valid, reason=reason, anchor_price=20_000.0,
        target=20_000.0 + move, expected_move=move,
        band_half=band, volatility=25.0,
    )


class TestDirection:
    def test_a_clear_up_projection_is_bullish(self):
        assert build_outlook(forecast(+10.0), 10).direction == BULLISH

    def test_a_clear_down_projection_is_bearish(self):
        assert build_outlook(forecast(-10.0), 10).direction == BEARISH

    def test_the_horizon_is_carried_through(self):
        assert build_outlook(forecast(+10.0), 10).horizon_minutes == 10

    def test_target_and_anchor_are_reported(self):
        o = build_outlook(forecast(+10.0), 10)
        assert o.anchor == 20_000.0 and o.target == 20_010.0


class TestNoCall:
    def test_a_noise_sized_move_makes_no_call(self):
        """A projection far smaller than a typical move must not colour the page."""
        o = build_outlook(forecast(+0.5, band=20.0), 10)
        assert o.direction == NEUTRAL
        assert "too small" in o.reason

    def test_an_invalid_forecast_makes_no_call(self):
        o = build_outlook(forecast(0.0, valid=False, reason="learning: 10/200"), 10)
        assert o.direction == NEUTRAL
        assert "learning" in o.reason

    def test_a_missing_forecast_makes_no_call(self):
        assert build_outlook(None, 10).direction == NEUTRAL

    def test_a_flat_projection_stays_neutral_at_zero_threshold(self):
        """With the threshold off, a zero move must not read as bearish."""
        o = build_outlook(forecast(0.0), 10, min_ratio=0.0)
        assert o.direction == NEUTRAL

    def test_a_tiny_move_does_call_at_zero_threshold(self):
        """Turning the threshold off is what makes marginal calls visible."""
        assert build_outlook(forecast(+0.03), 10, min_ratio=0.0).direction == BULLISH
        assert build_outlook(forecast(-0.03), 10, min_ratio=0.0).direction == BEARISH

    def test_zero_band_does_not_divide_by_zero(self):
        assert build_outlook(forecast(+5.0, band=0.0), 10).direction == NEUTRAL

    def test_the_threshold_is_relative_to_typical_move(self):
        """The same points figure calls in a quiet market and not in a busy one."""
        assert build_outlook(forecast(+2.0, band=5.0), 10).direction == BULLISH
        assert build_outlook(forecast(+2.0, band=100.0), 10).direction == NEUTRAL


class TestConfidence:
    def test_tiers_rise_with_relative_size(self):
        low = build_outlook(forecast(+3.0, band=20.0), 10)  # 0.15
        medium = build_outlook(forecast(+10.0, band=20.0), 10)  # 0.50
        high = build_outlook(forecast(+18.0, band=20.0), 10)  # 0.90
        assert (low.confidence, medium.confidence, high.confidence) == (
            "low", "medium", "high",
        )

    def test_no_call_has_no_confidence(self):
        assert build_outlook(forecast(+0.1, band=20.0), 10).confidence == "none"

    def test_negative_skill_caps_confidence_at_low(self):
        """A model losing to the baseline must not present as confident."""
        o = build_outlook(forecast(+18.0, band=20.0), 10, skill=-0.15)
        assert o.confidence == "low"
        assert o.capped_by_skill
        assert "not beating" in o.reason

    def test_positive_skill_leaves_confidence_alone(self):
        o = build_outlook(forecast(+18.0, band=20.0), 10, skill=0.08)
        assert o.confidence == "high"
        assert not o.capped_by_skill

    def test_unknown_skill_leaves_confidence_alone(self):
        o = build_outlook(forecast(+18.0, band=20.0), 10, skill=None)
        assert o.confidence == "high"

    def test_capping_never_flips_the_direction(self):
        o = build_outlook(forecast(+18.0, band=20.0), 10, skill=-0.5)
        assert o.direction == BULLISH


class TestSerialisation:
    def test_payload_has_what_the_page_needs(self):
        payload = build_outlook(forecast(+10.0), 10, skill=0.05).as_dict()
        for key in ("direction", "horizon_minutes", "expected_move", "target",
                    "confidence", "reason", "skill", "capped_by_skill"):
            assert key in payload

    def test_neutral_serialises(self):
        assert build_outlook(None, 10).as_dict()["direction"] == NEUTRAL
