import math
import random

from app.core.features import (
    ewma_volatility,
    garman_klass_variance,
    volatility_persistence,
)
from app.core.forecast import band_half_width
from app.models import Bar


def bar(o, h, l, c, ts=0):
    return Bar(ts=ts, open=o, high=h, low=l, close=c, volume=100, complete=True)


def flat_series(n=400, half=1.0, price=100.0):
    return [bar(price, price + half, price - half, price, i * 300) for i in range(n)]


class TestGarmanKlass:
    def test_a_zero_range_bar_has_zero_variance(self):
        assert garman_klass_variance(bar(100, 100, 100, 100)) == 0.0

    def test_a_wider_bar_has_more_variance(self):
        assert garman_klass_variance(bar(100, 105, 95, 100)) > garman_klass_variance(
            bar(100, 101, 99, 100)
        )

    def test_never_negative(self):
        # A bar that closes far from its open with a narrow range can drive the
        # raw formula below zero; it must be clamped.
        assert garman_klass_variance(bar(100, 110, 90, 110)) >= 0.0

    def test_invalid_prices_are_rejected(self):
        assert garman_klass_variance(bar(0, 0, 0, 0)) == 0.0
        assert garman_klass_variance(bar(100, 100, -1, 100)) == 0.0


class TestEwmaVolatility:
    def test_one_value_per_bar(self):
        assert len(ewma_volatility(flat_series(50))) == 50

    def test_steady_series_gives_steady_volatility(self):
        series = ewma_volatility(flat_series(200))
        assert math.isclose(series[-1], series[-2], rel_tol=1e-6)

    def test_it_reacts_to_a_shock(self):
        calm = flat_series(200)
        before = ewma_volatility(calm)[-1]
        after = ewma_volatility(calm + [bar(100, 140, 60, 100, 200 * 300)])[-1]
        assert after > before * 3

    def test_a_lower_lambda_reacts_faster(self):
        calm = flat_series(200) + [bar(100, 140, 60, 100, 200 * 300)]
        assert ewma_volatility(calm, lam=0.80)[-1] > ewma_volatility(calm, lam=0.99)[-1]

    def test_empty_input(self):
        assert ewma_volatility([]) == []


class TestVolatilityPersistence:
    def test_a_constant_series_has_no_persistence(self):
        """Floating-point noise must not read as near-perfect persistence."""
        assert volatility_persistence(flat_series(300)) == 0.0

    def test_clustered_volatility_is_detected(self):
        series = []
        for i in range(400):
            half = 5.0 if (i // 40) % 2 else 0.5
            series.append(bar(100, 100 + half, 100 - half, 100, i * 300))
        assert volatility_persistence(series) > 0.5

    def test_randomly_varying_volatility_scores_low(self):
        random.seed(3)
        series = [
            bar(100, 100 + h, 100 - h, 100, i * 300)
            for i, h in enumerate(random.uniform(0.5, 5.0) for _ in range(400))
        ]
        assert volatility_persistence(series) < 0.35

    def test_result_is_bounded(self):
        for series in (flat_series(300), flat_series(30)):
            assert 0.0 <= volatility_persistence(series) <= 1.0

    def test_too_little_data_returns_zero(self):
        assert volatility_persistence(flat_series(5)) == 0.0


class TestBandHalfWidth:
    @staticmethod
    def drifting(n=500, step=3.0):
        series = []
        price = 20_000.0
        random.seed(9)
        for i in range(n):
            move = random.gauss(0, step)
            close = price + move
            series.append(bar(price, max(price, close) + 2, min(price, close) - 2, close, i * 300))
            price = close
        return series

    def test_positive_and_finite(self):
        width = band_half_width(self.drifting())
        assert width > 0 and math.isfinite(width)

    def test_too_little_data_is_zero(self):
        assert band_half_width(flat_series(2)) == 0.0

    def test_coverage_is_near_the_nominal_quantile(self):
        """The band is built for 68%; it must actually deliver roughly that."""
        series = self.drifting(700)
        hits = 0
        total = 0
        for i in range(400, len(series) - 1):
            width = band_half_width(series[: i + 1])
            if abs(series[i + 1].close - series[i].close) <= width:
                hits += 1
            total += 1
        assert 0.58 <= hits / total <= 0.78

    def test_a_calm_series_gets_a_narrower_band_than_a_wild_one(self):
        calm = self.drifting(500, step=1.0)
        wild = self.drifting(500, step=10.0)
        assert band_half_width(calm) < band_half_width(wild)
