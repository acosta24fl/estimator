import math

from app.core.prediction_log import Prediction, ScoredPrediction
from app.core.trading import (
    DEFAULT_COST_POINTS,
    DOLLARS_PER_POINT,
    bootstrap_profit_factor,
    sharpe,
    simulate,
)


def scored(anchor, target, actual, ts=0):
    prediction = Prediction(
        anchor_ts=ts, target_ts=ts + 300, anchor_price=anchor, target=target,
        low=target - 5, high=target + 5,
        direction="up" if target > anchor else "down",
    )
    return ScoredPrediction(
        prediction=prediction, actual=actual, error=actual - target,
        direction_correct=(actual - anchor > 0) == (target - anchor > 0),
        within_band=abs(actual - target) <= 5,
    )


class TestSimulate:
    def test_a_correct_long_wins(self):
        stats, trades = simulate([scored(100, 110, 108)], cost_points=0.0)
        assert trades[0].direction == 1
        assert trades[0].gross == 8.0
        assert stats.wins == 1

    def test_a_correct_short_wins(self):
        stats, trades = simulate([scored(100, 90, 94)], cost_points=0.0)
        assert trades[0].direction == -1
        assert trades[0].gross == 6.0  # short from 100, price fell to 94
        assert stats.wins == 1

    def test_a_wrong_call_loses(self):
        stats, _ = simulate([scored(100, 110, 95)], cost_points=0.0)
        assert stats.losses == 1
        assert stats.gross_loss == 5.0

    def test_costs_are_charged_per_trade(self):
        _, trades = simulate([scored(100, 110, 108)], cost_points=0.75)
        assert trades[0].net == 8.0 - 0.75

    def test_costs_can_turn_a_winner_into_a_loser(self):
        stats, _ = simulate([scored(100, 110, 100.5)], cost_points=0.75)
        assert stats.losses == 1

    def test_threshold_filters_small_projections(self):
        items = [scored(100, 100.5, 101), scored(100, 120, 121)]
        stats, _ = simulate(items, cost_points=0.0, threshold=5.0)
        assert stats.trades == 1

    def test_a_flat_projection_is_never_traded(self):
        stats, _ = simulate([scored(100, 100, 105)], cost_points=0.0)
        assert stats.trades == 0

    def test_profit_factor_is_gross_profit_over_gross_loss(self):
        items = [scored(100, 110, 120), scored(100, 110, 95)]  # +20, -5
        stats, _ = simulate(items, cost_points=0.0)
        assert math.isclose(stats.profit_factor, 20 / 5)

    def test_profit_factor_is_none_without_losses(self):
        stats, _ = simulate([scored(100, 110, 120)], cost_points=0.0)
        assert stats.profit_factor is None

    def test_dollars_follow_the_contract_multiplier(self):
        stats, _ = simulate([scored(100, 110, 110)], cost_points=0.0)
        assert stats.net_dollars == 10.0 * DOLLARS_PER_POINT

    def test_max_drawdown_tracks_the_worst_dip(self):
        items = [scored(100, 110, 120), scored(100, 110, 90), scored(100, 110, 111)]
        stats, _ = simulate(items, cost_points=0.0)  # +20, -10, +11
        assert stats.max_drawdown == 10.0

    def test_empty_input(self):
        stats, trades = simulate([], cost_points=DEFAULT_COST_POINTS)
        assert stats.trades == 0 and trades == []


class TestRequiredWinRate:
    def test_symmetric_bet_needs_two_thirds_for_pf_two(self):
        """PF 2.0 on an even-money bet requires a 66.7% win rate."""
        items = [scored(100, 110, 110), scored(100, 110, 90)]  # +10 win, -10 loss
        stats, _ = simulate(items, cost_points=0.0)
        assert math.isclose(stats.required_win_rate_for(2.0), 2 / 3, abs_tol=1e-9)

    def test_pf_one_needs_a_coin_flip(self):
        items = [scored(100, 110, 110), scored(100, 110, 90)]
        stats, _ = simulate(items, cost_points=0.0)
        assert math.isclose(stats.required_win_rate_for(1.0), 0.5, abs_tol=1e-9)

    def test_bigger_winners_lower_the_required_win_rate(self):
        items = [scored(100, 110, 130), scored(100, 110, 95)]  # +30 win, -5 loss
        stats, _ = simulate(items, cost_points=0.0)
        assert stats.required_win_rate_for(2.0) < 0.5


class TestBootstrap:
    @staticmethod
    def _trades(nets):
        from app.core.trading import Trade

        return [Trade(i, 1, 0.0, 0.0, n, n) for i, n in enumerate(nets)]

    def test_too_few_trades_returns_nothing(self):
        assert bootstrap_profit_factor(self._trades([1.0] * 5)) is None

    def test_interval_brackets_the_point_estimate(self):
        nets = [5.0, -3.0, 4.0, -2.0, 6.0, -4.0] * 20
        low, high = bootstrap_profit_factor(self._trades(nets))
        point = sum(n for n in nets if n > 0) / -sum(n for n in nets if n < 0)
        assert low <= point <= high

    def test_more_trades_give_a_tighter_interval(self):
        pattern = [5.0, -3.0, 4.0, -6.0]
        small = bootstrap_profit_factor(self._trades(pattern * 5))
        large = bootstrap_profit_factor(self._trades(pattern * 200))
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_a_small_sample_interval_is_wide_enough_to_be_useless(self):
        """The whole point: 30 noisy trades cannot establish an edge."""
        import random

        random.seed(4)
        nets = [random.gauss(0.5, 10.0) for _ in range(30)]
        low, high = bootstrap_profit_factor(self._trades(nets))
        assert low < 1.0 < high


class TestSharpe:
    def test_needs_at_least_two_trades(self):
        from app.core.trading import Trade

        assert sharpe([Trade(0, 1, 0, 0, 1, 1)]) is None

    def test_constant_returns_have_no_deviation(self):
        from app.core.trading import Trade

        assert sharpe([Trade(i, 1, 0, 0, 1, 1) for i in range(5)]) is None

    def test_positive_for_a_profitable_series(self):
        from app.core.trading import Trade

        trades = [Trade(i, 1, 0, 0, n, n) for i, n in enumerate([2.0, 1.0, 3.0, -1.0])]
        assert sharpe(trades) > 0
