"""Profitability arithmetic.

The question this module answers is not "did the backtest end positive" but
"is the per-trade edge bigger than its own error bar, after costs". These
tests pin down the cases where those two answers disagree, because that gap is
where a backtest misleads.
"""

import numpy as np
import pandas as pd
import pytest

from mnq.backtest.profit import (
    MIN_TRADES_FOR_INFERENCE,
    expectancy,
    round_turn_cost_usd,
    target_coverage,
)
from mnq.config import Config


def _trades(net, start="2024-01-02", hours=6):
    """A trade log with the columns the report reads."""
    n = len(net)
    entry = pd.date_range(start, periods=n, freq=f"{hours}h", tz="UTC")
    return pd.DataFrame(
        {
            "entry_time": entry,
            "exit_time": entry + pd.Timedelta(hours=2),
            "net_usd": np.asarray(net, dtype=float),
        }
    )


class TestRoundTurnCost:
    def test_matches_the_contract_spec(self):
        cfg = Config()
        # 1 tick/side x 0.25 pt x 2 sides x $2/pt = $1.00, plus 2 x $0.37.
        assert round_turn_cost_usd(cfg) == pytest.approx(1.74)

    def test_scales_with_slippage(self):
        cfg = Config()
        cfg.trade.slippage_ticks = 2.0
        assert round_turn_cost_usd(cfg) == pytest.approx(2.74)


class TestExpectancy:
    def test_no_trades_is_reported_not_crashed(self):
        got = expectancy(pd.DataFrame(), Config())
        assert got["n_trades"] == 0 and got["profitable"] is False

    def test_breakeven_win_rate_follows_the_payoff_ratio(self):
        # 40 wins of +$100, 60 losses of -$50 -> payoff 2:1 -> breakeven 33.3%.
        got = expectancy(_trades([100.0] * 40 + [-50.0] * 60), Config())
        assert got["payoff_ratio"] == pytest.approx(2.0)
        assert got["breakeven_win_rate"] == pytest.approx(1 / 3, abs=1e-6)
        assert got["win_rate"] == pytest.approx(0.40)
        assert got["win_rate_margin"] == pytest.approx(0.40 - 1 / 3, abs=1e-6)

    def test_a_losing_edge_is_not_profitable(self):
        # 25% win rate at 2:1 is below the 33% breakeven.
        got = expectancy(_trades([100.0] * 25 + [-50.0] * 75), Config())
        assert got["expectancy_usd"] < 0
        assert got["profitable"] is False
        assert "expectancy is" in got["reason"]

    def test_a_clear_edge_is_profitable(self):
        got = expectancy(_trades([100.0] * 60 + [-50.0] * 60), Config())
        assert got["expectancy_usd"] == pytest.approx(25.0)
        assert got["profitable"] is True
        assert got["ci95_low_usd"] > 0

    def test_positive_total_with_an_interval_spanning_zero_is_rejected(self):
        """The case the plain total cannot see.

        One large winner among many small losers sums positive while the
        per-trade expectancy is indistinguishable from noise. Reporting that
        as profitable is exactly the failure this module exists to prevent.
        """
        net = [-40.0] * 59 + [3_000.0]
        got = expectancy(_trades(net), Config())
        assert got["total_net_usd"] > 0          # the headline looks good
        assert got["expectancy_usd"] > 0         # so does the mean
        assert got["ci95_low_usd"] < 0           # but it is not distinguishable
        assert got["profitable"] is False
        assert "includes zero" in got["reason"]

    def test_too_few_trades_is_unmeasured_not_profitable(self):
        got = expectancy(_trades([100.0] * 5 + [-10.0] * 5), Config())
        assert got["n_trades"] < MIN_TRADES_FOR_INFERENCE
        assert got["profitable"] is False
        assert "need" in got["reason"]

    def test_costs_are_reported_against_gross(self):
        got = expectancy(_trades([10.0] * 50 + [-5.0] * 50), Config())
        # Gross = net expectancy + the round-turn cost already deducted.
        assert got["gross_expectancy_usd"] == pytest.approx(
            got["expectancy_usd"] + round_turn_cost_usd(Config())
        )
        assert 0 < got["cost_share_of_gross"] < 1

    def test_expectancy_in_points_matches_dollars(self):
        got = expectancy(_trades([100.0] * 50 + [-50.0] * 50), Config())
        assert got["expectancy_points"] == pytest.approx(got["expectancy_usd"] / 2.0)

    def test_monthly_projection_uses_the_observed_rate(self):
        # 120 trades, one every 6h -> 30 days -> ~120/month.
        got = expectancy(_trades([10.0] * 60 + [-5.0] * 60, hours=6), Config())
        assert 100 < got["trades_per_month"] < 140
        assert got["expected_monthly_usd"] == pytest.approx(
            got["expectancy_usd"] * got["trades_per_month"]
        )

    def test_all_winners_gives_infinite_payoff_without_crashing(self):
        got = expectancy(_trades([50.0] * 40), Config())
        assert not np.isfinite(got["payoff_ratio"])
        assert got["profitable"] is True

    def test_p_value_is_one_sided_and_small_for_a_strong_edge(self):
        got = expectancy(_trades([100.0] * 80 + [-20.0] * 40), Config())
        assert got["p_value"] < 0.001
        assert got["t_stat"] > 3


class TestTargetCoverage:
    def _matrix(self, atr):
        idx = pd.date_range("2024-01-02", periods=len(atr), freq="1h", tz="UTC")
        return pd.DataFrame({"atr": atr}, index=idx)

    def test_reports_the_share_clearing_the_minimum(self):
        cfg = Config()  # tp_atr_mult 2.0, min_edge_points 25
        # ATR 20 -> target 40 (clears); ATR 5 -> target 10 (does not).
        m = self._matrix([20.0] * 60 + [5.0] * 40)
        got = target_coverage(m, cfg)
        assert got["min_points"] == 25.0
        assert got["share_clearing"] == pytest.approx(0.60)

    def test_uses_an_explicit_override(self):
        m = self._matrix([20.0] * 100)
        assert target_coverage(m, Config(), min_points=100.0)["share_clearing"] == 0.0

    def test_empty_matrix_is_safe(self):
        assert target_coverage(pd.DataFrame(), Config())["bars"] == 0
