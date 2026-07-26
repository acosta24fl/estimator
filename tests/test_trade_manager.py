"""Trade lifecycle rules.

The backtest is only trustworthy if these rules behave exactly as described,
because the same code runs the live position.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mnq.config import LabelConfig, TICK_SIZE, TradeConfig
from mnq.trade.manager import (
    EXIT_EARLY, EXIT_STOP, EXIT_TARGET, EXIT_TIME, EXIT_TRAIL,
    LONG, SHORT, Bar, Trade, TradeManager,
)
from mnq.trade.signals import build_signal, select_direction

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def bar(i: int, o: float, h: float, l: float, c: float, atr: float = 10.0) -> Bar:
    return Bar(timestamp=T0 + timedelta(minutes=5 * i), open=o, high=h, low=l, close=c, atr=atr)


def make_trade(direction: int = LONG, entry: float = 100.0) -> Trade:
    risk, reward = 10.0, 20.0
    return Trade(
        direction=direction,
        entry_time=T0,
        entry_price=entry,
        initial_stop=entry - risk * direction,
        target=entry + reward * direction,
        atr_at_entry=10.0,
    )


@pytest.fixture
def cfg() -> TradeConfig:
    # Management rules off by default; each test enables what it exercises.
    return TradeConfig(
        breakeven_at_r=0.0, trail_atr_mult=0.0, early_exit_prob=0.0,
        max_hold_bars=1000, enable_monitor=True,
    )


def test_long_stop_closes_the_trade(cfg):
    t = make_trade(LONG)
    TradeManager(cfg).on_bar(t, bar(1, 100, 101, 89, 90))
    assert not t.is_open
    assert t.exit_reason == EXIT_STOP
    assert t.exit_price == pytest.approx(90.0)


def test_long_target_closes_the_trade(cfg):
    t = make_trade(LONG)
    TradeManager(cfg).on_bar(t, bar(1, 100, 121, 99, 120))
    assert t.exit_reason == EXIT_TARGET
    assert t.exit_price == pytest.approx(120.0)


def test_short_barriers_are_mirrored(cfg):
    t = make_trade(SHORT)          # entry 100, stop 110, target 80
    TradeManager(cfg).on_bar(t, bar(1, 100, 101, 79, 80))
    assert t.exit_reason == EXIT_TARGET
    assert t.realised_points == pytest.approx(20.0)

    t2 = make_trade(SHORT)
    TradeManager(cfg).on_bar(t2, bar(1, 100, 111, 99, 110))
    assert t2.exit_reason == EXIT_STOP
    assert t2.realised_points == pytest.approx(-10.0)


def test_stop_wins_when_one_bar_spans_both_barriers(cfg):
    """Pessimistic by design: the path inside the bar is unknown."""
    t = make_trade(LONG)
    TradeManager(cfg).on_bar(t, bar(1, 100, 130, 85, 100))
    assert t.exit_reason == EXIT_STOP


def test_gap_through_the_stop_fills_at_the_open(cfg):
    """A stop is an order, not a guarantee; a gap fills worse."""
    t = make_trade(LONG)           # stop at 90
    TradeManager(cfg).on_bar(t, bar(1, 80, 82, 78, 81))
    assert t.exit_price == pytest.approx(80.0)
    assert t.realised_points == pytest.approx(-20.0)


def test_breakeven_moves_the_stop_above_entry(cfg):
    cfg.breakeven_at_r = 1.0
    t = make_trade(LONG)           # risk 10, so +1R is 110
    m = TradeManager(cfg)
    m.on_bar(t, bar(1, 100, 111, 99, 110))
    assert t.is_open
    assert t.breakeven_moved
    assert t.stop == pytest.approx(100.0 + TICK_SIZE)

    # A pullback to entry now closes for a tick, not a full stop-out.
    m.on_bar(t, bar(2, 105, 106, 99, 100))
    assert not t.is_open
    assert t.exit_reason == EXIT_TRAIL
    assert t.realised_points > 0


def test_trailing_stop_only_ratchets_forward(cfg):
    cfg.trail_atr_mult = 1.0
    cfg.trail_start_r = 1.0
    m = TradeManager(cfg)
    t = make_trade(LONG)

    m.on_bar(t, bar(1, 100, 112, 99, 111, atr=10.0))
    assert t.trailing_active
    first = t.stop
    assert first == pytest.approx(102.0)   # extreme 112 minus 1 ATR

    # A quieter bar must not loosen the stop back down.
    m.on_bar(t, bar(2, 111, 111.5, 105, 106, atr=10.0))
    assert t.stop == pytest.approx(first)

    # A new high advances it.
    m.on_bar(t, bar(3, 106, 118, 105, 117, atr=10.0))
    assert t.stop == pytest.approx(108.0)


def test_trailing_waits_for_the_start_threshold(cfg):
    cfg.trail_atr_mult = 1.0
    cfg.trail_start_r = 1.5     # needs +15 points
    t = make_trade(LONG)
    TradeManager(cfg).on_bar(t, bar(1, 100, 108, 99, 107, atr=10.0))
    assert not t.trailing_active
    assert t.stop == pytest.approx(90.0)


def test_early_exit_fires_when_confidence_collapses(cfg):
    cfg.early_exit_prob = 0.35
    cfg.early_exit_min_bars = 2
    m = TradeManager(cfg)
    t = make_trade(LONG)

    # Below the threshold, but too early to act.
    m.on_bar(t, bar(1, 100, 103, 98, 102), continuation_prob=0.10)
    assert t.is_open

    m.on_bar(t, bar(2, 102, 103, 98, 101), continuation_prob=0.10)
    assert not t.is_open
    assert t.exit_reason == EXIT_EARLY
    assert t.exit_price == pytest.approx(101.0)   # exits at the close


def test_confident_model_keeps_the_trade_open(cfg):
    cfg.early_exit_prob = 0.35
    cfg.early_exit_min_bars = 1
    t = make_trade(LONG)
    TradeManager(cfg).on_bar(t, bar(1, 100, 103, 98, 102), continuation_prob=0.80)
    assert t.is_open


def test_missing_probability_disables_early_exit(cfg):
    """No score is not the same as a bad score."""
    cfg.early_exit_prob = 0.99
    cfg.early_exit_min_bars = 1
    t = make_trade(LONG)
    TradeManager(cfg).on_bar(t, bar(1, 100, 103, 98, 102), continuation_prob=None)
    assert t.is_open


def test_time_stop_flattens_a_stagnant_trade(cfg):
    cfg.max_hold_bars = 3
    cfg.time_stop_r = 0.5
    m = TradeManager(cfg)
    t = make_trade(LONG)
    for i in range(1, 4):
        m.on_bar(t, bar(i, 100, 101, 99, 100))
    assert not t.is_open
    assert t.exit_reason == EXIT_TIME


def test_time_stop_spares_a_winning_trade(cfg):
    cfg.max_hold_bars = 3
    cfg.time_stop_r = 0.5   # +5 points or better survives
    m = TradeManager(cfg)
    t = make_trade(LONG)
    for i in range(1, 5):
        m.on_bar(t, bar(i, 108, 109, 107, 108))
    assert t.is_open


def test_excursions_are_tracked(cfg):
    m = TradeManager(cfg)
    t = make_trade(LONG)
    m.on_bar(t, bar(1, 100, 108, 96, 104))
    assert t.mfe_points == pytest.approx(8.0)
    assert t.mae_points == pytest.approx(4.0)


def test_r_multiple_uses_original_risk_not_the_trailed_stop(cfg):
    cfg.breakeven_at_r = 1.0
    m = TradeManager(cfg)
    t = make_trade(LONG)
    m.on_bar(t, bar(1, 100, 112, 99, 111))
    assert t.risk_points == pytest.approx(10.0)      # unchanged by the move
    assert t.r_multiple(120.0) == pytest.approx(2.0)


def test_net_usd_charges_slippage_and_commission():
    tcfg = TradeConfig(slippage_ticks=1.0, commission_usd_per_side=0.37, contracts=1)
    t = make_trade(LONG)
    t.exit_price = 120.0
    t.exit_time = T0
    # 20 points, minus 2 ticks of slippage (0.5 pts), at $2/pt, minus $0.74 fees.
    assert t.net_usd(tcfg) == pytest.approx((20 - 0.5) * 2.0 - 0.74)


def test_a_closed_trade_ignores_further_bars(cfg):
    m = TradeManager(cfg)
    t = make_trade(LONG)
    m.on_bar(t, bar(1, 100, 121, 99, 120))
    held, price = t.bars_held, t.exit_price
    m.on_bar(t, bar(2, 120, 130, 60, 70))
    assert t.bars_held == held and t.exit_price == price


# --------------------------------------------------------------- signals


def test_signal_geometry_and_tick_rounding():
    sig = build_signal(
        T0, LONG, entry=21000.13, atr=25.0, probability=0.7,
        lcfg=LabelConfig(), tcfg=TradeConfig(min_probability=0.5, min_edge_points=20),
    )
    assert sig is not None
    assert sig.entry % TICK_SIZE == pytest.approx(0.0)
    assert sig.stop % TICK_SIZE == pytest.approx(0.0)
    assert sig.target % TICK_SIZE == pytest.approx(0.0)
    assert sig.target > sig.entry > sig.stop
    assert sig.reward_risk == pytest.approx(2.0)


def test_signal_rejected_below_the_probability_gate():
    assert build_signal(
        T0, LONG, 21000.0, 25.0, probability=0.51,
        lcfg=LabelConfig(), tcfg=TradeConfig(min_probability=0.60),
    ) is None


def test_signal_rejected_when_the_move_is_too_small():
    """ATR 5 gives a 10-point target: not worth the risk after costs."""
    assert build_signal(
        T0, LONG, 21000.0, 5.0, probability=0.9,
        lcfg=LabelConfig(), tcfg=TradeConfig(min_probability=0.5, min_edge_points=20.0),
    ) is None


def test_signal_rejected_on_bad_atr():
    tcfg = TradeConfig(min_probability=0.5)
    assert build_signal(T0, LONG, 21000.0, 0.0, 0.9, LabelConfig(), tcfg) is None
    assert build_signal(T0, LONG, 21000.0, float("nan"), 0.9, LabelConfig(), tcfg) is None


def test_direction_selection():
    tcfg = TradeConfig(min_probability=0.58)
    assert select_direction(0.70, 0.20, tcfg)[0] == LONG
    assert select_direction(0.20, 0.70, tcfg)[0] == SHORT
    assert select_direction(0.50, 0.50, tcfg) is None
    # Both fire and agree closely: genuine disagreement, so stand aside.
    assert select_direction(0.60, 0.61, tcfg) is None
    # Both fire but one clearly dominates.
    assert select_direction(0.85, 0.59, tcfg)[0] == LONG


def test_direction_selection_tolerates_nan():
    tcfg = TradeConfig(min_probability=0.58)
    assert select_direction(float("nan"), 0.70, tcfg)[0] == SHORT
    assert select_direction(float("nan"), float("nan"), tcfg) is None
