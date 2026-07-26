"""Indicator correctness and, above all, causality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mnq import indicators as ind


@pytest.fixture
def series() -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(
        21000 + np.cumsum(rng.normal(0, 5, 500)),
        index=pd.date_range("2025-01-01", periods=500, freq="5min", tz="UTC"),
    )


@pytest.fixture
def ohlc() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    close = 21000 + np.cumsum(rng.normal(0, 5, 500))
    high = close + np.abs(rng.normal(0, 3, 500))
    low = close - np.abs(rng.normal(0, 3, 500))
    open_ = close + rng.normal(0, 2, 500)
    return pd.DataFrame(
        {
            "open": open_, "high": np.maximum.reduce([open_, high, close]),
            "low": np.minimum.reduce([open_, low, close]), "close": close,
            "volume": rng.gamma(2, 100, 500),
        },
        index=pd.date_range("2025-01-01", periods=500, freq="5min", tz="UTC"),
    )


def test_ema_matches_recursive_definition(series):
    span = 10
    out = ind.ema(series, span)
    alpha = 2 / (span + 1)
    # First defined value is the SMA-seeded recursion pandas uses with
    # min_periods; check the recursion holds from there on.
    prev = out.iloc[span - 1]
    for i in range(span, span + 20):
        expected = alpha * series.iloc[i] + (1 - alpha) * prev
        assert out.iloc[i] == pytest.approx(expected)
        prev = out.iloc[i]


def test_ema_has_no_lookahead(series):
    """Truncating the input must not change earlier EMA values."""
    full = ind.ema(series, 20)
    partial = ind.ema(series.iloc[:300], 20)
    pd.testing.assert_series_equal(full.iloc[:300], partial, check_freq=False)


def test_slope_recovers_a_known_gradient():
    x = pd.Series(np.arange(100, dtype=float) * 3.0 + 7.0)
    out = ind.slope(x, 20)
    assert out.dropna().round(9).eq(3.0).all()


def test_slope_sign_follows_direction():
    up = pd.Series(np.linspace(100, 200, 60))
    down = pd.Series(np.linspace(200, 100, 60))
    assert ind.slope(up, 20).dropna().gt(0).all()
    assert ind.slope(down, 20).dropna().lt(0).all()


def test_rsi_bounds_and_extremes(series):
    out = ind.rsi(series, 14).dropna()
    assert out.between(0, 100).all()

    # Monotonic advance has no losses, so RSI pins at 100.
    rising = pd.Series(np.arange(60, dtype=float) + 100)
    assert ind.rsi(rising, 14).dropna().iloc[-1] == pytest.approx(100.0)

    falling = pd.Series(200 - np.arange(60, dtype=float))
    assert ind.rsi(falling, 14).dropna().iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_atr_is_positive_and_tracks_range(ohlc):
    a = ind.atr(ohlc["high"], ohlc["low"], ohlc["close"], 14).dropna()
    assert (a > 0).all()
    # ATR must sit inside the observed true-range envelope.
    tr = ind.true_range(ohlc["high"], ohlc["low"], ohlc["close"]).dropna()
    assert a.mean() < tr.max()


def test_true_range_accounts_for_gaps():
    idx = pd.date_range("2025-01-01", periods=2, freq="5min", tz="UTC")
    high = pd.Series([100.0, 130.0], index=idx)
    low = pd.Series([99.0, 128.0], index=idx)
    close = pd.Series([99.5, 129.0], index=idx)
    tr = ind.true_range(high, low, close)
    # Gapped up from 99.5 to a 128-130 bar: TR is 130-99.5, not 130-128.
    assert tr.iloc[1] == pytest.approx(30.5)


def test_macd_histogram_is_line_minus_signal(series):
    m = ind.macd(series)
    diff = (m["macd"] - m["macd_signal"]).dropna()
    pd.testing.assert_series_equal(diff, m["macd_hist"].dropna(), check_names=False)


def test_adx_in_range_and_trends_high_on_a_trend(ohlc):
    out = ind.adx(ohlc["high"], ohlc["low"], ohlc["close"], 14).dropna()
    assert out["adx"].between(0, 100).all()

    n = 200
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    close = pd.Series(np.arange(n, dtype=float) + 100, index=idx)
    trend = ind.adx(close + 1, close - 1, close, 14).dropna()
    # A clean one-way move should register as strongly trending.
    assert trend["adx"].iloc[-1] > 40


def test_bollinger_width_widens_with_volatility():
    idx = pd.date_range("2025-01-01", periods=200, freq="5min", tz="UTC")
    rng = np.random.default_rng(3)
    calm = pd.Series(100 + rng.normal(0, 0.1, 200), index=idx)
    wild = pd.Series(100 + rng.normal(0, 5.0, 200), index=idx)
    assert ind.bollinger_width(wild, 20).dropna().mean() > ind.bollinger_width(calm, 20).dropna().mean()


def test_atr_acceleration_detects_expansion():
    idx = pd.date_range("2025-01-01", periods=60, freq="5min", tz="UTC")
    a = pd.Series(np.concatenate([np.full(30, 10.0), np.full(30, 20.0)]), index=idx)
    accel = ind.atr_acceleration(a, 12)
    # Index 35 sits after the jump while its 12-bar lag (index 23) sits before,
    # so the ratio is 2. By index 59 both ends are post-jump and it returns to 1.
    assert accel.iloc[35] == pytest.approx(2.0)
    assert accel.iloc[-1] == pytest.approx(1.0)


def test_hh_ll_counts_are_bounded(ohlc):
    hh = ind.higher_highs_count(ohlc["high"], 5).dropna()
    ll = ind.lower_lows_count(ohlc["low"], 5).dropna()
    assert hh.between(0, 5).all()
    assert ll.between(0, 5).all()


def test_body_and_wick_ratios_partition_the_bar(ohlc):
    b = ind.body_ratio(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
    w = ind.wick_ratio(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"])
    total = (b + w).dropna()
    # Body plus both wicks is exactly the bar's range.
    assert total.between(0.999, 1.001).all()


def test_doji_and_marubozu_extremes():
    idx = pd.date_range("2025-01-01", periods=1, freq="5min", tz="UTC")
    o = pd.Series([100.0], index=idx)
    c = pd.Series([100.0], index=idx)
    h = pd.Series([105.0], index=idx)
    l = pd.Series([95.0], index=idx)
    assert ind.body_ratio(o, h, l, c).iloc[0] == pytest.approx(0.0)

    o2, c2 = pd.Series([95.0], index=idx), pd.Series([105.0], index=idx)
    assert ind.body_ratio(o2, h, l, c2).iloc[0] == pytest.approx(1.0)


def test_zero_range_bar_does_not_divide_by_zero():
    idx = pd.date_range("2025-01-01", periods=1, freq="5min", tz="UTC")
    flat = pd.Series([100.0], index=idx)
    out = ind.body_ratio(flat, flat, flat, flat)
    assert out.isna().all()  # undefined, not inf


def test_relative_volume_centres_on_one():
    idx = pd.date_range("2025-01-01", periods=100, freq="5min", tz="UTC")
    v = pd.Series(np.full(100, 500.0), index=idx)
    assert ind.relative_volume(v, 20).dropna().eq(1.0).all()
