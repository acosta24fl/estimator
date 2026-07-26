"""Lookahead tests.

These are the tests that decide whether any other number in this project means
anything. A backtest built on leaked data can show any Sharpe you like.

The strongest available check is the truncation test: rebuild the features using
only the bars available at time T, and confirm the resulting row is identical to
the one produced when the whole future was present. If a feature peeks ahead,
the two disagree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mnq.config import FeatureConfig, LabelConfig
from mnq.data.synthetic import generate_frames
from mnq.features.builder import (
    TIMEFRAME_MINUTES, build_feature_matrix, feature_columns,
)
from mnq.labeling import LONG, SHORT, build_labels


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    return generate_frames(n_minutes=60 * 24 * 30, seed=17)


@pytest.fixture(scope="module")
def matrix(frames) -> pd.DataFrame:
    return build_feature_matrix(frames, FeatureConfig())


def test_truncating_history_does_not_change_past_features(frames):
    """The decisive test: features at time T must not depend on data after T."""
    cfg = FeatureConfig()
    full = build_feature_matrix(frames, cfg)

    cutoff = full.index[int(len(full) * 0.7)]
    truncated_frames = {
        tf: df[df.index <= cutoff] for tf, df in frames.items()
    }
    partial = build_feature_matrix(truncated_frames, cfg)

    cols = feature_columns(full)
    common = partial.index.intersection(full.index)
    assert len(common) > 500, "truncation left too few rows to compare"

    a = full.loc[common, cols]
    b = partial.loc[common, cols]
    # NaNs must match position-for-position too.
    assert (a.isna() == b.isna()).all().all(), "NaN pattern changed under truncation"
    pd.testing.assert_frame_equal(a, b, check_freq=False, rtol=1e-9)


@pytest.mark.parametrize("tf,prefix", [("15m", "tf15_"), ("4h", "tf4h_")])
def test_higher_timeframe_values_are_only_used_after_they_close(frames, matrix, tf, prefix):
    """A 15m bar spanning 10:00-10:15 must not appear on the 5m bar at 10:05."""
    from mnq.features.builder import build_timeframe_features

    htf_feats = build_timeframe_features(frames[tf], FeatureConfig(), prefix)
    span = pd.Timedelta(minutes=TIMEFRAME_MINUTES[tf])

    # Pick a representative feature that changes every bar.
    col = f"{prefix}rsi"
    assert col in matrix.columns

    checked = 0
    for ts in matrix.index[::97]:
        value = matrix.at[ts, col]
        if pd.isna(value):
            continue
        # Which higher-timeframe bars had closed by this 5m bar's close?
        available = htf_feats[htf_feats.index + span <= ts + pd.Timedelta(minutes=5)]
        if available.empty or available[col].dropna().empty:
            continue
        expected = available[col].dropna().iloc[-1]
        assert value == pytest.approx(expected), (
            f"{col} at {ts} used a bar that had not closed yet"
        )
        checked += 1
    assert checked > 30, "test did not actually compare enough rows"


def test_no_feature_correlates_perfectly_with_the_future(matrix):
    """A feature that mirrors the next bar's return is a leak, by definition."""
    future_ret = matrix["close"].pct_change().shift(-1)
    cols = feature_columns(matrix)
    sample = matrix[cols].iloc[200:]
    fut = future_ret.iloc[200:]

    suspicious = []
    for col in sample.columns:
        s = sample[col]
        pair = pd.concat([s, fut], axis=1).dropna()
        if len(pair) < 200 or pair.iloc[:, 0].nunique() < 5:
            continue
        corr = abs(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
        if corr > 0.30:
            suspicious.append((col, round(float(corr), 3)))
    assert not suspicious, f"features suspiciously predictive of the next bar: {suspicious}"


def test_label_entry_is_the_next_bar_open(matrix):
    labels = build_labels(matrix, LabelConfig(), LONG)
    expected = matrix["open"].shift(-1)
    got = labels["entry_price"]
    pd.testing.assert_series_equal(
        got.dropna(), expected.loc[got.dropna().index], check_names=False
    )


def test_labels_never_resolve_before_entry(matrix):
    """Exit must land strictly after the entry bar."""
    for direction in (LONG, SHORT):
        labels = build_labels(matrix, LabelConfig(), direction)
        resolved = labels[labels["exit_index"] >= 0]
        positions = matrix.index.get_indexer(resolved.index)
        assert (resolved["exit_index"].to_numpy() > positions).all()


def test_labels_respect_the_horizon(matrix):
    cfg = LabelConfig()
    labels = build_labels(matrix, cfg, LONG)
    held = labels["bars_held"].dropna()
    assert held.min() >= 1
    assert held.max() <= cfg.horizon_bars


def test_same_bar_touching_both_barriers_is_scored_a_loss():
    """The pessimistic tie-break: OHLC hides the intra-bar path."""
    idx = pd.date_range("2025-01-01", periods=6, freq="5min", tz="UTC")
    # Bar 2 spans a huge range covering both the target and the stop.
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 140.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 60.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "volume": [1.0] * 6,
            "atr": [10.0] * 6,
        },
        index=idx,
    )
    cfg = LabelConfig(horizon_bars=3, tp_atr_mult=2.0, sl_atr_mult=1.0, min_target_points=1.0)
    labels = build_labels(df, cfg, LONG)
    # Entry at bar 1's open (100); bar 2 hits both +20 and -10.
    assert labels["label"].iloc[0] == 0.0
    assert labels["outcome"].iloc[0] == "sl"


def test_long_and_short_barriers_are_mirrored():
    idx = pd.date_range("2025-01-01", periods=4, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100.0] * 4, "high": [101.0] * 4, "low": [99.0] * 4,
            "close": [100.0] * 4, "volume": [1.0] * 4, "atr": [10.0] * 4,
        },
        index=idx,
    )
    cfg = LabelConfig(horizon_bars=2, tp_atr_mult=2.0, sl_atr_mult=1.0, min_target_points=1.0)
    long_lab = build_labels(df, cfg, LONG)
    short_lab = build_labels(df, cfg, SHORT)
    assert long_lab["tp_price"].iloc[0] == pytest.approx(120.0)
    assert long_lab["sl_price"].iloc[0] == pytest.approx(90.0)
    assert short_lab["tp_price"].iloc[0] == pytest.approx(80.0)
    assert short_lab["sl_price"].iloc[0] == pytest.approx(110.0)


def test_targets_below_the_minimum_are_not_labelled():
    idx = pd.date_range("2025-01-01", periods=10, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100.0] * 10, "high": [101.0] * 10, "low": [99.0] * 10,
            "close": [100.0] * 10, "volume": [1.0] * 10,
            "atr": [1.0] * 10,  # 2 x 1.0 = 2 points, far below the 20 minimum
        },
        index=idx,
    )
    labels = build_labels(df, LabelConfig(min_target_points=20.0), LONG)
    assert labels["label"].isna().all()
    assert not labels["tradeable"].any()
