"""Cross-asset context features.

The same rule that governs higher-timeframe price features governs these: a
context bar is invisible until it has closed. Cross-asset data makes leakage
*easier* to introduce, because the other instrument's bars sit on a different
clock and it is tempting to align on the nearest stamp rather than the last
completed one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mnq.config import Config
from mnq.data.context import ContextConfig, DEFAULT_BASKET, describe_basket, fetch_context
from mnq.data.synthetic import generate_context, generate_minute_bars
from mnq.data.yahoo import resample_ohlcv
from mnq.features.builder import WIDE_TIMEFRAMES, build_feature_matrix
from mnq.features.context import build_context_features, context_feature_columns


@pytest.fixture(scope="module")
def minutes() -> pd.DataFrame:
    return generate_minute_bars(n_minutes=60 * 24 * 40, seed=31)


@pytest.fixture(scope="module")
def base(minutes) -> pd.DataFrame:
    return resample_ohlcv(minutes, "5min")


@pytest.fixture(scope="module")
def context(minutes) -> dict[str, pd.DataFrame]:
    return generate_context(minutes, interval="1h", seed=32)


@pytest.fixture(scope="module")
def ctx_features(base, context) -> pd.DataFrame:
    return build_context_features(
        base.index, base["close"], context, ContextConfig(), base_minutes=5
    )


def test_context_features_are_produced(ctx_features):
    cols = context_feature_columns(ctx_features)
    assert len(cols) > 40
    assert all(c.startswith("ctx_") for c in cols)


def test_expected_feature_families_exist(ctx_features):
    cols = set(ctx_features.columns)
    assert any(c.endswith("_corr") for c in cols), "rolling correlations missing"
    assert any("_rs" in c for c in cols), "relative strength missing"
    assert "ctx_risk_appetite" in cols
    assert "ctx_vix_percentile" in cols


def test_context_values_come_only_from_closed_bars(base, context):
    """A 1h context bar spanning 10:00-11:00 must not reach the 10:05 base bar."""
    cfg = ContextConfig()
    feats = build_context_features(base.index, base["close"], context, cfg, 5)

    es = context["es"]
    span = pd.Timedelta(hours=1)
    col = "ctx_es_ret1"
    assert col in feats

    # Rebuild the raw per-symbol series the merge should have drawn from.
    raw = 100.0 * (es["close"] / es["close"].shift(1) - 1.0)

    checked = 0
    for ts in base.index[::211]:
        value = feats.at[ts, col]
        if pd.isna(value):
            continue
        available = raw[raw.index + span <= ts + pd.Timedelta(minutes=5)].dropna()
        if available.empty:
            continue
        assert value == pytest.approx(available.iloc[-1]), (
            f"{col} at {ts} used a context bar that had not closed"
        )
        checked += 1
    assert checked > 20


def test_truncating_context_does_not_change_past_features(base, context):
    """Removing future context bars must not alter earlier feature values."""
    cfg = ContextConfig()
    full = build_context_features(base.index, base["close"], context, cfg, 5)

    cutoff = base.index[int(len(base) * 0.7)]
    truncated_ctx = {k: v[v.index <= cutoff] for k, v in context.items()}
    truncated_base = base[base.index <= cutoff]
    partial = build_context_features(
        truncated_base.index, truncated_base["close"], truncated_ctx, cfg, 5
    )

    common = partial.index.intersection(full.index)
    assert len(common) > 500
    cols = context_feature_columns(full)
    a, b = full.loc[common, cols], partial.loc[common, cols]
    assert (a.isna() == b.isna()).all().all()
    pd.testing.assert_frame_equal(a, b, check_freq=False, rtol=1e-9)


def test_context_does_not_predict_the_next_bar_directly(base, ctx_features):
    """No context feature should correlate strongly with MNQ's next move."""
    future = base["close"].pct_change().shift(-1)
    suspicious = []
    for col in context_feature_columns(ctx_features):
        pair = pd.concat([ctx_features[col], future], axis=1).dropna()
        if len(pair) < 200 or pair.iloc[:, 0].nunique() < 5:
            continue
        corr = abs(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
        if corr > 0.30:
            suspicious.append((col, round(float(corr), 3)))
    assert not suspicious, f"context features leaking the next bar: {suspicious}"


def test_missing_symbols_degrade_gracefully(base, context):
    """A partial basket must still produce features, not raise."""
    partial = {k: context[k] for k in list(context)[:3]}
    feats = build_context_features(base.index, base["close"], partial, ContextConfig(), 5)
    assert len(context_feature_columns(feats)) > 5


def test_empty_context_returns_an_empty_block(base):
    feats = build_context_features(base.index, base["close"], {}, ContextConfig(), 5)
    assert feats.empty or feats.shape[1] == 0
    assert list(feats.index) == list(base.index)


def test_fetch_context_refuses_a_mostly_failed_basket(tmp_path):
    """Training on 2 of 12 symbols should be a loud failure, not a quiet one."""
    cfg = ContextConfig(symbols=DEFAULT_BASKET[:3], min_symbols=4)
    # refresh=False with an empty cache means nothing loads.
    with pytest.raises(RuntimeError, match="context symbols loaded"):
        fetch_context(cfg, tmp_path, refresh=False)


def test_disabled_context_returns_nothing(tmp_path):
    cfg = ContextConfig(enabled=False)
    assert fetch_context(cfg, tmp_path, refresh=False) == {}


def test_basket_description_lists_every_symbol():
    text = describe_basket(ContextConfig())
    for sym in DEFAULT_BASKET:
        assert sym.ticker in text


# ------------------------------------------------------------ wide profile


def test_wide_profile_builds_from_hourly_base(minutes):
    frames = {
        "1h": resample_ohlcv(minutes, "1h"),
        "4h": resample_ohlcv(minutes, "4h"),
        "1d": resample_ohlcv(minutes, "1D"),
    }
    matrix = build_feature_matrix(frames, Config().features, WIDE_TIMEFRAMES)
    assert len(matrix) == len(frames["1h"])
    assert any(c.startswith("tf1h_") for c in matrix.columns)
    assert any(c.startswith("tf1d_") for c in matrix.columns)


def test_wide_profile_higher_timeframes_do_not_leak(minutes):
    """The truncation test, applied to the hourly stack."""
    frames = {
        "1h": resample_ohlcv(minutes, "1h"),
        "4h": resample_ohlcv(minutes, "4h"),
        "1d": resample_ohlcv(minutes, "1D"),
    }
    cfg = Config().features
    full = build_feature_matrix(frames, cfg, WIDE_TIMEFRAMES)

    cutoff = full.index[int(len(full) * 0.7)]
    truncated = {k: v[v.index <= cutoff] for k, v in frames.items()}
    partial = build_feature_matrix(truncated, cfg, WIDE_TIMEFRAMES)

    common = partial.index.intersection(full.index)
    cols = [c for c in full.columns if c.startswith(("tf1h_", "tf4h_", "tf1d_"))]
    a, b = full.loc[common, cols], partial.loc[common, cols]
    assert (a.isna() == b.isna()).all().all()
    pd.testing.assert_frame_equal(a, b, check_freq=False, rtol=1e-9)


def test_profile_selection_and_validation():
    cfg = Config()
    assert cfg.data.timeframes()[0].key == "5m"
    cfg.data.profile = "wide"
    assert cfg.data.timeframes()[0].key == "1h"
    cfg.data.profile = "nonsense"
    with pytest.raises(ValueError, match="unknown data profile"):
        cfg.data.timeframes()


def test_prepare_accepts_context_end_to_end(minutes, context):
    from mnq.models.train import prepare

    cfg = Config()
    frames = {
        "5m": resample_ohlcv(minutes, "5min"),
        "15m": resample_ohlcv(minutes, "15min"),
        "4h": resample_ohlcv(minutes, "4h"),
    }
    with_ctx = prepare(frames, cfg, context)
    without = prepare(frames, cfg, None)
    assert len(with_ctx.feature_names) > len(without.feature_names)
    assert any(c.startswith("ctx_") for c in with_ctx.feature_names)
    assert not any(c.startswith("ctx_") for c in without.feature_names)
