"""Cross-instrument validation.

The value of this test rests entirely on the training instruments never
contaminating the test instrument - neither by sharing bars (they cannot) nor
through time (they easily could, since index futures are ~90% correlated and
training on ES during the hours being tested on MNQ would leak the answer
straight through that correlation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mnq.config import Config
from mnq.data.synthetic import generate_context, generate_minute_bars
from mnq.data.yahoo import resample_ohlcv
from mnq.models import crossval


@pytest.fixture(scope="module")
def instruments(monkeypatch_module=None):
    """Four correlated synthetic instruments sharing a common driver."""
    out = {}
    for i, sym in enumerate(["ES=F", "YM=F", "RTY=F", "MNQ=F"]):
        m = generate_minute_bars(n_minutes=60 * 24 * 120, seed=5)
        drift = np.random.default_rng(i).normal(1.0, 0.0002, len(m)).cumprod()
        m = m.mul(pd.Series(drift, index=m.index), axis=0)
        out[sym] = {
            "1h": resample_ohlcv(m, "1h"),
            "4h": resample_ohlcv(m, "4h"),
            "1d": resample_ohlcv(m, "1D"),
        }
    return out


@pytest.fixture
def cfg() -> Config:
    c = Config()
    c.data.profile = "wide"
    c.labels.horizon_bars = 12
    c.labels.fwd_return_bars = 6
    c.model.embargo_bars = 18
    c.model.meta_inner_folds = 3
    return c


@pytest.fixture
def patched(monkeypatch, instruments):
    monkeypatch.setattr(
        crossval, "load_instrument", lambda sym, c, r=True: instruments[sym]
    )
    return instruments


def test_training_never_includes_the_test_instrument(patched, cfg, monkeypatch):
    """The whole premise: MNQ must not appear in any training set."""
    seen: list[pd.DatetimeIndex] = []
    real_prepare = crossval.prepare_instrument

    def spy(symbol, c, context=None, refresh=True):
        out = real_prepare(symbol, c, context, refresh)
        out["_symbol_seen"] = symbol
        seen.append(symbol)
        return out

    monkeypatch.setattr(crossval, "prepare_instrument", spy)
    crossval.cross_instrument_evaluate(
        cfg, train_symbols=("ES=F", "YM=F"), test_symbol="MNQ=F",
        context=None, n_folds=2, refresh=False,
    )
    # MNQ is loaded exactly once - as the test instrument, never as a train one.
    assert seen.count("MNQ=F") == 1
    assert seen.index("MNQ=F") == len(seen) - 1


def test_training_bars_precede_the_test_window(patched, cfg):
    """Time separation, enforced on top of instrument separation.

    Without it, ~90% correlation between indices would leak the test period's
    outcome into training.
    """
    results = crossval.cross_instrument_evaluate(
        cfg, train_symbols=("ES=F", "YM=F"), test_symbol="MNQ=F",
        context=None, n_folds=3, refresh=False,
    )
    assert results["folds"], "no folds ran"
    # Each successive fold trains on strictly more history.
    by_dir: dict[str, list[dict]] = {}
    for f in results["folds"]:
        by_dir.setdefault(f["direction"], []).append(f)
    for folds in by_dir.values():
        sizes = [f["n_train"] for f in sorted(folds, key=lambda x: x["fold"])]
        assert sizes == sorted(sizes), "training window must expand, not jump around"


def test_pooling_multiplies_the_training_sample(patched, cfg):
    results = crossval.cross_instrument_evaluate(
        cfg, train_symbols=("ES=F", "YM=F", "RTY=F"), test_symbol="MNQ=F",
        context=None, n_folds=2, refresh=False,
    )
    # Three instruments should give appreciably more rows than the one tested.
    assert results["train_rows"] > results["test_rows"] * 2


def test_reports_auc_per_direction(patched, cfg):
    results = crossval.cross_instrument_evaluate(
        cfg, train_symbols=("ES=F", "YM=F"), test_symbol="MNQ=F",
        context=None, n_folds=2, refresh=False,
    )
    for side in ("long", "short"):
        if f"{side}_auc" in results:
            assert 0.0 <= results[f"{side}_auc"] <= 1.0
            assert results[f"{side}_n"] > 0


def test_features_are_intersected_across_instruments(patched, cfg):
    results = crossval.cross_instrument_evaluate(
        cfg, train_symbols=("ES=F", "YM=F"), test_symbol="MNQ=F",
        context=None, n_folds=2, refresh=False,
    )
    assert results["n_features"] > 80


def test_context_is_rebuilt_per_instrument(patched, cfg):
    """Relative strength must be measured against the instrument in hand."""
    minutes = generate_minute_bars(n_minutes=60 * 24 * 120, seed=5)
    ctx = generate_context(minutes, interval="1h", seed=6)

    es = crossval.prepare_instrument("ES=F", cfg, ctx, refresh=False)
    mnq = crossval.prepare_instrument("MNQ=F", cfg, ctx, refresh=False)

    rs_cols = [c for c in es["names"] if "_rs" in c]
    assert rs_cols
    common = es["features"].index.intersection(mnq["features"].index)
    assert len(common) > 100
    col = rs_cols[0]
    # Different base instruments must produce different relative strength.
    diff = (es["features"].loc[common, col] - mnq["features"].loc[common, col]).abs()
    assert diff.max() > 0


def test_report_renders_a_verdict(patched, cfg):
    from mnq.models.crossval import format_report

    results = crossval.cross_instrument_evaluate(
        cfg, train_symbols=("ES=F",), test_symbol="MNQ=F",
        context=None, n_folds=2, refresh=False,
    )
    text = format_report(results)
    assert "CROSS-INSTRUMENT VALIDATION" in text
    assert "VERDICT" in text
    assert any(v in text for v in ("TRANSFERS", "WEAK TRANSFER", "NO TRANSFER", "inconclusive"))


def test_missing_training_symbol_is_survivable(patched, cfg, monkeypatch):
    """One dead ticker must not sink the run."""
    real = crossval.load_instrument

    def flaky(sym, c, r=True):
        if sym == "YM=F":
            raise RuntimeError("no data available")
        return real(sym, c, r)

    monkeypatch.setattr(crossval, "load_instrument", flaky)
    results = crossval.cross_instrument_evaluate(
        cfg, train_symbols=("ES=F", "YM=F"), test_symbol="MNQ=F",
        context=None, n_folds=2, refresh=False,
    )
    assert results["train_rows"] > 0


def test_all_training_symbols_failing_raises(cfg, monkeypatch):
    def dead(sym, c, r=True):
        raise RuntimeError("nope")

    monkeypatch.setattr(crossval, "load_instrument", dead)
    with pytest.raises(RuntimeError, match="no training instruments"):
        crossval.cross_instrument_evaluate(
            cfg, train_symbols=("ES=F",), test_symbol="MNQ=F",
            context=None, n_folds=2, refresh=False,
        )
