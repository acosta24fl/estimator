"""Datetime resolution consistency.

A real run died with

    MergeError: incompatible merge keys [0] datetime64[us, UTC] and
                datetime64[s, UTC], must be the same type

after every symbol had downloaded successfully, surfacing as "no training
instruments could be loaded". The cause was a fresh yfinance download (second
resolution) merged against a cached CSV (microsecond resolution).

No existing test caught it because synthetic frames and CSV fixtures agree on
a unit. Only mixing a *live-shaped* frame with a *cache-shaped* one disagrees,
so that is exactly what these tests build.
"""

import numpy as np
import pandas as pd
import pytest

from mnq.config import Config
from mnq.features.builder import build_feature_matrix, feature_columns
from mnq.timeutil import UTC_NS, align_merge_keys, normalise_index, to_utc_ns

RESOLUTIONS = ["s", "ms", "us", "ns"]


def _frame(periods=400, freq="5min", unit="us", price=15_000.0, seed=1):
    """An OHLCV frame stamped at a specific datetime resolution."""
    idx = pd.date_range("2024-01-02", periods=periods, freq=freq, tz="UTC")
    idx = idx.astype(f"datetime64[{unit}, UTC]")
    rng = np.random.default_rng(seed)
    close = price + np.cumsum(rng.normal(0, 3, periods))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 4,
            "low": close - 4,
            "close": close,
            "volume": 1_000.0,
        },
        index=idx,
    )


class TestToUtcNs:
    @pytest.mark.parametrize("unit", RESOLUTIONS)
    def test_every_resolution_normalises(self, unit):
        idx = pd.date_range("2024-01-02", periods=3, freq="1h", tz="UTC")
        got = to_utc_ns(idx.astype(f"datetime64[{unit}, UTC]"))
        assert str(got.dtype) == UTC_NS

    def test_naive_input_is_treated_as_utc(self):
        got = to_utc_ns(pd.DatetimeIndex(["2024-01-02 00:00", "2024-01-02 01:00"]))
        assert str(got.dtype) == UTC_NS
        assert str(got[0]) == "2024-01-02 00:00:00+00:00"

    def test_other_timezones_convert_rather_than_relabel(self):
        idx = pd.DatetimeIndex(["2024-01-02 09:30"], tz="America/New_York")
        got = to_utc_ns(idx)
        assert str(got[0]) == "2024-01-02 14:30:00+00:00"

    def test_instants_survive_the_cast(self):
        idx = pd.date_range("2024-01-02", periods=5, freq="1h", tz="UTC")
        assert (to_utc_ns(idx.astype("datetime64[s, UTC]")) == idx).all()

    def test_strings_parse(self):
        assert str(to_utc_ns(["2024-01-02T00:00:00Z"]).dtype) == UTC_NS


class TestNormaliseIndex:
    @pytest.mark.parametrize("unit", RESOLUTIONS)
    def test_frame_index_normalises(self, unit):
        got = normalise_index(_frame(periods=5, unit=unit))
        assert str(got.index.dtype) == UTC_NS
        assert got.index.name == "timestamp"

    def test_empty_frame_is_returned_untouched(self):
        empty = pd.DataFrame()
        assert normalise_index(empty) is empty

    def test_already_canonical_frame_is_not_copied(self):
        df = _frame(periods=5, unit="ns")
        assert normalise_index(df) is df


class TestAlignMergeKeys:
    def test_reproduces_and_fixes_the_original_failure(self):
        """The exact pairing that killed the crossval run."""
        left = pd.DataFrame(
            {"_asof": pd.DatetimeIndex(["2024-01-02", "2024-01-03"], tz="UTC")
             .astype("datetime64[s, UTC]")}
        )
        right = pd.DataFrame(
            {
                "_available_at": pd.DatetimeIndex(["2024-01-02"], tz="UTC")
                .astype("datetime64[us, UTC]"),
                "x": [1.0],
            }
        )

        # Confirm the failure is real before asserting the fix, so this test
        # cannot quietly pass on a pandas version that stops caring.
        with pytest.raises(Exception):
            pd.merge_asof(
                left, right, left_on="_asof", right_on="_available_at",
                direction="backward",
            )

        a, b = align_merge_keys(left, "_asof", right, "_available_at")
        merged = pd.merge_asof(
            a, b, left_on="_asof", right_on="_available_at", direction="backward"
        )
        assert merged["x"].tolist() == [1.0, 1.0]

    @pytest.mark.parametrize("left_unit", RESOLUTIONS)
    @pytest.mark.parametrize("right_unit", RESOLUTIONS)
    def test_every_pairing_merges(self, left_unit, right_unit):
        left = pd.DataFrame(
            {"_asof": pd.DatetimeIndex(["2024-01-02"], tz="UTC")
             .astype(f"datetime64[{left_unit}, UTC]")}
        )
        right = pd.DataFrame(
            {
                "_available_at": pd.DatetimeIndex(["2024-01-02"], tz="UTC")
                .astype(f"datetime64[{right_unit}, UTC]"),
                "x": [7.0],
            }
        )
        a, b = align_merge_keys(left, "_asof", right, "_available_at")
        merged = pd.merge_asof(
            a, b, left_on="_asof", right_on="_available_at", direction="backward"
        )
        assert merged["x"].iloc[0] == 7.0


class TestFeatureMatrixAcrossResolutions:
    """The end-to-end path that actually broke."""

    def test_mixed_resolution_frames_build_a_matrix(self):
        """Base timeframe live-shaped, higher timeframes cache-shaped."""
        cfg = Config()
        frames = {
            "5m": _frame(periods=600, freq="5min", unit="s"),
            "15m": _frame(periods=200, freq="15min", unit="us"),
            "4h": _frame(periods=40, freq="4h", unit="ns"),
        }
        matrix = build_feature_matrix(frames, cfg.features)
        assert len(matrix) > 0
        assert len(feature_columns(matrix)) > 50
        assert str(matrix.index.dtype) == UTC_NS

    def test_result_matches_a_uniform_resolution_build(self):
        """Resolution must change nothing about the numbers."""
        cfg = Config()
        uniform = {
            "5m": _frame(periods=600, freq="5min", unit="ns"),
            "15m": _frame(periods=200, freq="15min", unit="ns"),
            "4h": _frame(periods=40, freq="4h", unit="ns"),
        }
        mixed = {
            "5m": _frame(periods=600, freq="5min", unit="s"),
            "15m": _frame(periods=200, freq="15min", unit="us"),
            "4h": _frame(periods=40, freq="4h", unit="ms"),
        }
        a = build_feature_matrix(uniform, cfg.features)
        b = build_feature_matrix(mixed, cfg.features)
        cols = feature_columns(a)
        pd.testing.assert_frame_equal(a[cols], b[cols], check_freq=False)


class TestCrossvalErrorReporting:
    """The failure message has to point at the real cause.

    The original run reported "no training instruments could be loaded" after
    truncating each symbol's error at 80 characters - which cut
    "...must be the same type" in half. The visible message named three
    symbols as unavailable, so it read as a Yahoo outage rather than a dtype
    bug in this repository.
    """

    def _evaluate(self, monkeypatch, errors):
        from mnq.models import crossval

        def fake_prepare(symbol, cfg, context, refresh):
            raise errors[symbol]

        monkeypatch.setattr(crossval, "prepare_instrument", fake_prepare)
        return crossval

    def test_identical_failures_are_called_out_as_systemic(self, monkeypatch):
        from mnq.config import Config

        err = pd.errors.MergeError(
            "incompatible merge keys [0] datetime64[us, UTC] and "
            "datetime64[s, UTC], must be the same type"
        )
        crossval = self._evaluate(
            monkeypatch, {s: err for s in ("ES=F", "YM=F", "RTY=F")}
        )
        with pytest.raises(RuntimeError) as excinfo:
            crossval.cross_instrument_evaluate(
                Config(), train_symbols=("ES=F", "YM=F", "RTY=F"),
                test_symbol="MNQ=F", context={}, n_folds=2, refresh=False,
            )
        message = str(excinfo.value)
        assert "systemic" in message
        # The whole error, not the first 80 characters of it.
        assert "must be the same type" in message

    def test_differing_failures_are_listed_per_symbol(self, monkeypatch):
        from mnq.config import Config

        crossval = self._evaluate(
            monkeypatch,
            {
                "ES=F": RuntimeError("Yahoo returned no rows"),
                "YM=F": ValueError("missing columns ['volume']"),
            },
        )
        with pytest.raises(RuntimeError) as excinfo:
            crossval.cross_instrument_evaluate(
                Config(), train_symbols=("ES=F", "YM=F"),
                test_symbol="MNQ=F", context={}, n_folds=2, refresh=False,
            )
        message = str(excinfo.value)
        assert "systemic" not in message
        assert "Yahoo returned no rows" in message
        assert "missing columns" in message
