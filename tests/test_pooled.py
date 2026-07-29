"""Pooled multi-instrument training.

Pooling is only legitimate if every fold trains exclusively on bars that
precede its test window - for *all* instruments, including the target itself.
Index futures are ~90% correlated, so training on ES during the hours being
tested on MNQ would leak the answer through that correlation, and it would
look like a spectacular result rather than a bug.

These tests stub the models out and inspect exactly which rows reach ``fit``.
"""

import numpy as np
import pandas as pd
import pytest

from mnq.config import Config
from mnq.labeling import LONG, SHORT
from mnq.models import pooled


@pytest.fixture
def cfg():
    c = Config()
    c.data.profile = "wide"
    c.context.enabled = False
    c.model.embargo_bars = 36
    return c


def _instrument(symbol, n=1200, seed=0, start="2024-01-02"):
    """A minimal prepare_instrument payload: two features, both directions."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    names = ["tf1h_a", "tf1h_b"]
    feats = pd.DataFrame(rng.normal(size=(n, 2)), index=idx, columns=names)
    labels = {
        d: pd.DataFrame({"label": rng.integers(0, 2, n).astype(float)}, index=idx)
        for d in (LONG, SHORT)
    }
    matrix = feats.assign(
        open=15_000.0, high=15_010.0, low=14_990.0, close=15_000.0,
        volume=1_000.0, atr=40.0,
    )
    return {
        "symbol": symbol,
        "matrix": matrix,
        "features": feats,
        "labels": labels,
        "fwd": pd.Series(rng.normal(size=n), index=idx),
        "names": names,
    }


class _Recorder:
    """Stands in for the ensemble, recording the training index it is given."""

    seen: list[pd.DatetimeIndex] = []

    def __init__(self, *a, **k):
        # save_bundle records these alongside the model; the real
        # DirectionalEnsemble sets them during fit.
        self.train_stats: dict = {}

    def fit(self, X, *a, **k):
        _Recorder.seen.append(X.index)
        return self

    def predict(self, X):
        return np.zeros(len(X))

    def predict_components(self, X, fwd):
        return pd.DataFrame(
            {
                "p_meta": np.linspace(0.4, 0.6, len(X)),
                "p_xgb": 0.5, "p_lgbm": 0.5, "fwd_pred": 0.0,
            },
            index=X.index,
        )


@pytest.fixture
def stubbed(monkeypatch):
    """Fast pooled run: real fold logic, stub models."""
    _Recorder.seen = []
    pool = {
        "MNQ=F": _instrument("MNQ=F", seed=1),
        "ES=F": _instrument("ES=F", seed=2),
        "YM=F": _instrument("YM=F", seed=3),
    }
    monkeypatch.setattr(
        pooled, "prepare_instrument",
        lambda sym, cfg, ctx, refresh: pool[sym],
    )
    monkeypatch.setattr(pooled, "DirectionalEnsemble", _Recorder)
    monkeypatch.setattr(pooled, "SharedRegressor", _Recorder)
    return pool


class TestNoLookahead:
    def test_training_never_reaches_into_the_test_window(self, cfg, stubbed):
        """The property the whole approach rests on."""
        _, _, metrics = pooled.pooled_walk_forward(
            cfg, pool=("MNQ=F", "ES=F", "YM=F"), n_folds=3, refresh=False,
        )
        assert metrics["folds"], "expected at least one fold"

        embargo = pd.Timedelta(hours=cfg.model.embargo_bars)
        # Recorder pairs: regressor then ensemble, per fold.
        starts = [pd.Timestamp(f["test_start"]) for f in metrics["folds"]]
        for train_idx, start in zip(_Recorder.seen[::2], starts):
            assert train_idx.max() < start - embargo, (
                f"fold tested from {start} but trained on bars up to "
                f"{train_idx.max()}"
            )

    def test_the_target_instrument_is_time_separated_too(self, cfg, stubbed):
        """MNQ's own earlier bars are fair; its later ones are not.

        This is the subtle half. Excluding other instruments' future bars but
        letting the target's own through would leak directly.
        """
        _, _, metrics = pooled.pooled_walk_forward(
            cfg, pool=("MNQ=F", "ES=F", "YM=F"), n_folds=3, refresh=False,
        )
        embargo = pd.Timedelta(hours=cfg.model.embargo_bars)
        target_idx = stubbed["MNQ=F"]["features"].index

        for fold, train_idx in zip(metrics["folds"], _Recorder.seen[::2]):
            cutoff = pd.Timestamp(fold["test_start"]) - embargo
            from_target = train_idx.intersection(target_idx)
            assert len(from_target) > 0, "the target should contribute training rows"
            assert from_target.max() < cutoff

    def test_embargo_is_actually_applied(self, cfg, stubbed):
        """A zero embargo must admit strictly more rows than a wide one."""
        cfg.model.embargo_bars = 0
        _Recorder.seen = []
        pooled.pooled_walk_forward(cfg, pool=("MNQ=F", "ES=F"), n_folds=2, refresh=False)
        wide_open = max(len(i) for i in _Recorder.seen)

        cfg.model.embargo_bars = 200
        _Recorder.seen = []
        pooled.pooled_walk_forward(cfg, pool=("MNQ=F", "ES=F"), n_folds=2, refresh=False)
        embargoed = max(len(i) for i in _Recorder.seen)

        assert embargoed < wide_open


class TestPooling:
    def test_pooling_multiplies_the_training_sample(self, cfg, stubbed):
        _Recorder.seen = []
        pooled.pooled_walk_forward(cfg, pool=("MNQ=F",), n_folds=2, refresh=False)
        alone = max(len(i) for i in _Recorder.seen)

        _Recorder.seen = []
        pooled.pooled_walk_forward(
            cfg, pool=("MNQ=F", "ES=F", "YM=F"), n_folds=2, refresh=False
        )
        together = max(len(i) for i in _Recorder.seen)

        # Three instruments over the same span: roughly three times the rows.
        assert together > 2.5 * alone

    def test_the_target_is_added_when_missing_from_the_pool(self, cfg, stubbed):
        _, _, metrics = pooled.pooled_walk_forward(
            cfg, pool=("ES=F", "YM=F"), target="MNQ=F", n_folds=2, refresh=False,
        )
        assert "MNQ=F" in metrics["pool"]

    def test_predictions_are_indexed_on_the_target(self, cfg, stubbed):
        _, preds, _ = pooled.pooled_walk_forward(
            cfg, pool=("MNQ=F", "ES=F"), n_folds=2, refresh=False
        )
        assert not preds.empty
        assert "long_p_meta" in preds and "short_p_meta" in preds
        assert preds.index.isin(stubbed["MNQ=F"]["features"].index).all()
        assert preds.index.is_monotonic_increasing
        assert not preds.index.has_duplicates

    def test_training_rows_grow_with_each_fold(self, cfg, stubbed):
        _, _, metrics = pooled.pooled_walk_forward(
            cfg, pool=("MNQ=F", "ES=F"), n_folds=3, refresh=False
        )
        longs = [f["n_train"] for f in metrics["folds"] if f["direction"] == "long"]
        assert longs == sorted(longs), "expanding window should not shrink"

    def test_every_instrument_contributes(self, cfg, stubbed):
        _, _, metrics = pooled.pooled_walk_forward(
            cfg, pool=("MNQ=F", "ES=F", "YM=F"), n_folds=2, refresh=False
        )
        for fold in metrics["folds"]:
            assert set(fold["train_by_symbol"]) == {"MNQ=F", "ES=F", "YM=F"}


class TestFailureModes:
    def test_an_unloadable_target_is_fatal(self, cfg, monkeypatch):
        def boom(sym, cfg, ctx, refresh):
            raise RuntimeError("Yahoo returned no rows")

        monkeypatch.setattr(pooled, "prepare_instrument", boom)
        with pytest.raises(RuntimeError, match="nothing to predict"):
            pooled.pooled_walk_forward(cfg, pool=("MNQ=F", "ES=F"), refresh=False)

    def test_a_missing_training_symbol_is_survivable(self, cfg, monkeypatch):
        _Recorder.seen = []
        ok = {"MNQ=F": _instrument("MNQ=F", seed=1)}

        def maybe(sym, cfg, ctx, refresh):
            if sym not in ok:
                raise RuntimeError(f"{sym} unavailable")
            return ok[sym]

        monkeypatch.setattr(pooled, "prepare_instrument", maybe)
        monkeypatch.setattr(pooled, "DirectionalEnsemble", _Recorder)
        monkeypatch.setattr(pooled, "SharedRegressor", _Recorder)

        _, preds, metrics = pooled.pooled_walk_forward(
            cfg, pool=("MNQ=F", "GONE=F"), n_folds=2, refresh=False
        )
        assert metrics["pool"] == ["MNQ=F"]
        assert "GONE=F" in metrics["unavailable"]
        assert not preds.empty

    def test_report_renders_without_crashing(self, cfg, stubbed):
        _, _, metrics = pooled.pooled_walk_forward(
            cfg, pool=("MNQ=F", "ES=F"), n_folds=2, refresh=False
        )
        text = pooled.format_report(metrics)
        assert "POOLED MULTI-INSTRUMENT TRAINING" in text
        assert "MNQ=F" in text


class TestTheProductionModelIsActuallySaved:
    """Pooled training measured an edge and left no model behind.

    ``pooled_walk_forward`` fits a model per fold to *measure* the edge and
    discards each one - correct, because a fold model has only seen data up to
    its own cutoff. But nothing then fitted the model that gets used live.
    Training reported a profitable backtest, wrote its predictions, and never
    produced ``ensemble.joblib``, so the dashboard said "no trained model"
    however many times option 3 was run. The failure was completely silent:
    every number on screen looked right.
    """

    def test_a_pooled_run_writes_a_loadable_bundle(self, cfg, stubbed, tmp_path):
        from mnq.models.train import load_bundle

        cfg.model_dir = str(tmp_path / "models")
        _, _, metrics = pooled.pooled_walk_forward(
            cfg, pool=("MNQ=F", "ES=F", "YM=F"), n_folds=3, refresh=False,
            fit_final=True,
        )

        assert metrics.get("final_model"), metrics.get("final_model_error")
        assert (tmp_path / "models" / "ensemble.joblib").exists()

        bundle = load_bundle(cfg)
        assert set(bundle["directions"]) == {"long", "short"}
        assert bundle["shared_regressor"] is not None
        assert bundle["feature_names"] == ["tf1h_a", "tf1h_b"]

    def test_it_is_off_by_default_so_evaluation_stays_side_effect_free(
        self, cfg, stubbed, tmp_path
    ):
        cfg.model_dir = str(tmp_path / "models")
        _, _, metrics = pooled.pooled_walk_forward(
            cfg, pool=("MNQ=F", "ES=F", "YM=F"), n_folds=3, refresh=False,
        )
        assert "final_model" not in metrics
        assert not (tmp_path / "models" / "ensemble.joblib").exists()

    def test_the_final_fit_uses_every_pooled_row(self, cfg, stubbed, tmp_path):
        """Unlike the fold models, this one trains on all history.

        That is right for a production model and wrong for measurement, which
        is exactly why they are separate code paths.
        """
        common = ["tf1h_a", "tf1h_b"]
        bundle = pooled.fit_pooled_bundle(stubbed, common, cfg)

        total = sum(len(d["features"]) for d in stubbed.values())
        assert bundle["pooled_rows"] == total
        assert bundle["pool"] == sorted(stubbed)

    def test_a_failed_final_fit_does_not_discard_the_walk_forward(
        self, cfg, stubbed, tmp_path, monkeypatch
    ):
        """An hour of evaluation must survive a broken production fit."""
        cfg.model_dir = str(tmp_path / "models")

        def boom(*a, **k):
            raise RuntimeError("out of memory")

        monkeypatch.setattr(pooled, "fit_pooled_bundle", boom)
        _, preds, metrics = pooled.pooled_walk_forward(
            cfg, pool=("MNQ=F", "ES=F", "YM=F"), n_folds=3, refresh=False,
            fit_final=True,
        )

        assert "out of memory" in metrics["final_model_error"]
        assert metrics.get("long_auc") is not None, "the evaluation must survive"
        assert not preds.empty
