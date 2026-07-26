"""Validation splitters, bar store, webhook and the end-to-end pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from mnq.config import Config
from mnq.data.store import BarStore
from mnq.data.synthetic import generate_frames, generate_minute_bars
from mnq.data.yahoo import merge_cache, resample_ohlcv
from mnq.models.validation import PurgedKFoldInner, PurgedWalkForward


# --------------------------------------------------------------- splitters


def test_walk_forward_never_trains_on_the_future():
    for train, test in PurgedWalkForward(n_folds=5, initial_train_frac=0.5, embargo=10).split(1000):
        assert train.max() < test.min()


def test_walk_forward_purges_the_embargo_gap():
    embargo = 25
    for train, test in PurgedWalkForward(n_folds=4, initial_train_frac=0.5, embargo=embargo).split(1000):
        assert test.min() - train.max() > embargo - 1


def test_walk_forward_windows_expand_and_do_not_overlap():
    splits = list(PurgedWalkForward(n_folds=4, initial_train_frac=0.5, embargo=10).split(1000))
    assert len(splits) == 4
    sizes = [len(tr) for tr, _ in splits]
    assert sizes == sorted(sizes)
    seen: set[int] = set()
    for _, test in splits:
        assert not seen & set(test.tolist())
        seen |= set(test.tolist())


def test_walk_forward_rejects_impossible_configuration():
    with pytest.raises(ValueError):
        list(PurgedWalkForward(n_folds=5, initial_train_frac=1.5).split(1000))


def test_inner_folds_purge_on_both_sides():
    """An inner test block has training data after it as well as before."""
    embargo = 20
    for train, test in PurgedKFoldInner(n_folds=4, embargo=embargo).split(1000):
        assert not set(train.tolist()) & set(test.tolist())
        gap = np.abs(train[:, None] - test[None, :]).min()
        assert gap >= embargo


def test_inner_folds_cover_the_whole_sample():
    covered: set[int] = set()
    for _, test in PurgedKFoldInner(n_folds=4, embargo=5).split(1000):
        covered |= set(test.tolist())
    assert len(covered) == 1000


# -------------------------------------------------------------- bar store


def test_store_accumulates_and_resamples():
    store = BarStore()
    minutes = generate_minute_bars(n_minutes=300, seed=4)
    base = datetime.now(timezone.utc) - timedelta(minutes=400)
    for i, (_, row) in enumerate(minutes.iterrows()):
        store.add_bar(base + timedelta(minutes=i), row.open, row.high, row.low, row.close, row.volume)
    assert len(store) == 300
    frames = store.frames()
    assert len(frames["5m"]) == pytest.approx(60, abs=2)
    assert len(frames["15m"]) == pytest.approx(20, abs=2)


def test_store_excludes_the_in_progress_minute():
    store = BarStore()
    now = datetime.now(timezone.utc)
    store.add_bar(now - timedelta(minutes=2), 1, 2, 0.5, 1.5)
    store.add_bar(now, 1, 2, 0.5, 1.5)          # current, still forming
    assert len(store.minute_frame()) == 1
    assert len(store.minute_frame(include_current=True)) == 2


def test_add_price_builds_a_bar_from_prints():
    store = BarStore()
    ts = datetime.now(timezone.utc) - timedelta(minutes=5)
    for p in (100.0, 105.0, 95.0, 102.0):
        store.add_price(ts, p, volume=1.0)
    df = store.minute_frame()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["open"] == 100.0 and row["high"] == 105.0
    assert row["low"] == 95.0 and row["close"] == 102.0
    assert row["volume"] == 4.0


def test_store_normalises_naive_timestamps():
    """Mixing naive and aware stamps must not create duplicate minutes."""
    store = BarStore()
    naive = datetime(2025, 1, 1, 12, 0)
    aware = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    store.add_bar(naive, 1, 2, 0.5, 1.5)
    store.add_bar(aware, 1, 2, 0.5, 1.6)
    assert len(store) == 1


def test_store_round_trips_through_disk(tmp_path):
    path = tmp_path / "bars.csv"
    store = BarStore(path)
    base = datetime.now(timezone.utc) - timedelta(minutes=50)
    for i in range(30):
        store.add_bar(base + timedelta(minutes=i), 100 + i, 101 + i, 99 + i, 100.5 + i, 10)
    store.flush()
    assert len(BarStore(path)) == 30


def test_store_trims_to_the_retention_limit():
    store = BarStore(max_bars=50)
    base = datetime.now(timezone.utc) - timedelta(minutes=200)
    for i in range(120):
        store.add_bar(base + timedelta(minutes=i), 1, 2, 0.5, 1.5)
    assert len(store) == 50


# ------------------------------------------------------------ yahoo utils


def test_resample_stamps_bars_at_their_open():
    idx = pd.date_range("2025-01-01 00:00", periods=15, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {"open": range(15), "high": range(15), "low": range(15),
         "close": range(15), "volume": [1.0] * 15},
        index=idx,
    ).astype(float)
    out = resample_ohlcv(df, "5min")
    assert list(out.index.minute) == [0, 5, 10]
    assert out["open"].iloc[0] == 0.0
    assert out["close"].iloc[0] == 4.0     # bar 00:00 covers minutes 0-4
    assert out["volume"].iloc[0] == 5.0


def test_merge_cache_prefers_fresh_rows():
    idx = pd.date_range("2025-01-01", periods=3, freq="5min", tz="UTC")
    old = pd.DataFrame({"open": [1.0]*3, "high": [1.0]*3, "low": [1.0]*3,
                        "close": [1.0]*3, "volume": [1.0]*3}, index=idx)
    new = pd.DataFrame({"open": [2.0], "high": [2.0], "low": [2.0],
                        "close": [2.0], "volume": [2.0]}, index=idx[-1:])
    merged = merge_cache(old, new)
    assert len(merged) == 3
    assert merged["close"].iloc[-1] == 2.0


# --------------------------------------------------------------- webhook


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from mnq.server.app import create_app
    from mnq.server.engine import LiveEngine

    cfg = Config()
    cfg.server.webhook_secret = "topsecret"
    cfg.telegram.enabled = False
    cfg.server.bar_store_path = str(tmp_path / "bars.csv")
    cfg.server.state_path = str(tmp_path / "state.json")
    engine = LiveEngine(cfg, store=BarStore(tmp_path / "bars.csv"), load_models=False)
    return TestClient(create_app(cfg, engine)), engine


def test_webhook_requires_the_secret(client):
    c, _ = client
    assert c.post("/webhook", json={"close": 21000.0}).status_code == 401
    assert c.post("/webhook", json={"close": 21000.0, "secret": "wrong"}).status_code == 401
    assert c.post("/webhook", json={"close": 21000.0, "secret": "topsecret"}).status_code == 200


def test_webhook_accepts_the_secret_as_a_header(client):
    c, _ = client
    r = c.post("/webhook", json={"close": 21000.0}, headers={"x-webhook-secret": "topsecret"})
    assert r.status_code == 200


def test_webhook_stores_a_full_bar(client):
    c, engine = client
    ts = (datetime.now(timezone.utc) - timedelta(minutes=3)).isoformat()
    c.post("/webhook", json={
        "secret": "topsecret", "time": ts,
        "open": 21000.0, "high": 21010.0, "low": 20990.0, "close": 21005.0, "volume": 500,
    })
    row = engine.store.minute_frame().iloc[-1]
    assert row["high"] == 21010.0 and row["low"] == 20990.0


def test_webhook_handles_a_close_only_alert(client):
    c, engine = client
    ts = (datetime.now(timezone.utc) - timedelta(minutes=3)).isoformat()
    assert c.post("/webhook", json={"secret": "topsecret", "time": ts, "close": 21000.0}).status_code == 200
    assert engine.store.last_price() == 21000.0


@pytest.mark.parametrize("body,expected", [
    ("21123.75", 200),
    ('{"close": 21123.75}', 200),
    ("LONG 21050.25", 200),
    ("no price at all", 400),
])
def test_raw_webhook_parsing(client, body, expected):
    c, _ = client
    r = c.post("/webhook/raw", content=body, headers={"x-webhook-secret": "topsecret"})
    assert r.status_code == expected


def test_status_endpoint_reports_state(client):
    c, _ = client
    body = c.get("/status").json()
    assert body["models_loaded"] is False
    assert "bars" in body


# --------------------------------------------------------- pipeline smoke


def test_prepare_produces_aligned_features_and_labels():
    from mnq.labeling import LONG, SHORT
    from mnq.models.train import prepare

    cfg = Config()
    data = prepare(generate_frames(n_minutes=60 * 24 * 25, seed=8), cfg)
    assert len(data.features) == len(data.matrix) == len(data.fwd_return)
    for direction in (LONG, SHORT):
        assert len(data.labels[direction]) == len(data.features)
        assert data.labels[direction].index.equals(data.features.index)
    assert len(data.feature_names) > 80
    # The matrix must carry the OHLC the backtester needs.
    for col in ("open", "high", "low", "close", "atr"):
        assert col in data.matrix.columns


def test_backtest_runs_and_reports(tmp_path):
    """Full loop on forced signals: the engine must open, manage and close."""
    from mnq.backtest.engine import run_backtest
    from mnq.models.train import prepare

    cfg = Config()
    cfg.trade.min_probability = 0.50
    cfg.trade.cooldown_bars = 3
    data = prepare(generate_frames(n_minutes=60 * 24 * 25, seed=8), cfg)

    # Deterministic pseudo-predictions: no model needed to test the mechanics.
    rng = np.random.default_rng(0)
    preds = pd.DataFrame(
        {
            "long_p_meta": rng.uniform(0.3, 0.8, len(data.matrix)),
            "short_p_meta": rng.uniform(0.3, 0.8, len(data.matrix)),
        },
        index=data.matrix.index,
    )
    result = run_backtest(data.matrix, preds, cfg)
    assert result.metrics["n_trades"] > 0
    assert "Trades" in result.report()
    # Every trade must be closed and internally consistent.
    for t in result.trades:
        assert t.exit_time is not None and t.exit_price is not None
        assert t.exit_reason
        assert t.entry_time <= t.exit_time


def test_backtest_rejects_predictions_that_do_not_overlap():
    from mnq.backtest.engine import run_backtest
    from mnq.models.train import prepare

    cfg = Config()
    data = prepare(generate_frames(n_minutes=60 * 24 * 20, seed=8), cfg)
    stale = pd.DataFrame(
        {"long_p_meta": [0.9], "short_p_meta": [0.1]},
        index=pd.DatetimeIndex(["1999-01-01"], tz="UTC"),
    )
    with pytest.raises(ValueError):
        run_backtest(data.matrix, stale, cfg)


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("trade:\n  not_a_real_setting: 1\n")
    with pytest.raises(ValueError):
        Config.load(path)


def test_config_reads_secrets_from_the_environment(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc123")
    monkeypatch.setenv("TV_WEBHOOK_SECRET", "shh")
    cfg = Config.load()
    assert cfg.telegram.bot_token == "abc123"
    assert cfg.server.webhook_secret == "shh"
    # And must never echo them back out.
    assert cfg.to_dict()["telegram"]["bot_token"] == "***"
    assert cfg.to_dict()["server"]["webhook_secret"] == "***"


def test_wilson_bound_penalises_small_samples():
    from mnq.backtest.sweep import _wilson_lower_bound

    small = _wilson_lower_bound(4, 5)      # 80% from 5 trades
    large = _wilson_lower_bound(280, 500)  # 56% from 500 trades
    assert large > small
    assert 0.0 <= small <= 0.8
