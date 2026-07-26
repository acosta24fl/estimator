"""The validation suite must be able to reject as well as accept.

A test that only checks the report renders is worthless. These check that the
permutation test actually discriminates: a genuinely predictive signal should
pass it, and a random one should fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mnq.backtest.validate import (
    format_report, multiple_comparison_context, permutation_test,
    quarterly_breakdown, threshold_sensitivity, validate,
)
from mnq.config import Config
from mnq.data.synthetic import generate_frames
from mnq.models.train import prepare


@pytest.fixture(scope="module")
def data():
    cfg = Config()
    return prepare(generate_frames(n_minutes=60 * 24 * 40, seed=8), cfg), cfg


def _random_predictions(index, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "long_p_meta": rng.uniform(0.3, 0.8, len(index)),
            "short_p_meta": rng.uniform(0.3, 0.8, len(index)),
        },
        index=index,
    )


def _oracle_predictions(matrix, seed=0, strength=0.9) -> pd.DataFrame:
    """Predictions that genuinely know the future, as a positive control.

    If the permutation test cannot detect an oracle, it cannot detect anything,
    and a passing grade from it would be meaningless.
    """
    rng = np.random.default_rng(seed)
    fwd = matrix["close"].shift(-12) / matrix["open"].shift(-1) - 1.0
    noise = rng.uniform(0, 1, len(matrix))
    up = (fwd > 0).astype(float).to_numpy()
    p_long = strength * up + (1 - strength) * noise
    return pd.DataFrame(
        {
            "long_p_meta": np.clip(p_long * 0.5 + 0.35, 0, 1),
            "short_p_meta": np.clip((1 - p_long) * 0.5 + 0.35, 0, 1),
        },
        index=matrix.index,
    )


def test_permutation_test_rejects_random_predictions(data):
    """A coin flip must not pass."""
    d, cfg = data
    cfg = Config()
    cfg.trade.min_probability = 0.55
    preds = _random_predictions(d.matrix.index, seed=3)
    out = permutation_test(d.matrix, preds, cfg, n_permutations=40, seed=1)
    assert out["p_value"] > 0.05, (
        f"random predictions passed with p={out['p_value']}; the test cannot discriminate"
    )


def test_permutation_test_accepts_a_genuine_signal(data):
    """Positive control: an oracle must pass, or the test is useless."""
    d, cfg = data
    cfg = Config()
    cfg.trade.min_probability = 0.60
    preds = _oracle_predictions(d.matrix, seed=4)
    out = permutation_test(d.matrix, preds, cfg, n_permutations=40, seed=2)
    assert out["actual_net_usd"] > out["null_mean"]
    assert out["p_value"] <= 0.10, (
        f"an oracle only reached p={out['p_value']}; the test is too insensitive"
    )


def test_permutation_reports_the_null_distribution(data):
    d, _ = data
    cfg = Config()
    cfg.trade.min_probability = 0.55
    out = permutation_test(
        d.matrix, _random_predictions(d.matrix.index), cfg, n_permutations=20
    )
    for key in ("null_mean", "null_std", "null_p95", "null_max", "p_value",
                "null_pct_profitable"):
        assert key in out
    assert 0.0 <= out["p_value"] <= 1.0
    assert 0.0 <= out["null_pct_profitable"] <= 1.0


def test_threshold_curve_covers_the_grid(data):
    d, _ = data
    cfg = Config()
    curve = threshold_sensitivity(
        d.matrix, _random_predictions(d.matrix.index), cfg, thresholds=(0.55, 0.60, 0.65)
    )
    assert len(curve) >= 2
    # Raising the bar must not increase the trade count.
    counts = curve.sort_values("min_probability")["n_trades"].to_numpy()
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))


def test_quarterly_breakdown_splits_by_period(data):
    from mnq.backtest.engine import run_backtest

    d, _ = data
    cfg = Config()
    cfg.trade.min_probability = 0.52
    res = run_backtest(d.matrix, _random_predictions(d.matrix.index), cfg)
    q = quarterly_breakdown(res.frame)
    if not res.frame.empty:
        assert not q.empty
        assert q["n_trades"].sum() == len(res.frame)


def test_quarterly_breakdown_handles_no_trades():
    assert quarterly_breakdown(pd.DataFrame()).empty


def test_multiple_comparison_accounting():
    sweep = pd.DataFrame(
        {
            "n_trades": [50, 50, 60, 70],
            "net_usd": [100.0, 100.0, 200.0, -50.0],   # first two are duplicates
            "is_net_usd": [10.0, 10.0, 50.0, -10.0],
            "oos_net_usd": [90.0, 90.0, 150.0, -40.0],
        }
    )
    mc = multiple_comparison_context(sweep)
    assert mc["n_configs"] == 4
    assert mc["effective_configs"] == 3      # duplicates collapsed
    assert mc["n_passing"] == 3
    assert mc["expected_by_chance"] == pytest.approx(0.75)


def test_multiple_comparison_handles_empty():
    mc = multiple_comparison_context(pd.DataFrame())
    assert mc["n_configs"] == 0


def test_verdict_rejects_a_failing_permutation(data):
    """A high p-value must sink the verdict regardless of other checks."""
    d, _ = data
    cfg = Config()
    cfg.trade.min_probability = 0.55
    report = validate(
        d.matrix, _random_predictions(d.matrix.index, seed=9), cfg,
        sweep=None, n_permutations=25,
    )
    verdict, concerns = report.verdict()
    assert verdict in ("LIKELY NOISE", "INCONCLUSIVE")
    assert any("permutation" in c for c in concerns)

    text = format_report(report, cfg)
    assert "PERMUTATION TEST" in text
    assert "VERDICT" in text


def test_no_spurious_multiple_comparison_concern(data):
    """Without a sweep, that check must stay silent rather than invent one."""
    d, _ = data
    cfg = Config()
    cfg.trade.min_probability = 0.55
    report = validate(
        d.matrix, _random_predictions(d.matrix.index), cfg, sweep=None, n_permutations=15
    )
    _, concerns = report.verdict()
    assert not any("configurations were scored" in c for c in concerns)
