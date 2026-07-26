"""Self-backtesting: parameter search and unsupervised pattern discovery.

Two independent capabilities:

``sweep_decision_params`` re-runs the backtest across combinations of the gates
and management rules. It reuses one set of walk-forward predictions, so a grid
of hundreds of configurations costs seconds rather than hours.

``discover_patterns`` ignores the models entirely and mines the feature matrix
for conditions under which one direction wins more often than its base rate -
single features and pairs of them. This is the "find new patterns in combined
calculations" job, and it is where the next generation of hand-written rules
comes from.

A warning that applies to both: this is multiple hypothesis testing on a finite
sample. Sweeping 500 configurations will always surface a best one, and mining
thousands of feature buckets will always surface a striking bucket, even in pure
noise. Both functions therefore report out-of-sample stability and a
significance estimate rather than raw win rates, and the module refuses to
present a winner without them. Treat every output as a hypothesis to test on
fresh data, never as a result.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import replace
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ..config import Config
from ..labeling import LONG, SHORT
from .engine import run_backtest

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

DEFAULT_GRID: dict[str, Sequence[Any]] = {
    "min_probability": [0.52, 0.55, 0.58, 0.62, 0.66],
    "min_edge_points": [20.0, 30.0, 45.0],
    "early_exit_prob": [0.0, 0.30, 0.40],
    "breakeven_at_r": [0.0, 1.0],
    "trail_atr_mult": [0.0, 1.5, 2.5],
    "cooldown_bars": [6],
}


def sweep_decision_params(
    matrix: pd.DataFrame,
    predictions: pd.DataFrame,
    cfg: Config,
    grid: dict[str, Sequence[Any]] | None = None,
    min_trades: int = 20,
) -> pd.DataFrame:
    """Grid-search the trade-decision parameters.

    Only ``TradeConfig`` fields are varied. Anything that changes the *labels*
    (barrier geometry) would require retraining, which
    :func:`sweep_barrier_geometry` handles separately.

    Results are split in half by time: a configuration that only works in the
    first half is curve-fitted, and the ``oos_net_usd`` column is what exposes
    that.
    """
    grid = grid or DEFAULT_GRID
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    log.info("sweeping %d configurations", len(combos))

    # Fixed split point so every configuration is judged on the same two spans.
    split_at = predictions.index[len(predictions) // 2]

    rows = []
    for values in combos:
        params = dict(zip(keys, values))
        trial = replace(cfg.trade, **params)
        trial_cfg = replace(cfg, trade=trial)

        try:
            full = run_backtest(matrix, predictions, trial_cfg)
        except Exception as exc:  # noqa: BLE001 - a bad combo must not kill the sweep
            log.debug("combo %s failed: %s", params, exc)
            continue

        m = full.metrics
        if m["n_trades"] < min_trades:
            continue

        is_preds = predictions.loc[predictions.index < split_at]
        oos_preds = predictions.loc[predictions.index >= split_at]
        is_m = _safe_metrics(matrix, is_preds, trial_cfg)
        oos_m = _safe_metrics(matrix, oos_preds, trial_cfg)

        rows.append(
            {
                **params,
                "n_trades": m["n_trades"],
                "win_rate": m["win_rate"],
                "net_usd": m["net_usd"],
                "total_points": m["total_points"],
                "profit_factor": m["profit_factor"],
                "expectancy_r": m["expectancy_r"],
                "max_dd_usd": m["max_drawdown_usd"],
                "sharpe": m["sharpe"],
                "trades_per_day": m["trades_per_day"],
                "is_net_usd": is_m.get("net_usd", 0.0),
                "oos_net_usd": oos_m.get("net_usd", 0.0),
                "oos_n_trades": oos_m.get("n_trades", 0),
                "pct_hit_50pt": m["pct_hit_50pt"],
            }
        )

    if not rows:
        log.warning("no configuration produced at least %d trades", min_trades)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Rank on consistency, not on peak profit: a configuration that made all its
    # money in one half of the sample has not demonstrated anything.
    df["consistency"] = np.sign(df["is_net_usd"]) * np.sign(df["oos_net_usd"])
    df["score"] = np.where(
        df["consistency"] > 0,
        df[["is_net_usd", "oos_net_usd"]].min(axis=1),
        df[["is_net_usd", "oos_net_usd"]].min(axis=1) - df["net_usd"].abs() * 0.25,
    )
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def _safe_metrics(
    matrix: pd.DataFrame, preds: pd.DataFrame, cfg: Config
) -> dict[str, Any]:
    if preds.empty:
        return {}
    try:
        return run_backtest(matrix, preds, cfg).metrics
    except Exception:  # noqa: BLE001
        return {}


def sweep_barrier_geometry(
    frames: dict[str, pd.DataFrame],
    cfg: Config,
    tp_mults: Iterable[float] = (1.5, 2.0, 2.5, 3.0),
    sl_mults: Iterable[float] = (0.75, 1.0, 1.25),
    horizons: Iterable[int] = (18, 24, 36),
) -> pd.DataFrame:
    """Search TP/SL/horizon, retraining at every point.

    Expensive - each combination is a full relabel plus a full walk-forward
    retrain. Run it occasionally to choose the barrier geometry, then hold that
    fixed and use the cheap decision sweep day to day.
    """
    from ..models.train import prepare, walk_forward_evaluate

    rows = []
    combos = list(itertools.product(tp_mults, sl_mults, horizons))
    log.info("barrier sweep: %d combinations (retrains each)", len(combos))

    for tp, sl, hz in combos:
        if tp <= sl:
            continue  # a target inside the stop is not a strategy
        labels = replace(cfg.labels, tp_atr_mult=tp, sl_atr_mult=sl, horizon_bars=hz)
        trial_cfg = replace(cfg, labels=labels)
        try:
            data = prepare(frames, trial_cfg)
            preds, wf_metrics = walk_forward_evaluate(data, trial_cfg)
            result = run_backtest(data.matrix, preds, trial_cfg)
        except Exception as exc:  # noqa: BLE001
            log.warning("tp=%.2f sl=%.2f hz=%d failed: %s", tp, sl, hz, exc)
            continue

        m = result.metrics
        rows.append(
            {
                "tp_atr_mult": tp,
                "sl_atr_mult": sl,
                "horizon_bars": hz,
                "long_auc": wf_metrics.get("long_oos_auc"),
                "short_auc": wf_metrics.get("short_oos_auc"),
                "n_trades": m["n_trades"],
                "win_rate": m["win_rate"],
                "net_usd": m["net_usd"],
                "profit_factor": m["profit_factor"],
                "expectancy_r": m["expectancy_r"],
                "max_dd_usd": m["max_drawdown_usd"],
                "avg_win_points": m["avg_win_points"],
            }
        )
        log.info(
            "tp=%.2f sl=%.2f hz=%d -> %d trades, $%.0f, PF %.2f",
            tp, sl, hz, m["n_trades"], m["net_usd"], m["profit_factor"],
        )

    return pd.DataFrame(rows).sort_values("net_usd", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pattern discovery
# ---------------------------------------------------------------------------


def _wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the 95% CI on a win rate.

    Used instead of the raw rate so that 4 wins from 5 trades does not outrank
    280 wins from 500. Ranking on the lower bound automatically penalises small
    samples, which is most of what makes naive pattern mining useless.
    """
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1.0 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    return float((centre - margin) / denom)


def discover_patterns(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    direction: int,
    n_buckets: int = 5,
    min_samples: int = 150,
    top_features: int = 40,
    include_pairs: bool = True,
    max_pairs: int = 400,
) -> pd.DataFrame:
    """Mine feature buckets whose win rate beats the base rate.

    Splits each feature into quantile buckets and measures the label rate
    inside each. Then does the same for pairs of the most promising features,
    which is where genuinely non-obvious combinations show up - a condition that
    is unremarkable alone can be strong in conjunction.

    Ranked by the Wilson lower bound on the win rate, so the ranking already
    accounts for sample size. The ``lift`` column is the honest headline:
    win rate minus base rate.
    """
    y = labels["label"]
    mask = y.notna()
    X = features[mask]
    y = y[mask].astype(int)
    if len(y) < min_samples * 2:
        log.warning("only %d labelled rows; pattern discovery needs more", len(y))
        return pd.DataFrame()

    base_rate = float(y.mean())
    side = "LONG" if direction == LONG else "SHORT"
    rows: list[dict[str, Any]] = []

    # Rank features by univariate separation to choose pair candidates.
    separation: list[tuple[str, float]] = []

    for col in X.columns:
        s = X[col]
        if s.notna().sum() < min_samples * 2 or s.nunique() < n_buckets:
            continue
        try:
            buckets = pd.qcut(s, n_buckets, labels=False, duplicates="drop")
        except (ValueError, IndexError):
            continue

        edges = s.quantile(np.linspace(0, 1, n_buckets + 1)).to_numpy()
        best_gap = 0.0
        for b in sorted(pd.unique(buckets.dropna())):
            sel = buckets == b
            n = int(sel.sum())
            if n < min_samples:
                continue
            wins = int(y[sel].sum())
            rate = wins / n
            lb = _wilson_lower_bound(wins, n)
            best_gap = max(best_gap, abs(rate - base_rate))
            lo = edges[int(b)] if int(b) < len(edges) else np.nan
            hi = edges[int(b) + 1] if int(b) + 1 < len(edges) else np.nan
            rows.append(
                {
                    "direction": side,
                    "kind": "single",
                    "condition": f"{col} in [{lo:.4g}, {hi:.4g}]",
                    "features": col,
                    "n": n,
                    "win_rate": rate,
                    "base_rate": base_rate,
                    "lift": rate - base_rate,
                    "wilson_lb": lb,
                    "edge_vs_base": lb - base_rate,
                }
            )
        separation.append((col, best_gap))

    # ---- pairwise combinations -------------------------------------------
    if include_pairs and separation:
        separation.sort(key=lambda kv: kv[1], reverse=True)
        candidates = [c for c, _ in separation[:top_features]]
        tried = 0
        # Terciles keep pair buckets populated; quintile x quintile would leave
        # 25 cells with too few samples each to say anything.
        for a, b in itertools.combinations(candidates, 2):
            if tried >= max_pairs:
                break
            tried += 1
            try:
                ba = pd.qcut(X[a], 3, labels=False, duplicates="drop")
                bb = pd.qcut(X[b], 3, labels=False, duplicates="drop")
            except (ValueError, IndexError):
                continue
            ea = X[a].quantile([0, 1 / 3, 2 / 3, 1.0]).to_numpy()
            eb = X[b].quantile([0, 1 / 3, 2 / 3, 1.0]).to_numpy()

            for ia in sorted(pd.unique(ba.dropna())):
                for ib in sorted(pd.unique(bb.dropna())):
                    sel = (ba == ia) & (bb == ib)
                    n = int(sel.sum())
                    if n < min_samples:
                        continue
                    wins = int(y[sel].sum())
                    rate = wins / n
                    lb = _wilson_lower_bound(wins, n)
                    rows.append(
                        {
                            "direction": side,
                            "kind": "pair",
                            "condition": (
                                f"{a} in [{ea[int(ia)]:.4g}, {ea[int(ia)+1]:.4g}] AND "
                                f"{b} in [{eb[int(ib)]:.4g}, {eb[int(ib)+1]:.4g}]"
                            ),
                            "features": f"{a}|{b}",
                            "n": n,
                            "win_rate": rate,
                            "base_rate": base_rate,
                            "lift": rate - base_rate,
                            "wilson_lb": lb,
                            "edge_vs_base": lb - base_rate,
                        }
                    )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values("edge_vs_base", ascending=False).reset_index(drop=True)


def validate_patterns(
    patterns: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    split_frac: float = 0.6,
    top_n: int = 25,
) -> pd.DataFrame:
    """Re-measure discovered patterns on data they were not mined from.

    Patterns are found on the first ``split_frac`` of history and re-checked on
    the rest. Anything whose lift does not survive the holdout was noise, and
    this is the column that decides which patterns are worth a rule.
    """
    if patterns.empty:
        return patterns

    y = labels["label"]
    mask = y.notna()
    X = features[mask]
    y = y[mask].astype(int)
    cut = int(len(X) * split_frac)
    X_oos, y_oos = X.iloc[cut:], y.iloc[cut:]
    if len(y_oos) < 50:
        log.warning("holdout too small to validate patterns")
        return patterns

    base_oos = float(y_oos.mean())
    out = []
    for _, row in patterns.head(top_n).iterrows():
        sel = _apply_condition(X_oos, row["condition"])
        n = int(sel.sum())
        rec = row.to_dict()
        if n < 30:
            rec.update({"oos_n": n, "oos_win_rate": np.nan, "oos_lift": np.nan, "holds": False})
        else:
            rate = float(y_oos[sel].mean())
            rec.update(
                {
                    "oos_n": n,
                    "oos_win_rate": rate,
                    "oos_base_rate": base_oos,
                    "oos_lift": rate - base_oos,
                    # Survives only if it keeps most of its in-sample edge.
                    "holds": bool((rate - base_oos) > 0.5 * row["lift"] and row["lift"] > 0),
                }
            )
        out.append(rec)
    return pd.DataFrame(out)


def _apply_condition(X: pd.DataFrame, condition: str) -> pd.Series:
    """Re-evaluate a textual condition emitted by :func:`discover_patterns`."""
    sel = pd.Series(True, index=X.index)
    for clause in condition.split(" AND "):
        name, _, rng = clause.partition(" in ")
        name = name.strip()
        if name not in X.columns:
            return pd.Series(False, index=X.index)
        lo, hi = (float(v) for v in rng.strip().strip("[]").split(","))
        col = X[name]
        sel &= col.between(lo, hi) & col.notna()
    return sel
