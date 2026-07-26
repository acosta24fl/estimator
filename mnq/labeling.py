"""Triple-barrier labelling.

For each bar and each direction the question is: starting from the *next* bar's
open, does price reach the profit target before the stop, within the horizon?
That is the exact question the live system has to answer, so it is the one the
models are trained on.

Three choices here matter more than they look:

* **Entry is the next bar's open, not this bar's close.** Features are known
  only once the bar closes, so the earliest realistic fill is the following
  open. Labelling from the close would build a small, consistent, and entirely
  fictional edge into every result downstream.

* **When a bar's range spans both barriers, the stop is assumed to hit first.**
  5m OHLC does not record the path within the bar. Assuming the win would
  inflate the hit rate exactly on the violent bars where it matters most.

* **Barriers are ATR multiples**, so a target is equally reachable in a quiet
  overnight session and a volatile cash open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LabelConfig

LONG = 1
SHORT = -1

# Outcome codes recorded alongside each label.
OUTCOME_TP = "tp"
OUTCOME_SL = "sl"
OUTCOME_TIMEOUT = "timeout"


def _barrier_labels(
    entry: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tp_dist: np.ndarray,
    sl_dist: np.ndarray,
    horizon: int,
    direction: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Walk each bar's forward window until a barrier is touched.

    Returns ``(label, exit_index, exit_price, outcome_code)``. ``outcome_code``
    is 0 = timeout, 1 = target, -1 = stop. Loops rather than vectorises because
    the exit is path-dependent; the cost is acceptable at 5m resolution.
    """
    n = len(entry)
    label = np.full(n, np.nan)
    exit_idx = np.full(n, -1, dtype=np.int64)
    exit_px = np.full(n, np.nan)
    outcome = np.zeros(n, dtype=np.int8)

    for i in range(n):
        e = entry[i]
        if not np.isfinite(e) or not np.isfinite(tp_dist[i]) or not np.isfinite(sl_dist[i]):
            continue
        # Entry fills at bar i+1's open, so scanning starts there.
        start = i + 1
        stop = min(start + horizon, n)
        if start >= n:
            continue

        if direction == LONG:
            tp_level = e + tp_dist[i]
            sl_level = e - sl_dist[i]
        else:
            tp_level = e - tp_dist[i]
            sl_level = e + sl_dist[i]

        hit = 0
        j_hit = -1
        for j in range(start, stop):
            if direction == LONG:
                touched_sl = low[j] <= sl_level
                touched_tp = high[j] >= tp_level
            else:
                touched_sl = high[j] >= sl_level
                touched_tp = low[j] <= tp_level

            if touched_sl:
                # Pessimistic tie-break: a bar touching both counts as a loss.
                hit = -1
                j_hit = j
                break
            if touched_tp:
                hit = 1
                j_hit = j
                break

        if hit == 1:
            label[i] = 1.0
            exit_idx[i] = j_hit
            exit_px[i] = tp_level
            outcome[i] = 1
        elif hit == -1:
            label[i] = 0.0
            exit_idx[i] = j_hit
            exit_px[i] = sl_level
            outcome[i] = -1
        else:
            # Timeout: a trade that never reached either barrier is not a win.
            last = stop - 1
            if last >= start:
                label[i] = 0.0
                exit_idx[i] = last
                exit_px[i] = close[last]
                outcome[i] = 0

    return label, exit_idx, exit_px, outcome


def build_labels(
    matrix: pd.DataFrame, cfg: LabelConfig, direction: int
) -> pd.DataFrame:
    """Label every bar for one direction.

    ``matrix`` must carry 5m OHLC plus the ``atr`` column that
    :func:`mnq.features.builder.build_feature_matrix` adds.
    """
    if direction not in (LONG, SHORT):
        raise ValueError("direction must be LONG (1) or SHORT (-1)")

    high = matrix["high"].to_numpy(float)
    low = matrix["low"].to_numpy(float)
    close = matrix["close"].to_numpy(float)
    open_ = matrix["open"].to_numpy(float)
    atr = matrix["atr"].to_numpy(float)

    # Entry price is the next bar's open.
    entry = np.roll(open_, -1)
    entry[-1] = np.nan

    tp_dist = atr * cfg.tp_atr_mult
    sl_dist = atr * cfg.sl_atr_mult

    # A target that is unreachably small cannot cover costs; one that is absurdly
    # large will simply time out. Both are excluded rather than mislabelled.
    tradeable = (tp_dist >= cfg.min_target_points) & (tp_dist <= cfg.max_target_points)
    tp_dist = np.where(tradeable, tp_dist, np.nan)
    sl_dist = np.where(tradeable, sl_dist, np.nan)

    label, exit_idx, exit_px, outcome = _barrier_labels(
        entry, high, low, close, tp_dist, sl_dist, cfg.horizon_bars, direction
    )

    codes = {1: OUTCOME_TP, -1: OUTCOME_SL, 0: OUTCOME_TIMEOUT}
    out = pd.DataFrame(
        {
            "label": label,
            "entry_price": entry,
            "tp_price": np.where(
                direction == LONG, entry + tp_dist, entry - tp_dist
            ),
            "sl_price": np.where(
                direction == LONG, entry - sl_dist, entry + sl_dist
            ),
            "tp_points": tp_dist,
            "sl_points": sl_dist,
            "exit_index": exit_idx,
            "exit_price": exit_px,
            "outcome": [codes[c] for c in outcome],
            "tradeable": tradeable,
        },
        index=matrix.index,
    )

    # Bars-held drives the purge/embargo width during walk-forward training.
    held = np.where(exit_idx >= 0, exit_idx - np.arange(len(matrix)), np.nan)
    out["bars_held"] = held
    # Realised move in points, signed by direction: the P&L the label implies.
    out["realised_points"] = direction * (out["exit_price"] - out["entry_price"])
    out.loc[~out["tradeable"], "label"] = np.nan
    return out


def forward_return(matrix: pd.DataFrame, bars: int) -> pd.Series:
    """Shared regression target: percent return from the next open to the close
    ``bars`` later. Direction-agnostic, so both ensembles consume the same one.
    """
    entry = matrix["open"].shift(-1)
    future = matrix["close"].shift(-bars)
    return 100.0 * (future / entry - 1.0)


def label_summary(labels: pd.DataFrame) -> dict[str, float]:
    """Class balance and barrier statistics, for sanity-checking a run."""
    valid = labels.dropna(subset=["label"])
    if valid.empty:
        return {"n": 0}
    counts = valid["outcome"].value_counts()
    return {
        "n": int(len(valid)),
        "win_rate": float(valid["label"].mean()),
        "pct_tp": float(counts.get(OUTCOME_TP, 0) / len(valid)),
        "pct_sl": float(counts.get(OUTCOME_SL, 0) / len(valid)),
        "pct_timeout": float(counts.get(OUTCOME_TIMEOUT, 0) / len(valid)),
        "mean_tp_points": float(valid["tp_points"].mean()),
        "mean_sl_points": float(valid["sl_points"].mean()),
        "mean_bars_held": float(valid["bars_held"].mean()),
    }
