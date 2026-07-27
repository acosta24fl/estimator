"""Cross-asset and regime features, prefixed ``ctx_``.

Where :mod:`mnq.features.builder` asks "what is MNQ doing?", this module asks
"what is everything else doing, and what kind of market is this?" - the two
questions that price-derived indicators on a single instrument cannot answer.

Four groups:

* **Relative strength** - MNQ's return minus another asset's. Nasdaq
  outperforming the Dow is a different market than both rising together.
* **Rolling correlation** - not the level but the *regime*. When Nasdaq
  decouples from bonds, the thing driving it has changed.
* **Risk appetite** - a composite of credit, small caps, gold and the yen.
* **Volatility regime** - VIX level and, more usefully, its percentile against
  its own recent history. "VIX at 18" means nothing; "VIX in its 85th
  percentile" means a great deal.

The same causality rule as everywhere else applies: context bars are merged on
their *close* time, so a context bar is invisible until it has completed.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..data.context import ContextConfig
from ..timeutil import align_merge_keys

log = logging.getLogger(__name__)

PREFIX = "ctx_"


def _returns(close: pd.Series, window: int) -> pd.Series:
    return 100.0 * (close / close.shift(window) - 1.0)


def _percentile_rank(s: pd.Series, window: int) -> pd.Series:
    """Where does the current value sit within its own trailing history?

    Expressed 0-1. This is what makes a level interpretable across years: an
    absolute VIX of 20 meant something different in 2017 than in 2022.
    """
    return s.rolling(window, min_periods=max(20, window // 10)).rank(pct=True)


def build_context_features(
    base_index: pd.DatetimeIndex,
    base_close: pd.Series,
    context: dict[str, pd.DataFrame],
    cfg: ContextConfig,
    base_minutes: int,
) -> pd.DataFrame:
    """Build the ``ctx_`` block aligned to the base timeframe.

    ``base_index`` is stamped at bar open; ``base_minutes`` is the base bar's
    duration, used to derive close times for the leak-free merge.
    """
    if not context:
        return pd.DataFrame(index=base_index)

    base_close_time = base_index + pd.Timedelta(minutes=base_minutes)

    # Columns are accumulated and concatenated once. Assigning ~120 columns one
    # at a time fragments the frame badly enough that pandas warns about it.
    blocks: list[pd.Series] = []

    # MNQ's own returns, needed as the reference leg for relative strength.
    own_returns = {w: _returns(base_close, w) for w in cfg.return_windows}

    for name, frame in context.items():
        if frame.empty or "close" not in frame:
            continue

        # Infer this symbol's bar duration to compute its close times. Falls
        # back to the configured interval if the index is irregular.
        deltas = frame.index.to_series().diff().dropna()
        span = deltas.median() if len(deltas) else pd.Timedelta(minutes=60)
        if pd.isna(span) or span <= pd.Timedelta(0):
            span = pd.Timedelta(minutes=60)

        feats = pd.DataFrame(index=frame.index)
        close = frame["close"]

        for w in cfg.return_windows:
            feats[f"{PREFIX}{name}_ret{w}"] = _returns(close, w)

        # Distance from a slow mean: cheap trend-position measure per asset.
        ma = close.rolling(50, min_periods=25).mean()
        sd = close.rolling(50, min_periods=25).std(ddof=0).replace(0.0, np.nan)
        feats[f"{PREFIX}{name}_zscore"] = (close - ma) / sd

        feats["_available_at"] = frame.index + span

        merged = _merge_on_close(feats, base_close_time, base_index)
        for col in merged.columns:
            blocks.append(merged[col])

        # Relative strength and correlation are computed after alignment so both
        # legs sit on the same clock.
        for w in cfg.return_windows:
            col = f"{PREFIX}{name}_ret{w}"
            if col in merged:
                blocks.append((own_returns[w] - merged[col]).rename(f"{PREFIX}{name}_rs{w}"))

        ret1_col = f"{PREFIX}{name}_ret1"
        if ret1_col in merged:
            own1 = own_returns[cfg.return_windows[0]]
            blocks.append(
                _safe_rolling_corr(own1, merged[ret1_col], cfg.correlation_window)
                .rename(f"{PREFIX}{name}_corr")
            )

    out = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=base_index)

    # The derived blocks are appended in one concat for the same reason as
    # above: appending them individually re-fragments the frame.
    derived = [s for s in (_volatility_regime(out, cfg) + _risk_appetite(out, cfg))]
    if derived:
        out = pd.concat([out, *derived], axis=1)

    # Final guard. Tree models handle NaN natively but XGBoost rejects inf
    # outright, and a single infinite cell aborts an entire training run.
    return out.replace([np.inf, -np.inf], np.nan)


def _safe_rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """Rolling correlation that returns NaN instead of inf on a flat window.

    RTH-only instruments (the ETFs in the basket) go stale overnight, so the
    merged series repeats a single value for hours. Variance over that window is
    zero, the correlation is 0/0, and pandas can emit +/-inf rather than NaN.
    That is undefined, not infinite, so it is reported as missing.
    """
    min_periods = max(2, window // 2)
    corr = a.rolling(window, min_periods=min_periods).corr(b)

    # Mask windows where either leg is effectively constant.
    std_a = a.rolling(window, min_periods=min_periods).std(ddof=0)
    std_b = b.rolling(window, min_periods=min_periods).std(ddof=0)
    flat = (std_a.abs() < 1e-12) | (std_b.abs() < 1e-12)
    corr = corr.where(~flat)

    # Correlation is bounded; anything outside [-1, 1] is numerical noise.
    return corr.where(corr.abs() <= 1.0).replace([np.inf, -np.inf], np.nan)


def _merge_on_close(
    feats: pd.DataFrame, base_close_time: pd.DatetimeIndex, base_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Attach context features to base bars using completion times only."""
    left = pd.DataFrame({"_asof": base_close_time}).sort_values("_asof")
    right = feats.sort_values("_available_at")
    # The price frame may be a fresh download while the context frame came
    # from cache - same instants, different resolution, and merge_asof
    # refuses to mix them.
    left, right = align_merge_keys(left, "_asof", right, "_available_at")
    merged = pd.merge_asof(
        left, right,
        left_on="_asof", right_on="_available_at",
        direction="backward", allow_exact_matches=True,
    )
    merged.index = base_index
    return merged.drop(columns=["_asof", "_available_at"])


def _volatility_regime(out: pd.DataFrame, cfg: ContextConfig) -> list[pd.Series]:
    """VIX level, percentile and direction.

    The percentile is the feature that actually generalises - it is comparable
    across years in a way the raw level is not.
    """
    level_col = f"{PREFIX}vix_zscore"
    ret_col = f"{PREFIX}vix_ret{cfg.return_windows[0]}"
    if level_col not in out:
        return []

    # Ranking is monotonic, so the z-score ranks identically to the raw level.
    series = [
        _percentile_rank(out[level_col], cfg.vix_percentile_window)
        .rename(f"{PREFIX}vix_percentile")
    ]
    if ret_col in out:
        # A volatility spike is a regime change; a drift is not.
        series.append((out[ret_col] > 5.0).astype(float).rename(f"{PREFIX}vix_spike"))
    return series


def _risk_appetite(out: pd.DataFrame, cfg: ContextConfig) -> list[pd.Series]:
    """Composite risk-on/risk-off score.

    Averages the signals that tend to move together when the market's appetite
    for risk shifts: credit and small caps rise risk-on; gold and the yen rise
    risk-off. Averaging is deliberate - any single leg is noisy, and the
    agreement between them is the informative part.
    """
    w = cfg.return_windows[min(1, len(cfg.return_windows) - 1)]
    risk_on = [f"{PREFIX}hyg_ret{w}", f"{PREFIX}rty_ret{w}", f"{PREFIX}sox_ret{w}"]
    risk_off = [f"{PREFIX}gc_ret{w}", f"{PREFIX}jpy_ret{w}", f"{PREFIX}zn_ret{w}"]

    on = [c for c in risk_on if c in out]
    off = [c for c in risk_off if c in out]
    if not on and not off:
        return []

    # Standardise each leg before combining; otherwise the most volatile asset
    # silently dominates the composite.
    def _standardise(cols: list[str]) -> pd.Series | None:
        if not cols:
            return None
        block = out[cols]
        z = (block - block.rolling(200, min_periods=50).mean()) / block.rolling(
            200, min_periods=50
        ).std(ddof=0).replace(0.0, np.nan)
        return z.mean(axis=1)

    on_z, off_z = _standardise(on), _standardise(off)
    if on_z is not None and off_z is not None:
        appetite = on_z - off_z
    elif on_z is not None:
        appetite = on_z
    else:
        appetite = -off_z
    series = [appetite.rename(f"{PREFIX}risk_appetite")]

    # Breadth: equal-weight versus cap-weight. When RSP lags badly, a handful of
    # mega-caps are carrying the index - a notoriously fragile configuration for
    # a Nasdaq-heavy contract.
    rsp, es = f"{PREFIX}rsp_ret{w}", f"{PREFIX}es_ret{w}"
    if rsp in out and es in out:
        series.append((out[rsp] - out[es]).rename(f"{PREFIX}breadth_divergence"))
    return series


def context_feature_columns(matrix: pd.DataFrame) -> list[str]:
    return sorted(c for c in matrix.columns if c.startswith(PREFIX))
