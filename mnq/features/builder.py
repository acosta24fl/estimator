"""Multi-timeframe feature construction.

Produces one row per 5-minute bar carrying that bar's own features plus the most
recently *completed* 15m and 4h features, prefixed ``tf5_`` / ``tf15_`` /
``tf4h_``.

Two rules govern this module, and both exist to stop the model reading the
future:

1. Every feature at bar ``t`` is computed from bars ``<= t`` only.
2. A higher-timeframe bar is not visible until it has closed. A 15m bar stamped
   10:00 spans 10:00-10:15, so it first becomes usable on the 5m bar closing at
   10:15 - never on the 5m bars at 10:00 or 10:05. This is enforced by merging
   on bar *close* times rather than open stamps.

Values are also kept scale-free (ATR- or percent-normalised) rather than raw
price levels. A model fed raw EMA values learns the price range of its training
window and collapses the moment the index trades outside it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..config import FeatureConfig

TIMEFRAME_MINUTES = {"5m": 5, "15m": 15, "4h": 240}
PREFIXES = {"5m": "tf5_", "15m": "tf15_", "4h": "tf4h_"}


def build_timeframe_features(
    df: pd.DataFrame, cfg: FeatureConfig, prefix: str
) -> pd.DataFrame:
    """Compute the full feature block for a single timeframe."""
    o, h, l, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    out = pd.DataFrame(index=df.index)

    # ---------------------------------------------------------------- volatility
    atr = ind.atr(h, l, c, cfg.atr_window)
    # Guard against a dead-flat window producing divide-by-zero features.
    safe_atr = atr.replace(0.0, np.nan)

    out[f"{prefix}atr"] = atr
    out[f"{prefix}atr_pct"] = 100.0 * atr / c
    out[f"{prefix}atr_accel"] = ind.atr_acceleration(atr, cfg.atr_accel_lag)
    out[f"{prefix}bb_width"] = ind.bollinger_width(c, cfg.bb_window, cfg.bb_std)

    # ---------------------------------------------------------------- trend
    emas: dict[int, pd.Series] = {}
    for span in cfg.ema_spans:
        e = ind.ema(c, span)
        emas[span] = e
        # Distance from each EMA, in ATR units: "3 ATRs above the 200" means the
        # same thing in any volatility regime.
        out[f"{prefix}dist_ema{span}"] = (c - e) / safe_atr
        out[f"{prefix}ema{span}_slope"] = ind.normalised_slope(e, cfg.ema_slope_window)

    if 9 in emas and 21 in emas:
        out[f"{prefix}ema_spread_9_21"] = (emas[9] - emas[21]) / safe_atr
    if 8 in emas and 21 in emas:
        out[f"{prefix}ema_spread_8_21"] = (emas[8] - emas[21]) / safe_atr
    if 50 in emas and 200 in emas:
        out[f"{prefix}ema_spread_50_200"] = (emas[50] - emas[200]) / safe_atr

    for w in cfg.slope_windows:
        out[f"{prefix}slope{w}"] = ind.normalised_slope(c, w)

    out[f"{prefix}momentum{cfg.momentum_window}"] = ind.momentum(c, cfg.momentum_window)
    out[f"{prefix}hh_count{cfg.hh_ll_window}"] = ind.higher_highs_count(h, cfg.hh_ll_window)
    out[f"{prefix}ll_count{cfg.hh_ll_window}"] = ind.lower_lows_count(l, cfg.hh_ll_window)
    out[f"{prefix}hh_ll_balance"] = (
        out[f"{prefix}hh_count{cfg.hh_ll_window}"] - out[f"{prefix}ll_count{cfg.hh_ll_window}"]
    )

    # ---------------------------------------------------------------- oscillators
    out[f"{prefix}rsi"] = ind.rsi(c, cfg.rsi_window)

    macd = ind.macd(c, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    # MACD is in price units, so it is divided by ATR to stay comparable across
    # regimes and across timeframes.
    out[f"{prefix}macd"] = macd["macd"] / safe_atr
    out[f"{prefix}macd_signal"] = macd["macd_signal"] / safe_atr
    out[f"{prefix}macd_hist"] = macd["macd_hist"] / safe_atr

    adx_df = ind.adx(h, l, c, cfg.adx_window)
    out[f"{prefix}adx"] = adx_df["adx"]
    out[f"{prefix}di_diff"] = adx_df["plus_di"] - adx_df["minus_di"]

    slope20_key = f"{prefix}slope{cfg.slope_windows[0]}"
    out[f"{prefix}trend_strength"] = out[slope20_key].abs() * adx_df["adx"] / 100.0
    # Signed variant: strength alone cannot tell the long model from the short one.
    out[f"{prefix}trend_strength_signed"] = out[slope20_key] * adx_df["adx"] / 100.0

    # ---------------------------------------------------------------- candle
    out[f"{prefix}body_ratio"] = ind.body_ratio(o, h, l, c)
    out[f"{prefix}wick_ratio"] = ind.wick_ratio(o, h, l, c)
    out[f"{prefix}wick_skew"] = ind.signed_wick_skew(o, h, l, c)
    out[f"{prefix}bar_delta"] = (c - o) / safe_atr
    out[f"{prefix}rel_volume"] = ind.relative_volume(v, cfg.volume_sma)

    return out


def _close_times(index: pd.DatetimeIndex, minutes: int) -> pd.DatetimeIndex:
    """Bars are stamped at their open; this returns when each one completes."""
    return index + pd.Timedelta(minutes=minutes)


def build_feature_matrix(
    frames: dict[str, pd.DataFrame], cfg: FeatureConfig
) -> pd.DataFrame:
    """Assemble the 5m-indexed feature matrix from all three timeframes.

    Returns a frame indexed by 5m bar open time containing the raw 5m OHLCV
    (needed for barrier evaluation and fills) alongside every prefixed feature.
    """
    if "5m" not in frames:
        raise ValueError("a '5m' frame is required as the base timeframe")

    base = frames["5m"].sort_index()
    matrix = base.copy()
    matrix["atr"] = ind.atr(
        base["high"], base["low"], base["close"], cfg.atr_window
    )

    base_feats = build_timeframe_features(base, cfg, PREFIXES["5m"])
    matrix = matrix.join(base_feats)

    # Higher timeframes are joined on completion time, never on open stamp.
    base_close = _close_times(matrix.index, TIMEFRAME_MINUTES["5m"])

    for tf in ("15m", "4h"):
        if tf not in frames:
            continue
        htf = frames[tf].sort_index()
        feats = build_timeframe_features(htf, cfg, PREFIXES[tf])
        feats = feats.copy()
        feats["_available_at"] = _close_times(feats.index, TIMEFRAME_MINUTES[tf])

        left = pd.DataFrame({"_asof": base_close}).sort_values("_asof")
        right = feats.sort_values("_available_at")

        merged = pd.merge_asof(
            left,
            right,
            left_on="_asof",
            right_on="_available_at",
            direction="backward",
            allow_exact_matches=True,
        )
        merged.index = matrix.index
        merged = merged.drop(columns=["_asof", "_available_at"])
        matrix = matrix.join(merged)

    return matrix


def feature_columns(matrix: pd.DataFrame) -> list[str]:
    """Model inputs: prefixed features only.

    Raw price and volume are deliberately excluded - they are non-stationary and
    would let a tree memorise price levels. The absolute ``*_atr`` columns are
    excluded too; their scale-free ``*_atr_pct`` twin carries the same
    information without the level dependence.
    """
    cols = [
        c
        for c in matrix.columns
        if c.startswith(("tf5_", "tf15_", "tf4h_")) and not c.endswith("_atr")
    ]
    return sorted(cols)


def add_session_features(matrix: pd.DataFrame) -> pd.DataFrame:
    """Time-of-day and session context.

    Nasdaq futures behave very differently at 03:00 ET than at the cash open,
    and the model has no way to infer that from price alone. Encoded cyclically
    so 23:59 and 00:01 are adjacent rather than maximally distant.
    """
    out = matrix.copy()
    et = out.index.tz_convert("America/New_York")
    minute = et.hour * 60 + et.minute
    out["tf5_tod_sin"] = np.sin(2 * np.pi * minute / 1440.0)
    out["tf5_tod_cos"] = np.cos(2 * np.pi * minute / 1440.0)
    out["tf5_dow"] = et.dayofweek.astype(float)
    out["tf5_is_rth"] = (
        ((et.hour > 9) | ((et.hour == 9) & (et.minute >= 30)))
        & (et.hour < 16)
        & (et.dayofweek < 5)
    ).astype(float)
    return out
