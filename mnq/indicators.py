"""Vectorised technical indicators.

Every function takes and returns pandas objects indexed by bar timestamp and is
strictly causal: the value at index ``i`` uses only bars ``<= i``. That property
is what makes the feature matrix safe to train on, so any change here must
preserve it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Trend
# --------------------------------------------------------------------------


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with the standard 2/(span+1) smoothing."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def slope(series: pd.Series, window: int) -> pd.Series:
    """Least-squares slope of ``series`` over a rolling window, per bar.

    Implemented closed-form rather than via ``rolling.apply(polyfit)`` because
    the latter is ~200x slower and this runs over every timeframe.
    For x = 0..n-1 the OLS slope is cov(x, y) / var(x), and both the mean and
    the variance of x are constants, so only the rolling mean of ``y`` and of
    ``x*y`` are needed.
    """
    n = int(window)
    if n < 2:
        raise ValueError("slope window must be >= 2")
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(values: np.ndarray) -> float:
        return float(((values - values.mean()) * (x - x_mean)).sum() / x_var)

    return series.rolling(n, min_periods=n).apply(_slope, raw=True)


def normalised_slope(series: pd.Series, window: int) -> pd.Series:
    """Slope expressed in percent of price per bar, so it is comparable across
    timeframes and across the years-long range of an index future."""
    raw = slope(series, window)
    return 100.0 * raw / series.replace(0.0, np.nan)


def momentum(series: pd.Series, window: int) -> pd.Series:
    """Percent change over ``window`` bars."""
    return 100.0 * (series / series.shift(window) - 1.0)


def higher_highs_count(high: pd.Series, window: int) -> pd.Series:
    """Number of strictly higher highs in the trailing ``window`` bars."""
    up = (high > high.shift(1)).astype(float)
    return up.rolling(window, min_periods=window).sum()


def lower_lows_count(low: pd.Series, window: int) -> pd.Series:
    """Number of strictly lower lows in the trailing ``window`` bars."""
    down = (low < low.shift(1)).astype(float)
    return down.rolling(window, min_periods=window).sum()


# --------------------------------------------------------------------------
# Volatility
# --------------------------------------------------------------------------


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average true range using Wilder's smoothing (alpha = 1/window)."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def atr_acceleration(atr_series: pd.Series, lag: int = 12) -> pd.Series:
    """Ratio of current ATR to ATR ``lag`` bars ago: >1 means expanding range."""
    past = atr_series.shift(lag)
    return atr_series / past.replace(0.0, np.nan)


def bollinger_width(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """Band width as a fraction of the midline: (upper - lower) / midline.

    With the default 2-sigma bands that is the 4-sigma span the spec asks for.
    """
    mid = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std(ddof=0)
    return (2.0 * n_std * sd) / mid.replace(0.0, np.nan)


# --------------------------------------------------------------------------
# Oscillators
# --------------------------------------------------------------------------


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # A window with no losses is a valid, maximally overbought reading.
    return out.where(avg_loss.ne(0.0) | avg_gain.isna(), 100.0)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line and histogram."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": macd_line - signal_line,
        }
    )


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.DataFrame:
    """Wilder's ADX with the +DI / -DI components."""
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    alpha = 1.0 / window
    atr_w = true_range(high, low, close).ewm(
        alpha=alpha, adjust=False, min_periods=window
    ).mean()
    safe_atr = atr_w.replace(0.0, np.nan)

    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=window).mean() / safe_atr
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=window).mean() / safe_atr

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx_line = dx.ewm(alpha=alpha, adjust=False, min_periods=window).mean()

    return pd.DataFrame({"adx": adx_line, "plus_di": plus_di, "minus_di": minus_di})


# --------------------------------------------------------------------------
# Candle structure
# --------------------------------------------------------------------------


def body_ratio(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """|close - open| / (high - low): 1.0 is a marubozu, 0.0 a doji."""
    rng = (high - low).replace(0.0, np.nan)
    return (close - open_).abs() / rng


def wick_ratio(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """Combined upper+lower wick as a fraction of the bar range."""
    rng = (high - low).replace(0.0, np.nan)
    body_top = pd.concat([open_, close], axis=1).max(axis=1)
    body_bottom = pd.concat([open_, close], axis=1).min(axis=1)
    return ((high - body_top) + (body_bottom - low)) / rng


def signed_wick_skew(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """(lower wick - upper wick) / range. Positive means rejection of lows."""
    rng = (high - low).replace(0.0, np.nan)
    body_top = pd.concat([open_, close], axis=1).max(axis=1)
    body_bottom = pd.concat([open_, close], axis=1).min(axis=1)
    return ((body_bottom - low) - (high - body_top)) / rng


def relative_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume divided by its own trailing SMA."""
    base = volume.rolling(window, min_periods=window).mean().replace(0.0, np.nan)
    return volume / base
