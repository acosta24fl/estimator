"""Synthetic MNQ-like bar generator.

This exists for two jobs and neither of them is forecasting: exercising the full
pipeline where Yahoo is unreachable (CI, sandboxes, offline dev), and acting as
a control. Results measured here say the plumbing works; they say nothing about
whether the strategy makes money. Only ``fetch`` + ``backtest`` on real bars
answers that.

The process is a regime-switching random walk with volatility clustering and
mild momentum persistence, so the feature matrix has realistic autocorrelation
structure instead of pure white noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_minute_bars(
    n_minutes: int = 60 * 24 * 90,
    start: str = "2025-01-01",
    start_price: float = 21000.0,
    seed: int = 11,
    tz: str = "UTC",
) -> pd.DataFrame:
    """Generate 1-minute OHLCV bars resembling an index future.

    Regimes (trend up / trend down / chop) persist for hours, per-minute
    volatility follows a slow-moving GARCH-like process, and returns carry a
    small AR(1) term. Volume rises with absolute return and with the US session.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n_minutes, freq="1min", tz=tz)

    # --- regime chain: 0 = chop, 1 = up trend, 2 = down trend --------------
    # Mean dwell time ~4 hours, so a trend spans many 5m bars and the model has
    # something persistent to learn rather than per-bar coin flips.
    switch_p = 1.0 / (60 * 4)
    regimes = np.empty(n_minutes, dtype=int)
    state = 0
    for i in range(n_minutes):
        if rng.random() < switch_p:
            state = int(rng.choice([0, 1, 2], p=[0.5, 0.25, 0.25]))
        regimes[i] = state

    drift_per_regime = np.array([0.0, 3.2e-6, -3.2e-6])
    drift = drift_per_regime[regimes]

    # --- volatility clustering ---------------------------------------------
    # Calibrated so 5m ATR lands around 15-25 points at a 21k index level, which
    # is roughly what MNQ actually prints. Getting this scale right matters:
    # the point-based signal gates (min_edge_points) are meaningless if the
    # synthetic series is quieter than the real contract.
    base_vol = 2.8e-4
    vol = np.empty(n_minutes)
    v = base_vol
    for i in range(n_minutes):
        shock = rng.normal(0.0, 0.09)
        v = base_vol * 0.045 + v * 0.955 * float(np.exp(shock))
        vol[i] = min(max(v, base_vol * 0.25), base_vol * 6.0)

    # Intraday seasonality: the US cash session is the volatile part of the day.
    minute_of_day = idx.hour.values * 60 + idx.minute.values
    us_open = 14 * 60 + 30  # 09:30 ET in UTC (winter)
    dist = np.minimum(
        np.abs(minute_of_day - us_open), 1440 - np.abs(minute_of_day - us_open)
    )
    session = 0.55 + 1.45 * np.exp(-((dist / 200.0) ** 2))
    vol = vol * session

    # --- returns with AR(1) momentum ---------------------------------------
    eps = rng.standard_normal(n_minutes) * vol
    rets = np.empty(n_minutes)
    prev = 0.0
    phi = 0.06
    for i in range(n_minutes):
        r = drift[i] + phi * prev + eps[i]
        rets[i] = r
        prev = r

    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.empty(n_minutes)
    open_[0] = start_price
    open_[1:] = close[:-1]

    # Wicks scale with the bar's own volatility.
    wick = np.abs(rng.normal(0.0, 1.0, n_minutes)) * vol * close * 0.9
    high = np.maximum(open_, close) + wick * rng.uniform(0.2, 1.0, n_minutes)
    low = np.minimum(open_, close) - wick * rng.uniform(0.2, 1.0, n_minutes)

    volume = (
        rng.gamma(shape=2.0, scale=90.0, size=n_minutes)
        * session
        * (1.0 + 40.0 * np.abs(rets))
    )

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.round(volume),
        },
        index=idx,
    )
    df.index.name = "timestamp"

    # Snap to the 0.25 tick grid so downstream point maths matches a real book.
    for col in ("open", "high", "low", "close"):
        df[col] = (df[col] / 0.25).round() * 0.25
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    return df


def generate_frames(
    n_minutes: int = 60 * 24 * 90,
    seed: int = 11,
    profile: str = "intraday",
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Synthetic equivalent of :func:`mnq.data.yahoo.fetch_all`."""
    from .yahoo import resample_ohlcv

    minutes = generate_minute_bars(n_minutes=n_minutes, seed=seed, **kwargs)
    if profile == "wide":
        return {
            "1h": resample_ohlcv(minutes, "1h"),
            "4h": resample_ohlcv(minutes, "4h"),
            "1d": resample_ohlcv(minutes, "1D"),
        }
    return {
        "5m": resample_ohlcv(minutes, "5min"),
        "15m": resample_ohlcv(minutes, "15min"),
        "4h": resample_ohlcv(minutes, "4h"),
    }


def generate_context(
    base_minutes: pd.DataFrame,
    names: tuple[str, ...] = ("es", "zn", "vix", "gc", "hyg", "rty", "sox", "rsp", "jpy"),
    interval: str = "1h",
    seed: int = 23,
    correlation: float = 0.6,
) -> dict[str, pd.DataFrame]:
    """Synthetic cross-asset basket correlated with the base series.

    Each asset is a blend of the base instrument's returns and its own noise, so
    the correlation and relative-strength features have realistic structure to
    chew on. As with the price generator, this validates the plumbing only - the
    correlations are invented.
    """
    from .yahoo import resample_ohlcv

    rng = np.random.default_rng(seed)
    base_ret = np.log(base_minutes["close"]).diff().fillna(0.0).to_numpy()
    n = len(base_ret)
    out: dict[str, pd.DataFrame] = {}

    for i, name in enumerate(names):
        # VIX and bonds move against equities; the rest move with them.
        sign = -1.0 if name in ("vix", "zn", "gc", "jpy") else 1.0
        # VIX is far more volatile than an index future.
        scale = 6.0 if name == "vix" else 1.0
        own = rng.normal(0.0, 1.2e-4 * scale, n)
        blended = sign * correlation * base_ret * scale + (1 - correlation) * own

        start = {"vix": 18.0, "zn": 110.0, "gc": 2400.0, "hyg": 79.0,
                 "jpy": 0.0068}.get(name, 5000.0 + i * 300)
        close = start * np.exp(np.cumsum(blended))
        wick = np.abs(rng.normal(0, 1, n)) * close * 8e-5 * scale

        df = pd.DataFrame(
            {
                "open": np.concatenate([[close[0]], close[:-1]]),
                "high": close + wick,
                "low": close - wick,
                "close": close,
                "volume": rng.gamma(2.0, 50.0, n),
            },
            index=base_minutes.index,
        )
        df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
        df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
        out[name] = resample_ohlcv(df, interval)

    return out
