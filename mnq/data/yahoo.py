"""Yahoo Finance OHLCV loader for MNQ futures.

Yahoo limits intraday history by interval (roughly 60 days of sub-hourly bars,
730 days of hourly), so the 4h timeframe is built by resampling hourly bars
rather than requested directly. Downloads are cached to CSV: the 60-day 5m
window rolls forward every day, and merging each pull into the cache is what
lets the usable history grow past Yahoo's cap over time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import Config, DataConfig
from ..timeutil import to_utc_ns

log = logging.getLogger(__name__)

OHLCV = ["open", "high", "low", "close", "volume"]


def _normalise(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Flatten yfinance output into a tz-aware UTC OHLCV frame."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=OHLCV)

    # yfinance returns a column MultiIndex when auto-grouping by ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(-1, axis=1)

    df = df.rename(columns=str.lower)
    keep = [c for c in OHLCV if c in df.columns]
    missing = set(OHLCV) - set(keep)
    if missing:
        raise ValueError(f"Yahoo response missing columns: {sorted(missing)}")
    df = df[keep].copy()

    idx = to_utc_ns(df.index)
    df.index = idx.tz_convert(tz) if tz != "UTC" else idx
    df.index.name = "timestamp"

    df = df[~df.index.duplicated(keep="last")].sort_index()
    # Yahoo pads gaps with all-NaN or zero-volume placeholder rows.
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    return df.astype(float)


def download(
    symbol: str, interval: str, lookback: str, tz: str = "UTC"
) -> pd.DataFrame:
    """Fetch one interval from Yahoo. Import is local so the rest of the system
    (backtest, training, replay) runs without yfinance installed."""
    import yfinance as yf

    log.info("downloading %s %s over %s", symbol, interval, lookback)
    raw = yf.download(
        symbol,
        interval=interval,
        period=lookback,
        progress=False,
        auto_adjust=False,
        prepost=True,
    )
    df = _normalise(raw, tz)
    if df.empty:
        raise RuntimeError(
            f"Yahoo returned no rows for {symbol} {interval}/{lookback}. "
            "Check the symbol, the interval/lookback limits, and network egress."
        )
    log.info("  -> %d bars, %s .. %s", len(df), df.index[0], df.index[-1])
    return df


def cache_path(cfg: DataConfig, interval: str, root: Path) -> Path:
    safe = cfg.symbol.replace("=", "").replace("/", "")
    return root / f"{safe}_{interval}.csv"


def merge_cache(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Union two OHLCV frames, preferring the fresher row on overlap.

    Yahoo revises the most recent bars (volume especially), so new data wins.
    """
    if existing.empty:
        return fresh
    if fresh.empty:
        return existing
    combined = pd.concat([existing, fresh])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


def load_cached(path: Path, tz: str = "UTC") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=OHLCV)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    idx = to_utc_ns(df.index)
    df.index = idx.tz_convert(tz) if tz != "UTC" else idx
    df.index.name = "timestamp"
    return df[OHLCV].astype(float).sort_index()


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate bars to a coarser interval.

    ``label='left'`` and ``closed='left'`` keep the stamp at the bar's opening
    time, matching how the 5m and 15m series are stamped. Getting this wrong
    would shift a completed 4h bar backwards onto 5m bars that preceded it and
    leak future information into the feature matrix.
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = df.resample(rule, label="left", closed="left").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])


def drop_maintenance(df: pd.DataFrame) -> pd.DataFrame:
    """Remove CME's daily 17:00-18:00 ET settlement break.

    Those bars carry near-zero volume and erratic prints; leaving them in
    pollutes the volatility features and invents gaps the strategy cannot trade.
    """
    if df.empty:
        return df
    et = df.index.tz_convert("America/New_York")
    closed = (et.hour == 17)
    weekend = (et.dayofweek == 5) | ((et.dayofweek == 6) & (et.hour < 18))
    return df[~(closed | weekend)]


def fetch_all(cfg: Config, use_cache: bool = True, refresh: bool = True) -> dict[str, pd.DataFrame]:
    """Return the three timeframes keyed ``5m`` / ``15m`` / ``4h``.

    Each is cached and merged so repeated runs accumulate history beyond the
    window Yahoo will serve in a single request.
    """
    d = cfg.data
    root = cfg.path(d.cache_dir)
    root.mkdir(parents=True, exist_ok=True)

    if d.profile == "wide":
        # Hourly base plus daily context. The 4h slot is resampled from the
        # hourly pull rather than requested, so this costs two requests.
        requests = [
            ("1h", d.wide_base_interval, d.wide_base_lookback),
            ("1d", d.wide_high_interval, d.wide_high_lookback),
        ]
    else:
        requests = [
            ("5m", d.base_interval, d.base_lookback),
            ("15m", d.mid_interval, d.mid_lookback),
            ("1h", d.high_interval, d.high_lookback),
        ]

    frames: dict[str, pd.DataFrame] = {}
    for key, interval, lookback in requests:
        path = cache_path(d, interval, root)
        cached = load_cached(path, d.tz) if use_cache else pd.DataFrame(columns=OHLCV)
        if refresh:
            try:
                fresh = download(d.symbol, interval, lookback, d.tz)
                merged = merge_cache(cached, fresh)
                merged.to_csv(path)
            except Exception as exc:  # noqa: BLE001 - fall back to cache
                if cached.empty:
                    raise
                log.warning("download failed for %s (%s); using cache", interval, exc)
                merged = cached
        else:
            if cached.empty:
                raise RuntimeError(f"no cached data at {path}; run with refresh")
            merged = cached
        frames[key] = merged

    if d.profile == "wide":
        # Keep 1h as the base and derive 4h from it.
        frames["4h"] = resample_ohlcv(frames["1h"], d.wide_mid_resample)
    else:
        frames["4h"] = resample_ohlcv(frames.pop("1h"), d.high_resample)

    if d.drop_maintenance_break:
        # Daily bars have no intraday session to trim, and filtering them by
        # hour-of-day would silently delete the entire series.
        frames = {
            k: (v if k == "1d" else drop_maintenance(v)) for k, v in frames.items()
        }

    return frames
