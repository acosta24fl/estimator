"""Central configuration.

Every setting is overridable with an ``MNQ_``-prefixed environment variable, so
new deployments never need to edit this file.  Add a new field here + one line
in :func:`load_settings` and it is immediately available everywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent


def _env(name: str, default: str) -> str:
    return os.getenv(f"MNQ_{name}", default)


@dataclass(frozen=True)
class Settings:
    # --- instrument -------------------------------------------------------
    symbol: str = "MNQ=F"
    display_name: str = "Micro E-mini Nasdaq-100"

    # --- data feed --------------------------------------------------------
    feed: str = "yahoo"
    poll_seconds: float = 10.0
    intraday_range: str = "5d"
    daily_range: str = "1y"
    daily_refresh_seconds: float = 300.0
    request_timeout: float = 15.0

    # --- session ----------------------------------------------------------
    # CME Globex: the trade date rolls at 18:00 America/New_York.
    session_tz: str = "America/New_York"
    session_open_hour: int = 18

    # --- forecast ---------------------------------------------------------
    # Scales the projected 5-minute move. Below 1.0 damps it toward
    # "no change"; 0.0 disables the drift entirely. Measure first with
    # `python -m app.backtest`.
    forecast_strength: float = 1.0
    # Ridge penalty on the fitted coefficients. Larger = stronger shrinkage
    # toward "no move", which is the safe direction when signal is weak.
    forecast_ridge_lambda: float = 10.0
    # Fitted (features, outcome) pairs required before projecting at all.
    forecast_min_samples: int = 200
    # Horizon for the dashboard's headline bullish/bearish call, in minutes.
    # Uses the registered timeframe of the same length.
    signal_horizon_minutes: int = 10
    # Projected move must reach this fraction of a typical move to be a call
    # at all; below it the page stays neutral rather than colouring on noise.
    signal_min_ratio: float = 0.10

    # --- paper trading ----------------------------------------------------
    # Simulated only; nothing is ever sent to a broker.
    paper_trading: bool = True
    paper_cost_points: float = 0.75

    # --- storage ----------------------------------------------------------
    data_dir: Path = PROJECT_DIR / "data"
    max_1m_bars: int = 20_000

    # --- server -----------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8765
    web_dir: Path = PROJECT_DIR / "web"

    @property
    def bars_dir(self) -> Path:
        return self.data_dir / "bars"


def load_settings() -> Settings:
    return Settings(
        symbol=_env("SYMBOL", "MNQ=F"),
        display_name=_env("DISPLAY_NAME", "Micro E-mini Nasdaq-100"),
        feed=_env("FEED", "yahoo"),
        poll_seconds=float(_env("POLL_SECONDS", "10")),
        intraday_range=_env("INTRADAY_RANGE", "5d"),
        daily_range=_env("DAILY_RANGE", "1y"),
        daily_refresh_seconds=float(_env("DAILY_REFRESH_SECONDS", "300")),
        request_timeout=float(_env("REQUEST_TIMEOUT", "15")),
        session_tz=_env("SESSION_TZ", "America/New_York"),
        session_open_hour=int(_env("SESSION_OPEN_HOUR", "18")),
        forecast_strength=float(_env("FORECAST_STRENGTH", "1.0")),
        forecast_ridge_lambda=float(_env("FORECAST_RIDGE_LAMBDA", "10")),
        forecast_min_samples=int(_env("FORECAST_MIN_SAMPLES", "200")),
        signal_horizon_minutes=int(_env("SIGNAL_HORIZON_MINUTES", "10")),
        signal_min_ratio=float(_env("SIGNAL_MIN_RATIO", "0.10")),
        paper_trading=_env("PAPER_TRADING", "1") not in ("0", "false", "False", ""),
        paper_cost_points=float(_env("PAPER_COST_POINTS", "0.75")),
        data_dir=Path(_env("DATA_DIR", str(PROJECT_DIR / "data"))),
        max_1m_bars=int(_env("MAX_1M_BARS", "20000")),
        host=_env("HOST", "127.0.0.1"),
        port=int(_env("PORT", "8765")),
        web_dir=Path(_env("WEB_DIR", str(PROJECT_DIR / "web"))),
    )
