"""Central configuration.

Every number the strategy depends on lives here so the sweep can vary it and the
live engine can load the exact configuration that was backtested. Secrets come
from the environment, never from the YAML file.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def artifact_root() -> Path:
    """Where cached data, trained models and reports live.

    Defaults to the code directory, but ``MNQ_HOME`` moves it outside. That
    matters in practice: the code folder is replaced wholesale on every update,
    and without this every update would discard the Yahoo cache and the trained
    models, forcing a full re-download and retrain. Pointing MNQ_HOME at a
    stable location makes the code disposable and the expensive artifacts
    permanent.
    """
    env = os.getenv("MNQ_HOME")
    return Path(env).expanduser().resolve() if env else REPO_ROOT


ARTIFACT_DIR = artifact_root() / "artifacts"

if TYPE_CHECKING:
    from .data.context import ContextConfig


def _default_context():
    """Imported lazily: mnq.data.context imports from this module."""
    from .data.context import ContextConfig

    return ContextConfig()

# MNQ contract specification (CME Micro E-mini Nasdaq-100).
POINT_VALUE_USD = 2.0
TICK_SIZE = 0.25


@dataclass
class DataConfig:
    symbol: str = "MNQ=F"
    # "intraday" = 5m/15m/4h over ~60 days (one regime, fast signals).
    # "wide"     = 1h/4h/1d over ~730 days (many regimes, slower signals).
    # Use "wide" to find out whether an approach generalises at all; a result
    # from 60 days of 5m bars describes one quarter, not the market.
    profile: str = "intraday"
    # Yahoo caps intraday history by interval: 60d for <1h bars, 730d for 1h.
    base_interval: str = "5m"
    base_lookback: str = "60d"
    mid_interval: str = "15m"
    mid_lookback: str = "60d"
    high_interval: str = "1h"  # resampled up to 4h locally
    high_lookback: str = "730d"
    high_resample: str = "4h"
    # Wide-profile intervals. Yahoo serves ~730 days of hourly and years of
    # daily, so this profile is not constrained the way intraday is.
    wide_base_interval: str = "1h"
    wide_base_lookback: str = "730d"
    wide_mid_resample: str = "4h"
    wide_high_interval: str = "1d"
    wide_high_lookback: str = "5y"

    cache_dir: str = "artifacts/data"
    # Vendor-purchased contract history and the continuous series built from it.
    # Separate from cache_dir because this is bought data: it is never
    # re-downloadable for free and must not be cleared with the Yahoo cache.
    archive_dir: str = "artifacts/archive"
    # Yahoo 5m bars are stamped in exchange time; everything is normalised to UTC.
    tz: str = "UTC"
    # Drop bars outside CME's Globex session (23h/day, closed 17:00-18:00 ET).
    drop_maintenance_break: bool = True

    def timeframes(self):
        """The Timeframe specs for the configured profile."""
        from .features.builder import TIMEFRAME_PROFILES

        if self.profile not in TIMEFRAME_PROFILES:
            raise ValueError(
                f"unknown data profile {self.profile!r}; "
                f"expected one of {sorted(TIMEFRAME_PROFILES)}"
            )
        return TIMEFRAME_PROFILES[self.profile]


@dataclass
class FeatureConfig:
    ema_spans: tuple[int, ...] = (8, 9, 21, 50, 200)
    slope_windows: tuple[int, ...] = (20, 50)
    ema_slope_window: int = 5
    momentum_window: int = 10
    hh_ll_window: int = 5
    atr_window: int = 14
    atr_accel_lag: int = 12
    bb_window: int = 20
    bb_std: float = 2.0
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_window: int = 14
    volume_sma: int = 20


@dataclass
class LabelConfig:
    """Triple-barrier definition.

    Barriers are ATR multiples so a target automatically widens in fast markets
    and tightens in quiet ones, which is what keeps the 20-250 point ambition
    realistic instead of a fixed target that is trivial in one regime and
    impossible in another.
    """

    horizon_bars: int = 24          # 24 x 5m = 2 hours to resolve
    tp_atr_mult: float = 2.0
    sl_atr_mult: float = 1.0
    min_target_points: float = 20.0  # reject setups that cannot pay for the risk
    max_target_points: float = 300.0
    fwd_return_bars: int = 12        # horizon for the shared regression target


@dataclass
class ModelConfig:
    xgb_params: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 20,
            "reg_lambda": 2.0,
            "gamma": 0.1,
            "tree_method": "hist",
            "n_jobs": -1,
        }
    )
    lgbm_params: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 400,
            "num_leaves": 15,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "min_child_samples": 40,
            "reg_lambda": 2.0,
            "n_jobs": -1,
            "verbose": -1,
        }
    )
    reg_params: dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 20,
            "reg_lambda": 2.0,
            "tree_method": "hist",
            "n_jobs": -1,
        }
    )
    # Inner folds used to build out-of-fold base predictions for the meta model.
    meta_inner_folds: int = 4
    # Walk-forward: fraction of history in the first training window.
    wf_initial_train: float = 0.5
    wf_folds: int = 5
    # Bars purged either side of the test window; must exceed the label horizon.
    embargo_bars: int = 36
    random_state: int = 7


@dataclass
class TradeConfig:
    """Execution, risk and the live monitor's early-exit rules."""

    # Fill assumptions. MNQ is 0.25 ticks wide; one tick of slippage per side is
    # a realistic retail assumption and stops the backtest flattering itself.
    slippage_ticks: float = 1.0
    commission_usd_per_side: float = 0.37
    contracts: int = 1

    # Signal gating, tuned by the sweep.
    min_probability: float = 0.58
    min_edge_points: float = 20.0
    max_concurrent_trades: int = 1
    cooldown_bars: int = 6  # 30 minutes between signals

    # Monitoring: how an open trade is managed bar by bar.
    enable_monitor: bool = True
    breakeven_at_r: float = 1.0        # move stop to entry after +1R
    trail_atr_mult: float = 1.5        # trail this far behind the extreme
    trail_start_r: float = 1.5         # only start trailing after +1.5R
    early_exit_prob: float = 0.35      # continuation prob below this -> close
    early_exit_min_bars: int = 3       # let the trade breathe first
    max_hold_bars: int = 48            # 4 hours, then flatten
    time_stop_r: float = 0.0           # exit at max_hold only if below this R


@dataclass
class TelegramConfig:
    bot_token: str = ""
    chat_id: str = ""
    enabled: bool = True
    # Sent on every signal so the phone message is self-contained.
    include_features: bool = True


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    # Shared secret TradingView must echo back; the webhook is public via ngrok.
    webhook_secret: str = ""
    signal_interval_minutes: int = 10
    bar_store_path: str = "artifacts/live_bars.csv"
    state_path: str = "artifacts/live_state.json"
    max_bars_retained: int = 200_000


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    context: "ContextConfig" = field(default_factory=lambda: _default_context())
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trade: TradeConfig = field(default_factory=TradeConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    model_dir: str = "artifacts/models"

    # ---------------------------------------------------------------- loading

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        cfg = cls()
        if path is not None:
            raw = yaml.safe_load(Path(path).read_text()) or {}
            cfg = cls._merge(cfg, raw)
        cfg._apply_env()
        return cfg

    @staticmethod
    def _merge(cfg: "Config", raw: dict[str, Any]) -> "Config":
        for section, values in raw.items():
            if not hasattr(cfg, section):
                raise ValueError(f"unknown config section: {section}")
            current = getattr(cfg, section)
            if dataclasses.is_dataclass(current) and isinstance(values, dict):
                known = {f.name for f in dataclasses.fields(current)}
                unknown = set(values) - known
                if unknown:
                    raise ValueError(f"unknown keys in [{section}]: {sorted(unknown)}")
                setattr(cfg, section, dataclasses.replace(current, **values))
            else:
                setattr(cfg, section, values)
        return cfg

    def _apply_env(self) -> None:
        """Secrets are environment-only so a config file can be committed."""
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat = os.getenv("TELEGRAM_CHAT_ID")
        secret = os.getenv("TV_WEBHOOK_SECRET")
        if token:
            self.telegram.bot_token = token
        if chat:
            self.telegram.chat_id = chat
        if secret:
            self.server.webhook_secret = secret

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        # Never let a token reach a log line or a saved run manifest.
        out["telegram"]["bot_token"] = "***" if self.telegram.bot_token else ""
        out["server"]["webhook_secret"] = "***" if self.server.webhook_secret else ""
        return out

    def path(self, attr: str) -> Path:
        """Resolve a configured relative path against the artifact root.

        Relative paths follow MNQ_HOME so caches and models survive a code
        update; absolute paths are respected as given.
        """
        value = attr if "/" in attr else getattr(self, attr)
        p = Path(value)
        return p if p.is_absolute() else artifact_root() / p
