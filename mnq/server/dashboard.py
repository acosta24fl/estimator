"""Local dashboard: chart, live signal, projection and system state.

Runs on localhost alongside the existing webhook server. Everything is served
from this process - no CDN, no external scripts, no API keys - so it works with
the network off and cannot leak what it is showing.

One deliberate constraint: every number on the page comes from the same code
paths the backtest uses. A dashboard that computed its own indicators would
eventually disagree with the backtest, and the disagreement would be silent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

#: Bars sent to the browser. Enough to see structure, small enough to stay snappy.
DEFAULT_BARS = 240


def _clean(value: Any) -> Any:
    """JSON cannot carry NaN or numpy scalars; convert or drop them."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if not np.isfinite(v) else v
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def bars_payload(matrix: pd.DataFrame, limit: int = DEFAULT_BARS) -> dict[str, Any]:
    """OHLC plus the overlays the chart draws."""
    if matrix is None or matrix.empty:
        return {"bars": [], "emas": {}, "count": 0}

    tail = matrix.tail(limit)
    bars = [
        {
            "t": ts.isoformat(),
            "o": _clean(r["open"]), "h": _clean(r["high"]),
            "l": _clean(r["low"]), "c": _clean(r["close"]),
            "v": _clean(r.get("volume", 0.0)),
        }
        for ts, r in tail.iterrows()
    ]

    # Overlays are read from the feature matrix rather than recomputed, so the
    # chart cannot drift away from what the model actually saw.
    emas: dict[str, list] = {}
    for span in (21, 50, 200):
        for prefix in ("tf1h_", "tf5_"):
            col = f"{prefix}dist_ema{span}"
            if col in tail and "atr" in tail:
                # dist_ema is (close - ema) / atr, so the EMA is recoverable.
                ema = tail["close"] - tail[col] * tail["atr"]
                emas[f"ema{span}"] = [_clean(v) for v in ema]
                break

    return {"bars": bars, "emas": emas, "count": int(len(tail))}


def indicator_payload(matrix: pd.DataFrame) -> dict[str, Any]:
    """The latest value of the indicators worth showing as numbers."""
    if matrix is None or matrix.empty:
        return {}
    row = matrix.iloc[-1]

    wanted = {
        "rsi": ("tf1h_rsi", "tf5_rsi"),
        "adx": ("tf1h_adx", "tf5_adx"),
        "atr": ("atr",),
        "atr_pct": ("tf1h_atr_pct", "tf5_atr_pct"),
        "macd_hist": ("tf1h_macd_hist", "tf5_macd_hist"),
        "slope20": ("tf1h_slope20", "tf5_slope20"),
        "trend_strength": ("tf1h_trend_strength_signed", "tf5_trend_strength_signed"),
        "rel_volume": ("tf1h_rel_volume", "tf5_rel_volume"),
        "vix_percentile": ("ctx_vix_percentile",),
        "risk_appetite": ("ctx_risk_appetite",),
        "breadth": ("ctx_breadth_divergence",),
    }
    out: dict[str, Any] = {}
    for label, candidates in wanted.items():
        for col in candidates:
            if col in row.index:
                out[label] = _clean(row[col])
                break
    return out


def trend_summary(matrix: pd.DataFrame, lookbacks=(6, 24, 120)) -> dict[str, Any]:
    """Realised trend over several windows, in points.

    This is descriptive, not predictive - what price has already done. It sits
    beside the projection so the two are never confused.
    """
    if matrix is None or len(matrix) < 2:
        return {}
    close = matrix["close"]
    out: dict[str, Any] = {}
    for n in lookbacks:
        if len(close) <= n:
            continue
        change = float(close.iloc[-1] - close.iloc[-1 - n])
        out[f"last_{n}"] = {
            "points": _clean(change),
            "pct": _clean(100.0 * change / close.iloc[-1 - n]),
            "direction": "up" if change > 0 else ("down" if change < 0 else "flat"),
        }
    return out


def build_snapshot(
    cfg: Config,
    matrix: pd.DataFrame | None,
    score: dict[str, Any] | None,
    projection,
    engine_status: dict[str, Any] | None = None,
    calibration=None,
    limit: int = DEFAULT_BARS,
    autopilot: dict[str, Any] | None = None,
    horizons: list[dict[str, Any]] | None = None,
    trend_read: dict[str, Any] | None = None,
    alpha: dict[str, Any] | None = None,
    path: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Everything the page needs, in one response."""
    payload: dict[str, Any] = {
        "symbol": cfg.data.symbol,
        "profile": cfg.data.profile,
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "chart": bars_payload(matrix, limit) if matrix is not None else {"bars": []},
        "indicators": indicator_payload(matrix) if matrix is not None else {},
        "trend": trend_summary(matrix) if matrix is not None else {},
        "projection": projection.to_dict() if projection is not None else None,
        "gates": {
            "min_probability": cfg.trade.min_probability,
            "min_edge_points": cfg.trade.min_edge_points,
            "tp_atr_mult": cfg.labels.tp_atr_mult,
            "sl_atr_mult": cfg.labels.sl_atr_mult,
        },
        "engine": engine_status or {},
        "calibration": calibration.to_records() if calibration is not None else [],
        "autopilot": autopilot,
        "horizons": horizons or [],
        "trend_read": trend_read,
        "alpha": alpha,
        "path": path or [],
    }

    if score:
        payload["score"] = {k: _clean(v) for k, v in score.items()
                            if not isinstance(v, dict)}
        last = matrix.iloc[-1] if matrix is not None and not matrix.empty else None
        atr = float(score.get("atr") or 0.0)
        close = float(score.get("close") or 0.0)
        if atr > 0 and close > 0:
            # The levels a signal would use, so the chart can draw them even
            # when no trade is open.
            payload["levels"] = {
                "close": _clean(close),
                "long_target": _clean(close + cfg.labels.tp_atr_mult * atr),
                "long_stop": _clean(close - cfg.labels.sl_atr_mult * atr),
                "short_target": _clean(close - cfg.labels.tp_atr_mult * atr),
                "short_stop": _clean(close + cfg.labels.sl_atr_mult * atr),
                "atr": _clean(atr),
            }
    return payload


def read_page() -> str:
    """The dashboard HTML, read from disk so it can be edited without a rebuild."""
    path = STATIC_DIR / "dashboard.html"
    if not path.exists():
        return (
            "<h1>dashboard.html is missing</h1>"
            f"<p>Expected it at <code>{path}</code>.</p>"
        )
    return path.read_text(encoding="utf-8")
