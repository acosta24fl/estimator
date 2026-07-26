"""Performance statistics for a list of closed trades.

Reported in points *and* dollars. Points measure whether the strategy reads the
market; dollars measure whether it survives costs. A system can be positive in
points and negative in dollars, and that distinction is the whole reason the
backtester charges slippage and commission.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..config import POINT_VALUE_USD, TradeConfig
from ..trade.manager import LONG, Trade


def trades_to_frame(trades: Sequence[Trade], tcfg: TradeConfig) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    rows = []
    for t in trades:
        rows.append(
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "side": "LONG" if t.direction == LONG else "SHORT",
                "entry": t.entry_price,
                "exit": t.exit_price,
                "stop": t.initial_stop,
                "target": t.target,
                "points": t.realised_points,
                "net_usd": t.net_usd(tcfg),
                "r_multiple": (t.realised_points / t.risk_points) if t.risk_points else np.nan,
                "bars_held": t.bars_held,
                "mfe_points": t.mfe_points,
                "mae_points": t.mae_points,
                "probability": t.probability,
                "exit_reason": t.exit_reason,
                "exit_probability": t.exit_probability,
            }
        )
    return pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float((equity - peak).min())


def compute_metrics(
    trades: Sequence[Trade], tcfg: TradeConfig, bars_per_day: float = 288.0
) -> dict[str, Any]:
    """Headline statistics. Returns zeros rather than raising on no trades, so a
    sweep can score an over-restrictive configuration without special-casing."""
    df = trades_to_frame(trades, tcfg)
    if df.empty:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "total_points": 0.0,
            "net_usd": 0.0,
            "profit_factor": 0.0,
            "expectancy_points": 0.0,
            "max_drawdown_usd": 0.0,
            "sharpe": 0.0,
        }

    wins = df[df["net_usd"] > 0]
    losses = df[df["net_usd"] <= 0]
    gross_win = float(wins["net_usd"].sum())
    gross_loss = float(-losses["net_usd"].sum())

    equity = df["net_usd"].cumsum().to_numpy()
    returns = df["net_usd"].to_numpy()

    span_days = 1.0
    if len(df) > 1 and df["exit_time"].notna().any():
        span = df["exit_time"].max() - df["entry_time"].min()
        span_days = max(1.0, span.total_seconds() / 86400.0)

    # Per-trade Sharpe annualised by observed trade frequency. This is a
    # comparison statistic between configurations, not a claim about a fund.
    trades_per_year = len(df) / span_days * 365.0
    sharpe = 0.0
    if returns.std(ddof=1) > 0 and len(returns) > 2:
        sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(max(1.0, trades_per_year)))

    by_reason = df["exit_reason"].value_counts().to_dict()

    return {
        "n_trades": int(len(df)),
        "win_rate": float((df["net_usd"] > 0).mean()),
        "total_points": float(df["points"].sum()),
        "net_usd": float(df["net_usd"].sum()),
        "gross_win_usd": gross_win,
        "gross_loss_usd": gross_loss,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "expectancy_usd": float(df["net_usd"].mean()),
        "expectancy_points": float(df["points"].mean()),
        "expectancy_r": float(df["r_multiple"].mean()),
        "avg_win_points": float(wins["points"].mean()) if len(wins) else 0.0,
        "avg_loss_points": float(losses["points"].mean()) if len(losses) else 0.0,
        "best_points": float(df["points"].max()),
        "worst_points": float(df["points"].min()),
        "avg_bars_held": float(df["bars_held"].mean()),
        "max_drawdown_usd": _max_drawdown(equity),
        "sharpe": sharpe,
        "span_days": round(span_days, 1),
        "trades_per_day": round(len(df) / span_days, 2),
        "avg_mfe_points": float(df["mfe_points"].mean()),
        "avg_mae_points": float(df["mae_points"].mean()),
        "pct_hit_20pt": float((df["points"] >= 20).mean()),
        "pct_hit_50pt": float((df["points"] >= 50).mean()),
        "pct_hit_100pt": float((df["points"] >= 100).mean()),
        "exit_reasons": by_reason,
        "long_trades": int((df["side"] == "LONG").sum()),
        "short_trades": int((df["side"] == "SHORT").sum()),
        "long_net_usd": float(df.loc[df["side"] == "LONG", "net_usd"].sum()),
        "short_net_usd": float(df.loc[df["side"] == "SHORT", "net_usd"].sum()),
    }


def format_report(metrics: dict[str, Any], title: str = "Backtest") -> str:
    """Human-readable summary for the console."""
    if metrics.get("n_trades", 0) == 0:
        return f"{title}: no trades taken (gates too tight or no data)."

    lines = [
        f"=== {title} ===",
        f"Trades           : {metrics['n_trades']} over {metrics['span_days']} days "
        f"({metrics['trades_per_day']}/day)",
        f"Win rate         : {metrics['win_rate']:.1%}",
        f"Net P&L          : ${metrics['net_usd']:,.2f}  ({metrics['total_points']:,.1f} pts)",
        f"Profit factor    : {metrics['profit_factor']:.2f}",
        f"Expectancy       : ${metrics['expectancy_usd']:.2f} / trade "
        f"({metrics['expectancy_points']:.1f} pts, {metrics['expectancy_r']:.2f}R)",
        f"Avg win / loss   : {metrics['avg_win_points']:.1f} / {metrics['avg_loss_points']:.1f} pts",
        f"Best / worst     : {metrics['best_points']:.1f} / {metrics['worst_points']:.1f} pts",
        f"Max drawdown     : ${metrics['max_drawdown_usd']:,.2f}",
        f"Sharpe           : {metrics['sharpe']:.2f}",
        f"Avg hold         : {metrics['avg_bars_held']:.1f} bars",
        f"Moves >=20/50/100: {metrics['pct_hit_20pt']:.0%} / {metrics['pct_hit_50pt']:.0%} / "
        f"{metrics['pct_hit_100pt']:.0%}",
        f"Long / short     : {metrics['long_trades']} (${metrics['long_net_usd']:,.0f}) / "
        f"{metrics['short_trades']} (${metrics['short_net_usd']:,.0f})",
        f"Exits            : {metrics['exit_reasons']}",
    ]
    return "\n".join(lines)
