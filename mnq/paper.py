"""Paper-trade journal and backtest-drift detection.

The system passed its permutation test at p=0.0067, which justifies forward
paper trading and nothing more. This module is what makes that period
informative rather than just elapsed time: it records every simulated trade and
continuously asks one question -

    **is the live win rate consistent with the backtest, or has it broken?**

That question is easy to answer badly. A run of five losses feels like the edge
is gone and is entirely normal at a 37.5% win rate; conversely a strategy can
degrade for months while the equity curve still drifts up. So the comparison
here is a binomial test against the backtest's win rate, and - just as
importantly - it reports when the sample is still too small to conclude
anything. Silence about statistical power is how a paper-trading period
produces false confidence in both directions.

Nothing here places orders. There is no broker integration anywhere in this
repository, by design.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

COLUMNS = [
    "entry_time", "exit_time", "direction", "entry_price", "exit_price",
    "stop_price", "target_price", "realised_points", "net_usd",
    "exit_reason", "probability", "bars_held",
]

#: Below this, report the sample as too small rather than drawing a conclusion.
MIN_FOR_INFERENCE = 30

#: Two-sided alarm level for the win-rate comparison.
ALPHA = 0.05


@dataclass
class PaperJournal:
    """Append-only record of simulated trades."""

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- write

    def record(self, row: dict[str, Any]) -> None:
        """Append one closed trade. Written immediately so a crash keeps it."""
        frame = pd.DataFrame([{c: row.get(c) for c in COLUMNS}])
        header = not self.path.exists()
        frame.to_csv(self.path, mode="a", header=header, index=False)
        log.info(
            "journal: %s %s -> %s (%.1f pts, $%.2f)",
            row.get("direction"), row.get("entry_price"),
            row.get("exit_price"), row.get("realised_points") or 0.0,
            row.get("net_usd") or 0.0,
        )

    # ----------------------------------------------------------------- read

    def frame(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=COLUMNS)
        try:
            df = pd.read_csv(self.path)
        except Exception as exc:  # noqa: BLE001 - a corrupt journal must not stop trading
            log.warning("could not read journal %s: %s", self.path, exc)
            return pd.DataFrame(columns=COLUMNS)

        # read_csv succeeds on almost anything - a corrupt file parses as
        # headers rather than raising. Validate the schema explicitly, or a
        # later column access takes down the autopilot loop.
        missing = [c for c in ("net_usd", "entry_time") if c not in df.columns]
        if missing:
            log.warning(
                "journal %s is missing %s; ignoring it rather than reporting "
                "results from a file this code did not write",
                self.path, missing,
            )
            return pd.DataFrame(columns=COLUMNS)

        for col in ("entry_time", "exit_time"):
            if col in df:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        df["net_usd"] = pd.to_numeric(df["net_usd"], errors="coerce")
        return df

    def stats(self) -> dict[str, Any]:
        """Live performance so far."""
        df = self.frame()
        closed = df.dropna(subset=["net_usd"]) if "net_usd" in df else df
        if closed.empty:
            return {"n": 0, "win_rate": None, "net_usd": 0.0, "expectancy_usd": None}

        net = closed["net_usd"].astype(float)
        wins = int((net > 0).sum())
        out = {
            "n": int(len(net)),
            "wins": wins,
            "losses": int(len(net) - wins),
            "win_rate": float(wins / len(net)),
            "net_usd": float(net.sum()),
            "expectancy_usd": float(net.mean()),
            "best_usd": float(net.max()),
            "worst_usd": float(net.min()),
        }
        if len(net) > 1:
            se = float(net.std(ddof=1) / np.sqrt(len(net)))
            out["se_usd"] = se
            out["ci95_low_usd"] = out["expectancy_usd"] - 1.96 * se
            out["ci95_high_usd"] = out["expectancy_usd"] + 1.96 * se
        if "entry_time" in closed and closed["entry_time"].notna().any():
            span = closed["entry_time"].max() - closed["entry_time"].min()
            out["span_days"] = float(span.total_seconds() / 86400.0)
        return out


def compare_to_backtest(
    live: dict[str, Any], baseline_win_rate: float | None
) -> dict[str, Any]:
    """Has the live win rate broken away from the backtest?

    A binomial test on the observed wins against the backtest rate. The
    ``verdict`` deliberately distinguishes "no evidence of a problem" from
    "too early to tell", because at realistic trade rates the second state
    lasts for weeks and reporting it as the first would be misleading.
    """
    n = int(live.get("n") or 0)
    result: dict[str, Any] = {
        "n": n,
        "baseline_win_rate": baseline_win_rate,
        "live_win_rate": live.get("win_rate"),
        "verdict": "no trades yet",
        "detail": "",
    }
    if baseline_win_rate is None:
        result["verdict"] = "no baseline"
        result["detail"] = (
            "run `backtest` once so there is a backtest win rate to compare against"
        )
        return result
    if n == 0:
        result["detail"] = "the journal is empty"
        return result

    wins = int(live.get("wins") or 0)
    test = stats.binomtest(wins, n, baseline_win_rate, alternative="two-sided")
    result["p_value"] = float(test.pvalue)
    result["expected_wins"] = baseline_win_rate * n

    # How far the true rate could plausibly be from the backtest, given n.
    lo, hi = test.proportion_ci(confidence_level=0.95)
    result["ci95"] = [float(lo), float(hi)]

    if n < MIN_FOR_INFERENCE:
        result["verdict"] = "too early"
        result["detail"] = (
            f"{n} trades: the 95% interval for the live win rate is "
            f"{lo:.0%}-{hi:.0%}, wide enough to contain almost any outcome. "
            f"Need about {MIN_FOR_INFERENCE} before this test says anything."
        )
    elif test.pvalue < ALPHA and (live.get("win_rate") or 0) < baseline_win_rate:
        result["verdict"] = "broken"
        result["detail"] = (
            f"live win rate {live['win_rate']:.1%} vs backtest "
            f"{baseline_win_rate:.1%} over {n} trades (p={test.pvalue:.4f}). "
            f"That is a real gap, not a losing streak. Stop and re-examine."
        )
    elif test.pvalue < ALPHA:
        result["verdict"] = "outperforming"
        result["detail"] = (
            f"live win rate {live['win_rate']:.1%} is significantly above the "
            f"backtest {baseline_win_rate:.1%} (p={test.pvalue:.4f}). Pleasant, "
            f"but treat it as luck until it persists - the backtest is the "
            f"better estimate of the long run."
        )
    else:
        result["verdict"] = "consistent"
        result["detail"] = (
            f"live win rate {live['win_rate']:.1%} vs backtest "
            f"{baseline_win_rate:.1%} over {n} trades (p={test.pvalue:.2f}) - "
            f"no evidence the edge has changed."
        )
    return result


def baseline_from_artifacts(artifact_dir: Path) -> float | None:
    """The backtest win rate, read from what `backtest` already wrote."""
    import json

    path = Path(artifact_dir) / "profitability.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        value = payload.get("expectancy", {}).get("win_rate")
        return float(value) if value is not None else None
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read %s: %s", path, exc)
        return None


def trade_to_row(trade, probability: float | None = None) -> dict[str, Any]:
    """Flatten a :class:`mnq.trade.manager.Trade` for the journal."""
    from .config import TradeConfig
    from .labeling import LONG

    bars = None
    if trade.entry_time and trade.exit_time:
        bars = int((trade.exit_time - trade.entry_time).total_seconds() // 60)

    return {
        "entry_time": trade.entry_time,
        "exit_time": trade.exit_time,
        "direction": "LONG" if trade.direction == LONG else "SHORT",
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "stop_price": getattr(trade, "stop_price", None),
        "target_price": getattr(trade, "target_price", None),
        "realised_points": trade.realised_points,
        "net_usd": trade.net_usd(TradeConfig()),
        "exit_reason": getattr(trade, "exit_reason", None),
        "probability": probability,
        "bars_held": bars,
    }
