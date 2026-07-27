"""Is it mathematically profitable?

A positive backtest total is not an answer to that question. Three things
decide it, and only one of them is the headline number:

**Expectancy per trade.** Win rate alone says nothing - a 35% win rate is
excellent at 3:1 payoff and ruinous at 1:1. What matters is whether the
observed win rate clears the *breakeven* win rate implied by the actual
payoff ratio, after costs.

**Whether the margin is bigger than its own error bar.** Expectancy measured
over 80 trades has a standard error large enough to swallow most edges. This
module reports the t-statistic and a confidence interval, so a positive
expectancy that cannot be distinguished from zero is visible as such.

**What costs take.** Every round turn pays two ticks of slippage plus
commission. On a 25-point target that is a fixed tax on a small number; the
report states it as a percentage of the target so it cannot be waved away.

Nothing here selects a configuration. It prices one. Selection belongs in
:mod:`mnq.backtest.validate`, which corrects for how many were tried.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..config import POINT_VALUE_USD, TICK_SIZE, Config

#: Below this many trades, treat any expectancy as unmeasured rather than real.
MIN_TRADES_FOR_INFERENCE = 30


def round_turn_cost_usd(cfg: Config) -> float:
    """Slippage (both sides) plus commission (both sides), per contract."""
    t = cfg.trade
    slip = t.slippage_ticks * TICK_SIZE * 2.0 * POINT_VALUE_USD
    return slip + t.commission_usd_per_side * 2.0


def expectancy(trades: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    """Per-trade economics of a completed backtest.

    ``trades`` is :attr:`BacktestResult.frame`. Returns a dict of scalars; use
    :func:`format_report` to print it.
    """
    out: dict[str, Any] = {
        "n_trades": 0,
        "profitable": False,
        "reason": "no trades",
    }
    if trades is None or trades.empty or "net_usd" not in trades:
        return out

    net = trades["net_usd"].astype(float)
    n = len(net)
    wins, losses = net[net > 0], net[net <= 0]

    win_rate = len(wins) / n
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(-losses.mean()) if len(losses) else 0.0
    payoff = avg_win / avg_loss if avg_loss > 0 else float("inf")

    # The win rate this payoff ratio needs just to break even. Comparing the
    # observed rate against *this* is the whole question; comparing it against
    # 50% is meaningless for an asymmetric target.
    breakeven_wr = 1.0 / (1.0 + payoff) if np.isfinite(payoff) else 0.0

    exp_usd = float(net.mean())
    sd = float(net.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 and sd > 0 else 0.0
    t_stat = exp_usd / se if se > 0 else 0.0
    # One-sided: the only question is whether expectancy is above zero.
    p_value = float(stats.t.sf(t_stat, df=n - 1)) if n > 1 and se > 0 else 1.0
    ci_lo, ci_hi = (
        (exp_usd - 1.96 * se, exp_usd + 1.96 * se) if se > 0 else (exp_usd, exp_usd)
    )

    cost = round_turn_cost_usd(cfg)
    gross_usd = exp_usd + cost

    span_days = _span_days(trades)
    per_month = n / (span_days / 30.44) if span_days > 0 else 0.0

    out.update(
        {
            "n_trades": n,
            "win_rate": win_rate,
            "avg_win_usd": avg_win,
            "avg_loss_usd": avg_loss,
            "payoff_ratio": payoff,
            "breakeven_win_rate": breakeven_wr,
            "win_rate_margin": win_rate - breakeven_wr,
            "expectancy_usd": exp_usd,
            "expectancy_points": exp_usd / POINT_VALUE_USD,
            "gross_expectancy_usd": gross_usd,
            "cost_per_round_turn_usd": cost,
            "cost_share_of_gross": (cost / gross_usd) if gross_usd > 0 else float("inf"),
            "sd_usd": sd,
            "se_usd": se,
            "t_stat": t_stat,
            "p_value": p_value,
            "ci95_low_usd": ci_lo,
            "ci95_high_usd": ci_hi,
            "total_net_usd": float(net.sum()),
            "span_days": span_days,
            "trades_per_month": per_month,
            "expected_monthly_usd": exp_usd * per_month,
        }
    )

    # "Profitable" is deliberately strict: a positive mean that the sample
    # cannot distinguish from zero is not a finding.
    if n < MIN_TRADES_FOR_INFERENCE:
        out["profitable"] = False
        out["reason"] = f"only {n} trades; need {MIN_TRADES_FOR_INFERENCE} to measure"
    elif exp_usd <= 0:
        out["profitable"] = False
        out["reason"] = f"expectancy is {exp_usd:+.2f} USD per trade"
    elif ci_lo <= 0:
        out["profitable"] = False
        out["reason"] = (
            f"expectancy {exp_usd:+.2f} USD but the 95% interval "
            f"[{ci_lo:+.2f}, {ci_hi:+.2f}] includes zero"
        )
    else:
        out["profitable"] = True
        out["reason"] = (
            f"expectancy {exp_usd:+.2f} USD per trade, 95% interval "
            f"[{ci_lo:+.2f}, {ci_hi:+.2f}] excludes zero"
        )
    return out


def target_coverage(
    matrix: pd.DataFrame, cfg: Config, min_points: float | None = None
) -> dict[str, Any]:
    """How often the market actually offers a move of the required size.

    Asking for 25+ point moves is only sensible if 2xATR clears 25 points often
    enough to leave a tradeable number of bars. This answers that before any
    model is involved.
    """
    want = cfg.trade.min_edge_points if min_points is None else min_points
    if matrix is None or matrix.empty or "atr" not in matrix:
        return {"min_points": want, "bars": 0, "share_clearing": 0.0}

    atr = matrix["atr"].dropna()
    target = atr * cfg.labels.tp_atr_mult
    return {
        "min_points": want,
        "bars": int(len(target)),
        "median_target_points": float(target.median()),
        "share_clearing": float((target >= want).mean()),
        "median_atr_points": float(atr.median()),
    }


def _span_days(trades: pd.DataFrame) -> float:
    for col in ("entry_time", "exit_time"):
        if col in trades:
            ts = pd.to_datetime(trades[col], utc=True, errors="coerce").dropna()
            if len(ts) > 1:
                return float((ts.max() - ts.min()).total_seconds() / 86400.0)
    return 0.0


def format_report(exp: dict[str, Any], cov: dict[str, Any] | None = None) -> str:
    w = 78
    lines = ["=" * w, "  PROFITABILITY", "=" * w, ""]

    if cov:
        lines += [
            f"  Asking for moves of {cov['min_points']:.0f}+ points.",
            f"  Median 2xATR target : {cov.get('median_target_points', 0):.0f} points",
            f"  Bars clearing it    : {cov.get('share_clearing', 0):.0%} "
            f"of {cov.get('bars', 0):,}",
            "",
        ]

    if exp["n_trades"] == 0:
        lines += ["  No trades. Nothing to price.", "=" * w]
        return "\n".join(lines)

    lines += [
        "-" * w,
        "  Per trade",
        "-" * w,
        f"  Trades              : {exp['n_trades']:,} "
        f"({exp['trades_per_month']:.1f}/month over {exp['span_days']:.0f} days)",
        f"  Win rate            : {exp['win_rate']:.1%}",
        f"  Breakeven win rate  : {exp['breakeven_win_rate']:.1%}  "
        f"(implied by the {exp['payoff_ratio']:.2f}:1 payoff)",
        f"  Margin              : {exp['win_rate_margin']:+.1%}",
        "",
        f"  Average win         : ${exp['avg_win_usd']:+,.2f}",
        f"  Average loss        : ${-exp['avg_loss_usd']:+,.2f}",
        f"  Expectancy          : ${exp['expectancy_usd']:+,.2f} per trade "
        f"({exp['expectancy_points']:+.1f} points)",
        "",
        "-" * w,
        "  Costs",
        "-" * w,
        f"  Round turn          : ${exp['cost_per_round_turn_usd']:,.2f} "
        "(2 ticks slippage + commission)",
        f"  Gross expectancy    : ${exp['gross_expectancy_usd']:+,.2f} per trade",
        f"  Costs take          : {exp['cost_share_of_gross']:.0%} of gross"
        if np.isfinite(exp["cost_share_of_gross"])
        else "  Costs take          : all of it (gross is not positive)",
        "",
        "-" * w,
        "  Is the margin bigger than its error bar?",
        "-" * w,
        f"  Std dev per trade   : ${exp['sd_usd']:,.2f}",
        f"  Standard error      : ${exp['se_usd']:,.2f}",
        f"  95% interval        : [${exp['ci95_low_usd']:+,.2f}, "
        f"${exp['ci95_high_usd']:+,.2f}]",
        f"  t / one-sided p     : {exp['t_stat']:.2f} / {exp['p_value']:.4f}",
        "",
        f"  Total net           : ${exp['total_net_usd']:+,.2f}",
        f"  Expected monthly    : ${exp['expected_monthly_usd']:+,.2f} "
        "(1 contract, if the edge holds)",
        "",
        "=" * w,
        f"  VERDICT: {'PROFITABLE' if exp['profitable'] else 'NOT ESTABLISHED'}",
        "=" * w,
        f"  {exp['reason']}.",
        "",
    ]

    if exp["profitable"]:
        lines += [
            "  This prices one configuration. It does not account for how many",
            "  were tried to find it - run `validate` for the permutation test",
            "  before believing it.",
        ]
    else:
        lines += [
            "  Do not trade this. A positive total with an interval spanning",
            "  zero is the single most common way a backtest misleads.",
        ]
    lines.append("=" * w)
    return "\n".join(lines)
