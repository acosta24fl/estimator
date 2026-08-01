"""Measure the projection as a trading rule.

    python -m app.tradetest                    # default costs
    python -m app.tradetest --cost-points 0    # frictionless, for comparison
    python -m app.tradetest --target-pf 2.0

Replays stored history through the forecast, turns every projection into a
5-minute trade, and reports profit factor with costs.

Thresholds are chosen on the **first half** of the data and reported on the
**second half**. Picking the best threshold over the whole sample and quoting
its profit factor is the most common way to produce a great-looking backtest
that does not survive contact with the market, so the number this prints last
is the out-of-sample one.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import load_settings
from .core import timeframes
from .core.aggregator import aggregate
from .core.forecast import compute_forecast
from .core.prediction_log import PredictionLog
from .core.store import BarStore
from .core.trading import (
    DEFAULT_COST_POINTS,
    DOLLARS_PER_POINT,
    bootstrap_profit_factor,
    sharpe,
    simulate,
)
from .backtest import replay


def _fmt(stats) -> str:
    pf = stats.profit_factor
    pf_text = "n/a" if pf is None else f"{pf:.2f}"
    wr = stats.win_rate
    wr_text = "n/a" if wr is None else f"{wr * 100:.1f}%"
    return (
        f"trades {stats.trades:5d}  PF {pf_text:>5s}  win {wr_text:>6s}  "
        f"net {stats.net_points:+8.1f} pts (${stats.net_dollars:+,.0f})  "
        f"maxDD {stats.max_drawdown:6.1f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the projection as a trading rule.")
    parser.add_argument("--cost-points", type=float, default=DEFAULT_COST_POINTS,
                        help="round-turn cost in index points (default 0.75)")
    parser.add_argument("--target-pf", type=float, default=2.0)
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--min-trades", type=int, default=100,
                        help="trades a threshold needs before it may be selected")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    settings = load_settings()
    timeframes.configure_session(settings.session_tz, settings.session_open_hour)

    store = BarStore(settings.bars_dir, settings.max_1m_bars)
    store.ensure_dirs()
    if not store.load():
        print("No stored 1-minute bars yet. Run the dashboard first.")
        return 1

    bars_5m = aggregate(store.minute_series(), timeframes.get("5m"))
    session = timeframes.session_bucket()
    strength = args.strength if args.strength is not None else settings.forecast_strength

    logbook = PredictionLog(settings.data_dir / "_tradetest.jsonl", persist=False)
    replay(bars_5m, store.daily_series(), session, logbook, strength)
    scored = logbook.scored(bars_5m)
    if len(scored) < 100:
        print(f"Only {len(scored)} scored projections — not enough to judge.")
        return 1

    cost = args.cost_points
    print(f"\nscored projections : {len(scored)}")
    print(f"cost per round turn: {cost} pts (${cost * DOLLARS_PER_POINT:.2f} per contract)")
    print(f"target             : PF {args.target_pf}\n")

    # --- every projection traded, no filtering ---------------------------
    gross_stats, _ = simulate(scored, cost_points=0.0)
    net_stats, net_trades = simulate(scored, cost_points=cost)
    print("all projections traded")
    print(f"  before costs   {_fmt(gross_stats)}")
    print(f"  after costs    {_fmt(net_stats)}")
    sr = sharpe(net_trades)
    if sr is not None:
        print(f"  per-trade Sharpe {sr:+.3f}")
    ci = bootstrap_profit_factor(net_trades)
    if ci is not None:
        print(f"  PF 95% bootstrap CI  {ci[0]:.2f} to {ci[1]:.2f}")

    if net_stats.avg_loss > 0:
        need = net_stats.required_win_rate_for(args.target_pf)
        print(
            f"\n  to reach PF {args.target_pf} with this win/loss shape "
            f"(avg win {net_stats.avg_win:.2f} / avg loss {net_stats.avg_loss:.2f}) "
            f"you need a {need * 100:.1f}% win rate"
        )

    # --- threshold chosen in-sample, reported out-of-sample --------------
    half = len(scored) // 2
    train, test = scored[:half], scored[half:]
    thresholds = [0.0, 1, 2, 3, 5, 8, 12, 20]

    print(f"\nthreshold sweep on the FIRST half ({len(train)} projections)")
    best_threshold, best_pf = 0.0, float("-inf")
    for th in thresholds:
        stats, _ = simulate(train, cost_points=cost, threshold=th)
        pf = stats.profit_factor
        marker = ""
        if stats.trades >= args.min_trades and pf is not None and pf > best_pf:
            best_pf, best_threshold, marker = pf, th, "  <- best"
        print(f"  >= {th:5.1f} pts   {_fmt(stats)}{marker}")

    print(f"\nthat threshold ({best_threshold} pts) applied to the UNSEEN second half")
    out, out_trades = simulate(test, cost_points=cost, threshold=best_threshold)
    print(f"  {_fmt(out)}")
    pf = out.profit_factor
    out_ci = bootstrap_profit_factor(out_trades)
    if out_ci is not None:
        print(f"  PF 95% bootstrap CI  {out_ci[0]:.2f} to {out_ci[1]:.2f}")

    print()
    if pf is None:
        print("VERDICT: no losing trades in the out-of-sample slice — sample too small to judge.")
    elif out_ci is not None and out_ci[0] >= args.target_pf:
        print(f"VERDICT: PF {pf:.2f}, CI lower bound {out_ci[0]:.2f} >= {args.target_pf}.")
        print("         Target met with the sample large enough to support it.")
    elif pf >= args.target_pf:
        print(f"VERDICT: PF {pf:.2f} is above {args.target_pf}, but the 95% CI "
              f"({out_ci[0]:.2f} to {out_ci[1]:.2f}) includes lower values."
              if out_ci else f"VERDICT: PF {pf:.2f} is above target on too few trades.")
        print("         Point estimate meets the target; the data does not confirm it.")
    else:
        print(f"VERDICT: PF {pf:.2f} — does NOT meet the {args.target_pf} target.")
    print()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
