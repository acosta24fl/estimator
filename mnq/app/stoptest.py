"""Test a stop-loss / take-profit bracket on the trades the system takes.

    python -m app.stoptest --stop 5 --target 25
    python -m app.stoptest --stop 5 --target 25 --tf 5m
    python -m app.stoptest --sweep                    # a grid of both

Replays stored history, generates the same directional calls the live dashboard
would have made (walk-forward — each call uses only bars that had already
closed), enters at the bar's open, then holds to a stop or a target resolved
**minute by minute** against the 1-minute series rather than guessed at from
the parent bar's OHLC.

Every run prints two reference numbers before its own result:

* the **break-even win rate** the bracket needs, ``stop / (stop + target)``
  with costs folded in, and
* the **driftless win rate**, what a coin flip produces at that geometry.

A bracket cannot manufacture an edge. For a driftless price its expectancy is
exactly zero before costs whatever the levels are, so any result has to be read
against those two numbers, not against zero. A 1:5 bracket winning 17% of the
time has found nothing.

Two controls run alongside: the same signals with **no bracket** (the existing
hold-to-bar-close rule), and the same entries with **randomised directions**.
If the bracket result is not clearly better than the random control, the
bracket is reshaping noise.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys

from .config import load_settings
from .core import timeframes
from .core.aggregator import aggregate
from .core.exits import (
    STOP,
    TARGET,
    TIMEOUT,
    BracketRun,
    break_even_win_rate,
    run,
    theoretical_win_rate,
)
from .core.forecast import MIN_BARS, compute_forecast
from .core.outlook import build_outlook
from .core.store import BarStore
from .core.trading import DEFAULT_COST_POINTS, DOLLARS_PER_POINT

log = logging.getLogger(__name__)


def signals(bars, daily, session, settings, tf, min_ratio: float):
    """Walk the series, calling each bar from its predecessors only.

    Same rule the live trader uses: a bullish or bearish call at a bar's open
    is an entry at that open. Bars where the call is NO CALL are skipped, which
    is why the trade count is below the bar count.
    """
    completed = [b for b in bars if b.complete]
    out: list[tuple[int, int, float]] = []

    for cutoff in range(MIN_BARS, len(completed)):
        forecast = compute_forecast(
            completed[:cutoff],
            daily,
            session,
            horizon_seconds=tf.nominal_seconds,
            strength=settings.forecast_strength,
            ridge_lambda=settings.forecast_ridge_lambda,
            min_fit_samples=settings.forecast_min_samples,
        )
        if not forecast.valid:
            continue
        entry_bar = completed[cutoff]
        # The projected bar must be the one we are about to enter, or we would
        # be trading a call made for a different bar.
        if entry_bar.ts != forecast.anchor_ts:
            continue
        outlook = build_outlook(
            forecast,
            horizon_minutes=tf.nominal_seconds // 60,
            skill=None,
            min_ratio=min_ratio,
        )
        if outlook.direction == "bullish":
            out.append((entry_bar.ts, 1, entry_bar.open))
        elif outlook.direction == "bearish":
            out.append((entry_bar.ts, -1, entry_bar.open))
    return out


def _fmt(result: BracketRun, cost: float) -> list[str]:
    stats = result.stats
    reasons = result.by_reason()
    n = len(result.trades) or 1
    pf = stats.profit_factor
    lines = [
        f"    trades          {stats.trades}"
        + (f"   ({result.skipped_overlap} skipped, position already open)"
           if result.skipped_overlap else ""),
        f"    win rate        {0.0 if stats.win_rate is None else stats.win_rate:.1%}",
        f"    profit factor   {'n/a' if pf is None else f'{pf:.2f}'}",
        f"    net             {stats.net_points:+.1f} pts "
        f"(${stats.net_points * DOLLARS_PER_POINT:+,.0f} on one contract)",
        f"    per trade       {stats.net_points / n:+.3f} pts",
        f"    max drawdown    {stats.max_drawdown:.1f} pts",
        f"    exits           {reasons.get(TARGET, 0)} target / "
        f"{reasons.get(STOP, 0)} stop / {reasons.get(TIMEOUT, 0)} timeout",
        f"    avg hold        {result.mean('minutes_held'):.1f} min",
        f"    avg MFE / MAE   {result.mean('mfe'):+.2f} / {result.mean('mae'):+.2f} pts",
    ]
    return lines


def _report(title: str, result: BracketRun, cost: float) -> None:
    print(f"\n  {title}")
    for line in _fmt(result, cost):
        print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test a stop/target bracket.")
    parser.add_argument("--stop", type=float, default=5.0, help="stop in points (default 5)")
    parser.add_argument("--target", type=float, default=25.0, help="target in points")
    parser.add_argument("--tf", default=None, help="signal timeframe (default: configured)")
    parser.add_argument("--cost-points", type=float, default=DEFAULT_COST_POINTS)
    parser.add_argument("--max-hold", type=int, default=120,
                        help="minutes before an unresolved trade is closed out")
    parser.add_argument("--min-ratio", type=float, default=0.0,
                        help="call threshold; 0 trades every bar (default)")
    parser.add_argument("--allow-overlap", action="store_true",
                        help="stack positions instead of one at a time")
    parser.add_argument("--sweep", action="store_true", help="grid of stops and targets")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    settings = load_settings()
    timeframes.configure_session(settings.session_tz, settings.session_open_hour)

    store = BarStore(settings.bars_dir, settings.max_1m_bars)
    store.ensure_dirs()
    if not store.load():
        print("No stored 1-minute bars yet. Run the dashboard first.")
        return 1

    key = args.tf or f"{settings.signal_horizon_minutes}m"
    if key not in timeframes.TIMEFRAMES:
        print(f"unknown timeframe: {key}")
        return 1
    tf = timeframes.get(key)
    minutes = store.minute_series()
    bars = aggregate(minutes, tf)
    session = timeframes.session_bucket()

    print(f"\nsignal timeframe {tf.key}   {len(minutes):,} minute bars   "
          f"cost {args.cost_points} pts/round turn")
    print("generating walk-forward calls ...", flush=True)
    sig = signals(bars, store.daily_series(), session, settings, tf, args.min_ratio)
    if len(sig) < 30:
        print(f"\nOnly {len(sig)} calls in this history — not enough to judge.")
        return 1
    print(f"{len(sig)} directional calls")

    if args.sweep:
        return _sweep(sig, minutes, args)

    stop, target = args.stop, args.target
    breakeven = break_even_win_rate(stop, target, args.cost_points)
    driftless = theoretical_win_rate(stop, target)

    print(f"\n{'=' * 74}")
    print(f"stop {stop:g} pts   target {target:g} pts   "
          f"reward:risk {target / stop:.1f}:1" if stop else "")
    print("=" * 74)
    print(f"\n  the bar this has to clear")
    print(f"    break-even win rate   {breakeven:.1%}   "
          f"(with {args.cost_points} pts cost)")
    print(f"    driftless win rate    {driftless:.1%}   "
          f"(a coin flip at this geometry, by optional stopping)")
    print("    A bracket cannot create an edge. Beating the second number is the")
    print("    only thing that would mean the direction call is doing work.")

    bracket = run(sig, minutes, stop, target, args.cost_points,
                  args.max_hold, args.allow_overlap)
    _report(f"BRACKET   stop {stop:g} / target {target:g}", bracket, args.cost_points)

    # Control 1: the existing rule, no bracket at all.
    plain = run(sig, minutes, 0.0, 0.0, args.cost_points,
                tf.nominal_seconds // 60, args.allow_overlap)
    _report("CONTROL   no stop, no target, held one bar", plain, args.cost_points)

    # Control 2: same entries, coin-flip directions.
    rng = random.Random(args.seed)
    shuffled = [(ts, rng.choice((1, -1)), price) for ts, _, price in sig]
    control = run(shuffled, minutes, stop, target, args.cost_points,
                  args.max_hold, args.allow_overlap)
    _report(f"CONTROL   same bracket, RANDOM directions", control, args.cost_points)

    print(f"\n{'=' * 74}")
    wr = bracket.stats.win_rate or 0.0
    edge = wr - driftless
    print(f"  measured win rate {wr:.1%} vs {driftless:.1%} driftless "
          f"({edge:+.1%} from the direction call)")
    if bracket.stats.profit_factor is None:
        print("  VERDICT: no losing trades — sample too small to judge.")
    elif wr < breakeven:
        print(f"  VERDICT: {wr:.1%} is below the {breakeven:.1%} break-even for this")
        print("           geometry. The bracket loses money by construction here.")
    elif bracket.stats.net_points <= control.stats.net_points:
        print("  VERDICT: does not beat the random-direction control. The result is")
        print("           the bracket reshaping noise, not the call adding value.")
    else:
        print(f"  VERDICT: PF {bracket.stats.profit_factor:.2f}, above break-even and")
        print("           ahead of the random control. Worth a longer sample.")
    print()
    return 0


def _sweep(sig, minutes, args) -> int:
    stops = (3.0, 5.0, 8.0, 12.0, 20.0)
    targets = (10.0, 15.0, 25.0, 40.0)
    print(f"\n{'=' * 74}\ngrid: net points per trade (after {args.cost_points} pts cost)")
    print("=" * 74)
    print("\n   stop \\ target " + "".join(f"{t:>12g}" for t in targets))
    for stop in stops:
        cells = []
        for target in targets:
            result = run(sig, minutes, stop, target, args.cost_points,
                         args.max_hold, args.allow_overlap)
            n = len(result.trades) or 1
            cells.append(f"{result.stats.net_points / n:>12.3f}")
        print(f"   {stop:>6g}      " + "".join(cells))
    print("\n   Every cell is a different bracket on the SAME signals. For a driftless")
    print("   price every cell has the same true value -- zero minus costs -- so the")
    print("   spread across this grid is a direct read on how much of any single")
    print("   result is noise.\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
