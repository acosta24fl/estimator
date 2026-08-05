"""Next-candle prediction and entry placement, measured per timeframe.

    python -m app.candles                    # every timeframe
    python -m app.candles --tf 15m           # one, in detail
    python -m app.candles --quantile 0.9     # a wider envelope
    python -m app.candles --side short

Two questions, answered separately because they have separate answers:

**What will the next candle look like?** Not its direction — its shape. The
envelope is fitted walk-forward and scored on coverage, so a band claiming 80%
that delivers 55% is reported as broken rather than quietly used.

**Where in the bar should the entry go?** Every depth from market-at-open to
1.5 sigma below it, with the fill rate you actually get, the adverse selection
you pay for waiting, and expectancy per *signal* rather than per fill.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import load_settings
from .core import timeframes
from .core.aggregator import aggregate
from .core.candle import DEFAULT_LOOKBACK, calibrate, evaluate, predict
from .core.entries import (
    DEFAULT_OFFSETS,
    LONG,
    SHORT,
    SUSTAINED,
    TOUCH,
    profiles,
    recommend,
)
from .core.store import BarStore
from .core.trading import DEFAULT_COST_POINTS, DOLLARS_PER_POINT

SKIP = {"1d"}  # too few bars stored to standardise meaningfully


def _series(store: BarStore, key: str):
    if key == "1d":
        return store.daily_series()
    return aggregate(store.minute_series(), timeframes.get(key))


def _envelope_table(bars, quantile: float, lookback: int) -> None:
    rows = calibrate(bars, lookback=lookback)
    print("  envelope calibration (walk-forward, out of sample)")
    print("    quantile   high cov   low cov   both   range MAE   vs flat   vs last bar")
    for acc in rows:
        if not acc.bars:
            continue
        flag = ""
        # Coverage should land near the quantile it claims. More than 8
        # percentage points out and the band is not what it says it is.
        if abs(acc.high_coverage - acc.quantile) > 0.08 or abs(acc.low_coverage - acc.quantile) > 0.08:
            flag = "  <- miscalibrated"
        star = " *" if abs(acc.quantile - quantile) < 1e-9 else "  "
        print(
            f"   {star}{acc.quantile:5.2f}    {acc.high_coverage:7.1%}   {acc.low_coverage:7.1%}"
            f"  {acc.contained:5.1%}   {acc.range_mae:8.2f}   {acc.range_skill:+6.1%}"
            f"   {acc.naive_skill:+8.1%}{flag}"
        )


def _entry_table(bars, minutes, step: int, side: int, cost: float, offsets) -> None:
    both = profiles(bars, minutes, step, offsets, direction=side, cost_points=cost)
    label = "long" if side == LONG else "short"
    optimistic = both[TOUCH]
    conservative = {p.offset: p for p in both.get(SUSTAINED, ())}

    print(f"  entry depth sweep ({label}, {cost} pts round turn)")
    print("                 --- wick touch fills you ---   --- 1m close through ---")
    print("    depth   fill%   adv.sel   per fill  PER ATT     fill%   per fill  PER ATT")
    for p in optimistic:
        if not p.attempts:
            continue
        c = conservative.get(p.offset)
        right = (
            f"    {c.fill_rate:5.1%}   {c.points(c.per_fill):+8.2f}"
            f"  {c.points(c.per_attempt):+7.3f}"
            if c is not None and c.fills
            else "         -          -        -"
        )
        flag = ""
        if c is not None and p.per_attempt > 0 >= c.per_attempt:
            flag = "  <- fill assumption"
        thin = "" if p.fills >= 30 else "  (thin)"
        print(
            f"    {p.offset:4.2f}o  {p.fill_rate:5.1%}   {p.points(p.adverse_selection):+7.2f}"
            f"   {p.points(p.per_fill):+8.2f}  {p.points(p.per_attempt):+7.3f}"
            f"{right}{flag}{thin}"
        )
    print("    depths are multiples of sigma; adv.sel and per-* are points")


def _report(key: str, bars, minutes, args) -> None:
    completed = [b for b in bars if b.complete]
    print(f"\n{'=' * 78}\n{key}  ({len(completed)} completed bars)\n{'=' * 78}")
    if len(completed) < 80:
        print("  not enough history to standardise — needs ~80 bars")
        return

    acc = evaluate(completed, quantile=args.quantile, lookback=args.lookback)
    forecast = predict(
        completed,
        quantile=args.quantile,
        lookback=args.lookback,
        timeframe=key,
        step_seconds=timeframes.get(key).nominal_seconds if key in timeframes.TIMEFRAMES else 0,
    )

    if forecast.valid:
        print(
            f"  next candle   open {forecast.open:,.2f}   "
            f"high <= {forecast.high:,.2f}   low >= {forecast.low:,.2f}   "
            f"range {forecast.predicted_range:,.2f} pts"
        )
        print(
            f"                sigma {forecast.sigma:.2f} pts   "
            f"up {forecast.up_multiple:.2f}o   down {forecast.down_multiple:.2f}o   "
            f"typical range {forecast.typical_high - forecast.typical_low:,.2f} pts"
        )
        if acc.bars:
            print(
                f"                measured coverage {acc.high_coverage:.0%} high / "
                f"{acc.low_coverage:.0%} low against a {args.quantile:.0%} claim"
            )
    else:
        print(f"  next candle   unavailable: {forecast.reason}")

    step = timeframes.get(key).nominal_seconds
    print()
    _envelope_table(completed, args.quantile, args.lookback)
    print()
    _entry_table(completed, minutes, step, args.side, args.cost_points, args.offsets)

    rec = recommend(
        completed, timeframe=key, offsets=args.offsets,
        direction=args.side, cost_points=args.cost_points,
        minutes=minutes, step_seconds=step,
    )
    print(f"\n  ENTRY ({rec.fill_model} fills): {rec.reason}")
    if rec.best is None and rec.at_market is not None:
        m = rec.at_market
        pf = "" if m.profit_factor is None else f", PF {m.profit_factor:.2f}"
        print(f"    at the open: {m.points(m.per_attempt):+.3f} pts per signal{pf}")
        print(
            f"    after entering at the open a bar travels a median "
            f"{m.points(m.median_favourable):.2f} pts in favour and "
            f"{m.points(m.p90_adverse):.2f} pts against (90th pct) — "
            "size a stop and target off those, not off the entry study"
        )
    if rec.best is not None:
        b = rec.best
        where = "below" if args.side == LONG else "above"
        if b.offset > 0:
            print(
                f"    place the limit {b.points(b.offset):.2f} pts {where} the open "
                f"({b.offset:g} sigma) — fills {b.fill_rate:.0%} of bars"
            )
        else:
            print("    enter at the open; no limit depth pays for itself here")
        print(
            f"    stop  {b.points(b.p90_adverse):.2f} pts beyond entry "
            f"(covers 90% of post-entry drawdown)"
        )
        print(
            f"    target {b.points(b.median_favourable):.2f} pts "
            f"(the median favourable excursion from that entry)"
        )
        if rec.at_market is not None and rec.best is not rec.at_market:
            print(f"    vs entering at the open: {rec.improvement:+.3f} pts per signal")
        if rec.contested:
            print("    NOT CONFIRMED: this depth needs a wick fill to profit.")
        elif rec.fill_model == TOUCH:
            print(
                "    NOT CONFIRMED: no finer series exists to check the fills against, "
                "so this assumes one printed tick fills you."
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Predict the next candle and locate the best entry, per timeframe."
    )
    parser.add_argument("--tf", default=None, help="one timeframe, e.g. 15m")
    parser.add_argument("--quantile", type=float, default=0.80,
                        help="envelope quantile (default 0.80)")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK,
                        help="rolling window of standardised bars (default 500)")
    parser.add_argument("--cost-points", type=float, default=DEFAULT_COST_POINTS)
    parser.add_argument("--side", choices=("long", "short"), default="long")
    parser.add_argument("--offsets", type=float, nargs="*", default=list(DEFAULT_OFFSETS))
    args = parser.parse_args(argv)
    args.side = LONG if args.side == "long" else SHORT

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    settings = load_settings()
    timeframes.configure_session(settings.session_tz, settings.session_open_hour)

    store = BarStore(settings.bars_dir, settings.max_1m_bars)
    store.ensure_dirs()
    if not store.load():
        print("No stored 1-minute bars yet. Run the dashboard first.")
        return 1

    keys = [args.tf] if args.tf else [t.key for t in timeframes.ordered() if t.key not in SKIP]
    print(f"\ncost {args.cost_points} pts round turn "
          f"(${args.cost_points * DOLLARS_PER_POINT:.2f} per contract)   "
          f"envelope quantile {args.quantile:.0%}   side {'long' if args.side == LONG else 'short'}")

    minutes = store.minute_series()
    for key in keys:
        if key not in timeframes.TIMEFRAMES:
            print(f"unknown timeframe: {key}")
            return 1
        _report(key, _series(store, key), minutes, args)

    print(
        "\nTwo things to read correctly:\n"
        "\n"
        "  PER ATT, not per fill. A deeper limit always looks better per fill\n"
        "  because the entry is better; it only fills on bars that first went\n"
        "  against you, and the ones that ran away in your favour — the winners —\n"
        "  are silently dropped. 'adv.sel' is that cost, measured.\n"
        "\n"
        "  The right-hand block, not the left. The left assumes one printed tick\n"
        "  through your limit fills you in size. The right requires a whole minute\n"
        "  to close through it. Where they disagree in sign, the result is a\n"
        "  statement about fills, not about the market.\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
