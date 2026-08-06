"""Read the decision log: why each simulated trade was taken.

    python -m app.decisions                       # the last few, one line each
    python -m app.decisions --full                # every parameter of the newest
    python -m app.decisions --trade 1785983400    # one trade in full
    python -m app.decisions --by fit.samples      # does this parameter matter?
    python -m app.decisions --fields              # what can I group by?

``data/trades.jsonl`` says what happened. This says what the system believed at
the moment it acted — the fitted coefficients, the feature scores, the track
record as it stood, the configuration in force. Trades without that context
cannot be attributed to anything; with it you can ask whether the trades taken
on a strong momentum coefficient actually did better, and get a number.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

from .config import load_settings
from .core.decisions import DecisionLog, group_by, outcome_summary, value_at

#: Parameters worth slicing results by, with a short description each.
GROUPABLE = [
    ("fit.samples", "how much history the ridge was fitted on"),
    ("fit.r2", "in-sample fit quality"),
    ("factors.momentum.points", "the MACD term's contribution, in points"),
    ("factors.momentum.coefficient", "the fitted momentum coefficient"),
    ("factors.vwap.points", "the VWAP-distance term's contribution"),
    ("factors.structure.points", "the daily-structure term's contribution"),
    ("forecast.expected_move", "size of the projected move"),
    ("forecast.volatility", "the volatility unit at entry"),
    ("forecast.band_half", "half-width of the uncertainty cone"),
    ("outlook.strength", "projected move as a fraction of a typical one"),
    ("accuracy.skill_score", "measured skill at the moment of entry"),
    ("accuracy.direction_rate", "measured direction rate at entry"),
    ("envelope.sigma", "next-candle sigma"),
    ("envelope.range", "predicted next-bar range"),
    ("market.vwap_distance", "distance from session VWAP"),
    ("entry_study.per_attempt_points", "measured expectancy of this entry"),
]


def _when(ts: int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _one_line(d) -> str:
    outcome = d.outcome
    result = "open" if outcome is None else f"{outcome['net_points']:+7.2f} pts"
    move = d.forecast.get("expected_move")
    skill = d.accuracy.get("skill_score")
    return (
        f"  {_when(d.ts)}  {d.side:<5s} @ {d.entry:>10,.2f}  "
        f"{result:>12s}   proj {move if move is None else f'{move:+6.2f}'}  "
        f"conf {d.outlook.get('confidence', '?'):<6s} "
        f"skill {'n/a' if skill is None else f'{skill:+.1%}'}"
    )


def _section(title: str, body: dict) -> None:
    print(f"\n  {title}")
    for key, value in body.items():
        if isinstance(value, float):
            value = round(value, 4)
        print(f"    {key:<26} {value}")


def _full(d) -> None:
    print(f"\n{'=' * 78}")
    print(f"trade {d.trade_id}   {_when(d.ts)}   {d.side.upper()} @ {d.entry:,.2f}")
    print(f"{d.symbol}  {d.timeframe}  {d.horizon_minutes}m horizon  schema {d.schema}")
    print("=" * 78)

    if d.outcome:
        o = d.outcome
        verdict = "WIN" if o["won"] else "LOSS"
        print(
            f"\n  RESULT  {verdict}  {o['net_points']:+.2f} pts "
            f"(${o['net_dollars']:+,.2f})   exit {o['exit']:,.2f}"
        )
    else:
        print("\n  RESULT  still open")

    _section("outlook", d.outlook)
    _section("forecast", d.forecast)

    print("\n  factors (score x fitted coefficient = points)")
    for f in d.factors:
        print(
            f"    {f.get('label', f.get('key')):<22} "
            f"score {f.get('score', 0):+7.3f}  "
            f"coef {f.get('coefficient', 0):+8.3f}  "
            f"= {f.get('points', 0):+7.2f} pts"
        )

    _section("fit", d.fit)
    _section("accuracy at entry", d.accuracy)
    _section("next-candle envelope", d.envelope)
    _section("entry study", d.entry_study)
    _section("market", d.market)
    _section("config", d.config)


def _by(log: DecisionLog, path: str, buckets: int) -> int:
    closed = log.closed()
    if not closed:
        print("\nNo closed trades yet — nothing to attribute.")
        return 0

    groups = group_by(closed, path, buckets)
    if not groups:
        sample = value_at(closed[0], path)
        print(
            f"\nCannot group by '{path}' "
            f"({len(closed)} closed trades, sample value {sample!r}).\n"
            "Run --fields to see what is available."
        )
        return 1

    print(f"\nclosed trades grouped by {path}   ({len(closed)} trades)\n")
    print("  range                     trades   win%     PF    net pts   avg pts")
    for label, chunk in groups:
        s = outcome_summary(chunk)
        pf = "n/a" if s["profit_factor"] is None else f"{s['profit_factor']:5.2f}"
        print(
            f"  {label:<24} {s['trades']:>6d}  {s['win_rate']:5.1%}  {pf}  "
            f"{s['net_points']:+9.2f}  {s['avg_points']:+8.3f}"
        )

    overall = outcome_summary(closed)
    pf = "n/a" if overall["profit_factor"] is None else f"{overall['profit_factor']:.2f}"
    print(
        f"\n  all                      {overall['trades']:>6d}  "
        f"{overall['win_rate']:5.1%}  {pf:>5s}  {overall['net_points']:+9.2f}  "
        f"{overall['avg_points']:+8.3f}"
    )
    print(
        "\n  Equal-count buckets, so each row rests on the same number of trades.\n"
        "  A monotone column is the interesting case; noise usually is not, and\n"
        "  with a few dozen trades per bucket most of this will be noise.\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read the simulated-trade decision log.")
    parser.add_argument("--limit", type=int, default=20, help="rows in the summary list")
    parser.add_argument("--full", action="store_true", help="every parameter of the newest")
    parser.add_argument("--trade", default=None, help="one trade id, in full")
    parser.add_argument("--by", default=None, help="group closed trades by a parameter")
    parser.add_argument("--buckets", type=int, default=3)
    parser.add_argument("--fields", action="store_true", help="list groupable parameters")
    parser.add_argument("--json", action="store_true", help="raw records to stdout")
    args = parser.parse_args(argv)

    if args.fields:
        print("\ngroupable parameters (--by <name>)\n")
        for name, why in GROUPABLE:
            print(f"  {name:<34} {why}")
        print("\nAny dotted path into a record works, not just these.\n")
        return 0

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    settings = load_settings()
    log = DecisionLog(settings.data_dir / "decisions.jsonl")
    loaded = log.load()
    if not loaded:
        print(
            f"\nNo decisions logged yet at {settings.data_dir / 'decisions.jsonl'}.\n"
            "One is written each time the paper trader opens a position.\n"
        )
        return 1

    if args.json:
        for d in log.all():
            print(json.dumps(d.as_dict(), separators=(",", ":")))
        return 0

    if args.trade:
        decision = log.get(args.trade)
        if decision is None:
            print(f"\nNo decision for trade {args.trade}.\n")
            return 1
        _full(decision)
        return 0

    if args.by:
        return _by(log, args.by, args.buckets)

    if args.full:
        _full(log.all()[-1])
        return 0

    everything = log.all()
    closed = log.closed()
    print(f"\n{len(everything)} decisions logged, {len(closed)} closed\n")
    for d in everything[-args.limit :]:
        print(_one_line(d))

    if closed:
        s = outcome_summary(closed)
        pf = "n/a" if s["profit_factor"] is None else f"{s['profit_factor']:.2f}"
        print(
            f"\n  closed: {s['trades']} trades, {s['win_rate']:.1%} win, PF {pf}, "
            f"{s['net_points']:+.2f} pts"
        )
    print(
        "\n  --full for every parameter of the newest, --trade <id> for one,\n"
        "  --by <parameter> to test whether a parameter predicts the result,\n"
        "  --fields to list what you can group by.\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
