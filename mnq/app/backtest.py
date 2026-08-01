"""Replay stored history through the forecast to measure it.

    python -m app.backtest            # replay whatever 1-minute history exists
    python -m app.backtest --dry-run  # measure without writing predictions

Without this you would have to watch the dashboard for hours before the
accuracy panel said anything. The replay walks the 5-minute series in order,
computing each projection from *only* the bars that preceded it — the same
information the live engine would have had — then scores every one against the
bar that actually followed.

Predictions it records land in the same append-only log the live engine writes,
so the chart's "Past Projections" track and accuracy stats populate at once.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import load_settings
from .core import timeframes
from .core.aggregator import aggregate
from .core.forecast import MIN_BARS, compute_forecast
from .core.prediction_log import PredictionLog
from .core.store import BarStore

log = logging.getLogger(__name__)


def replay(
    bars_5m, daily_bars, session, logbook: PredictionLog, strength: float = 1.0
) -> int:
    """Walk the series, projecting each bar from its predecessors only."""
    recorded = 0
    completed = [b for b in bars_5m if b.complete]

    for cutoff in range(MIN_BARS, len(completed)):
        forecast = compute_forecast(
            completed[:cutoff], daily_bars, session, strength=strength
        )
        if not forecast.valid:
            continue
        # The projected bar is the one right after the history slice; a gap in
        # the series means we would be scoring against the wrong bar.
        if completed[cutoff].ts != forecast.anchor_ts:
            continue
        if logbook.observe(forecast) is not None:
            recorded += 1
    return recorded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay history through the forecast.")
    parser.add_argument("--dry-run", action="store_true", help="measure without writing")
    parser.add_argument("--window", type=int, default=500, help="accuracy window")
    parser.add_argument(
        "--strength",
        type=float,
        default=None,
        help="override MNQ_FORECAST_STRENGTH for this run (try 0.5, or 0 for no drift)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = load_settings()
    timeframes.configure_session(settings.session_tz, settings.session_open_hour)

    store = BarStore(settings.bars_dir, settings.max_1m_bars)
    store.ensure_dirs()
    loaded = store.load()
    if not loaded:
        print("No stored 1-minute bars yet. Run the dashboard first.")
        return 1

    bars_5m = aggregate(store.minute_series(), timeframes.get("5m"))
    session = timeframes.session_bucket()

    logbook = PredictionLog(
        settings.data_dir / "predictions.jsonl", persist=not args.dry_run
    )
    if not args.dry_run:
        logbook.load()

    strength = args.strength if args.strength is not None else settings.forecast_strength
    recorded = replay(bars_5m, store.daily_series(), session, logbook, strength)
    acc = logbook.accuracy(bars_5m, args.window)
    print(f"\n5-minute history replayed: {len(bars_5m)} bars (strength {strength})")
    print(f"predictions recorded:      {recorded}")
    if not acc["count"]:
        print("nothing matured yet")
        return 0

    print(f"scored:                    {acc['count']}")
    if acc["direction_rate"] is None:
        print("direction correct:         n/a (no directional calls)")
    else:
        print(
            f"direction correct:         {acc['direction_rate'] * 100:.1f}%"
            f"  over {acc['directional_count']} directional calls (50% = coin flip)"
        )
    print(f"landed in range:           {acc['band_rate'] * 100:.1f}%")
    if acc["skill_score"] is not None:
        print(f"skill score (1-MSE/MSEb):  {acc['skill_score'] * 100:+.1f}%")
    print(f"mean abs error:            {acc['mean_abs_error']:.2f} pts")
    print(f"no-move baseline error:    {acc['baseline_abs_error']:.2f} pts")
    verdict = (
        "model beats the baseline"
        if acc["mean_abs_error"] < acc["baseline_abs_error"]
        else "model does NOT beat the baseline"
    )
    print(f"verdict:                   {verdict}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
