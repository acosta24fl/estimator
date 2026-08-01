"""Profile the structure of your stored bars.

    python -m app.analyze                 # all six measurements
    python -m app.analyze --tf 5m         # focus one timeframe for hourly stats

Reports volatility, range acceleration, directional continuation, how bar size
scales with timeframe, wick/body shape, and how all of it varies by hour of the
session.

Every rate carries a 95% interval and a sample count. A rate whose interval
straddles the no-effect value is not evidence of anything, however far from it
the point estimate looks — at these sample sizes that is the normal case, and
the point of measuring is to find out which of these are exceptions.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import load_settings
from .core import timeframes
from .core.aggregator import aggregate
from .core.profiling import (
    acceleration_profile,
    continuation_profile,
    hourly_profile,
    scaling_profile,
    volatility_profile,
    wick_profile,
)
from .core.store import BarStore

_TF_MINUTES = {"1m": 1, "5m": 5, "10m": 10, "15m": 15, "30m": 30, "1h": 60, "4h": 240}


def _pct(value, digits=1):
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def _ci(interval):
    return "" if interval is None else f" [{interval[0] * 100:.2f}–{interval[1] * 100:.2f}]"


def _verdict(sig: bool) -> str:
    return "REAL" if sig else "noise"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile stored bar structure.")
    parser.add_argument("--tf", default="5m", help="timeframe for hourly/wick detail")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    settings = load_settings()
    timeframes.configure_session(settings.session_tz, settings.session_open_hour)

    store = BarStore(settings.bars_dir, settings.max_1m_bars)
    store.ensure_dirs()
    if not store.load():
        print("No stored 1-minute bars yet. Run the dashboard first.")
        return 1

    minutes = store.minute_series()
    series = {key: aggregate(minutes, timeframes.get(key)) for key in _TF_MINUTES}
    focus = args.tf if args.tf in series else "5m"

    flagged: list[bool] = []
    span_hours = (minutes[-1].ts - minutes[0].ts) / 3600 if len(minutes) > 1 else 0
    print(f"\n1-minute bars: {len(minutes)}   spanning {span_hours:.1f} hours")
    print(f"session timezone: {settings.session_tz}")

    # -- 1. volatility ----------------------------------------------------
    print("\n" + "=" * 78)
    print("1. VOLATILITY  (how far price travels per bar)")
    print("=" * 78)
    print(f"{'tf':>5s} {'bars':>6s} {'mean rng':>9s} {'median':>8s} "
          f"{'c2c sd':>8s} {'GK sd':>8s} {'p10':>7s} {'p90':>8s} {'clustering':>11s}")
    for key in _TF_MINUTES:
        v = volatility_profile(series[key])
        if not v.bars:
            continue
        print(f"{key:>5s} {v.bars:6d} {v.mean_range:9.2f} {v.median_range:8.2f} "
              f"{v.close_to_close_sd:8.2f} {v.garman_klass_sd:8.2f} "
              f"{v.range_p10:7.2f} {v.range_p90:8.2f} {v.persistence:11.3f}")
    print("  clustering = lag-1 autocorrelation of bar variance. Real futures")
    print("  typically show 0.2-0.4; near 0 means volatility is unpredictable too.")

    # -- 2. acceleration --------------------------------------------------
    print("\n" + "=" * 78)
    print("2. ACCELERATION  (does an expanding bar predict another?)")
    print("=" * 78)
    print(f"{'tf':>5s} {'n':>6s} {'autocorr':>9s} {'+/-2SE':>8s} "
          f"{'big->big':>9s} {'95% CI':>16s}  verdict")
    for key in _TF_MINUTES:
        a = acceleration_profile(series[key])
        if not a.samples:
            continue
        print(f"{key:>5s} {a.samples:6d} {a.expansion_autocorr:+9.3f} "
              f"{2 * a.expansion_stderr:8.3f} {_pct(a.big_then_big_rate):>9s} "
              f"{_ci(a.big_then_big_ci):>18s}  {_verdict(a.significant)}")
        flagged.append(a.significant)
    print("  'big->big' = P(next range above median | this range above median).")

    # -- 3. continuation --------------------------------------------------
    print("\n" + "=" * 78)
    print("3. CONTINUATION  (does direction persist? 50% = coin flip)")
    print("=" * 78)
    print(f"{'tf':>5s} {'n':>6s} {'same dir':>9s} {'95% CI':>16s} "
          f"{'autocorr':>9s} {'after big':>10s} {'after small':>12s}  verdict")
    for key in _TF_MINUTES:
        c = continuation_profile(series[key])
        if not c.samples:
            continue
        print(f"{key:>5s} {c.samples:6d} {_pct(c.same_direction_rate):>9s} "
              f"{_ci(c.ci):>16s} {c.return_autocorr:+9.3f} "
              f"{_pct(c.after_big_rate):>10s} {_pct(c.after_small_rate):>12s}  "
              f"{_verdict(c.significant)}")
        flagged.append(c.significant)
    print("  'after big' conditions on the largest third of moves, 'after small'")
    print("  on the smallest third — exhaustion and drift can differ.")

    # -- 4. scaling -------------------------------------------------------
    print("\n" + "=" * 78)
    print("4. BAR LENGTH vs TIMEFRAME  (how size grows with horizon)")
    print("=" * 78)
    s = scaling_profile({_TF_MINUTES[k]: series[k] for k in _TF_MINUTES})
    print(f"{'minutes':>8s} {'mean |move|':>12s} {'vs sqrt(T) rule':>17s}")
    if s.points:
        base_minutes, base_move = s.points[0]
        for m, value in s.points:
            expected = base_move * (m / base_minutes) ** 0.5
            print(f"{m:8d} {value:12.2f} {value / expected:16.2f}x")
    print(f"\n  fitted exponent b = {s.exponent:.3f}   (mean |move| ~ T^b)")
    print(f"  0.50 = random walk, >0.55 = trending, <0.45 = mean-reverting")
    print(f"  -> {s.interpretation}")

    # -- 5. wick / body ---------------------------------------------------
    print("\n" + "=" * 78)
    print("5. WICK / BODY SHAPE")
    print("=" * 78)
    print(f"{'tf':>5s} {'bars':>6s} {'body%':>7s} {'upper%':>7s} {'lower%':>7s} "
          f"{'wick:body':>10s}")
    for key in _TF_MINUTES:
        w = wick_profile(series[key])
        if not w.bars:
            continue
        print(f"{key:>5s} {w.bars:6d} {w.body_fraction * 100:6.1f}% "
              f"{w.upper_fraction * 100:6.1f}% {w.lower_fraction * 100:6.1f}% "
              f"{w.wick_to_body:9.2f}x")
    w = wick_profile(series[focus])
    print(f"\n  does wick asymmetry predict the next bar?  ({focus})")
    print(f"    after dominant LOWER wick, next closes up: "
          f"{_pct(w.after_long_lower_up_rate)}{_ci(w.after_long_lower_ci)} "
          f"n={w.lower_samples}  {_verdict(w.after_long_lower_significant)}")
    print(f"    after dominant UPPER wick, next closes up: "
          f"{_pct(w.after_long_upper_up_rate)}{_ci(w.after_long_upper_ci)} "
          f"n={w.upper_samples}  {_verdict(w.after_long_upper_significant)}")
    flagged.append(w.after_long_lower_significant)
    flagged.append(w.after_long_upper_significant)

    # -- 6. hour of day ---------------------------------------------------
    print("\n" + "=" * 78)
    print(f"6. BY HOUR OF DAY  ({focus} bars, {settings.session_tz})")
    print("=" * 78)
    print(f"{'hour':>5s} {'bars':>6s} {'mean rng':>9s} {'|move|':>8s} "
          f"{'drift':>8s} {'':>4s} {'body%':>7s} {'cont':>7s} {'95% CI':>16s} {'volume':>10s}")
    for h in hourly_profile(series[focus], settings.session_tz):
        flag = "*" if h.drift_significant else ""
        flagged.append(h.drift_significant)
        print(f"{h.hour:5d} {h.bars:6d} {h.mean_range:9.2f} {h.mean_abs_move:8.2f} "
              f"{h.mean_move:+8.3f} {flag:>4s} {h.body_fraction * 100:6.1f}% "
              f"{_pct(h.continuation_rate):>7s} {_ci(h.continuation_ci):>16s} "
              f"{h.mean_volume:10.0f}")
    print("  '*' marks an average move more than 2 standard errors from zero.")

    # -- multiple comparisons ---------------------------------------------
    total = len(flagged)
    hits = sum(flagged)
    expected = total * 0.05
    print("\n" + "=" * 78)
    print("HOW MANY OF THESE ARE REAL?")
    print("=" * 78)
    print(f"  tests run: {total}   flagged: {hits}   expected by chance alone: {expected:.1f}")
    if hits <= expected + 1:
        print("  -> consistent with pure chance. Treat every flag above as noise")
        print("     unless it survives on fresh data.")
    else:
        print("  -> more flags than chance predicts, but which ones are real is")
        print("     still unknown. Re-run on data this report has never seen;")
        print("     a genuine effect repeats, a coincidence does not.")
    print("\n  A 95% interval is wrong 1 time in 20 by construction. Running ~50")
    print("  tests therefore produces a couple of REALs on random data every time.")
    print()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
