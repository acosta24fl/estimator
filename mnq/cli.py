"""Command line interface.

    python -m mnq.cli fetch      # download and cache MNQ bars from Yahoo
    python -m mnq.cli train      # walk-forward evaluate, then fit final models
    python -m mnq.cli backtest   # simulate on out-of-sample predictions
    python -m mnq.cli sweep      # search decision parameters
    python -m mnq.cli discover   # mine the feature space for new patterns
    python -m mnq.cli serve      # run the webhook + 10-minute signal loop
    python -m mnq.cli status     # inspect saved models and cached data
    python -m mnq.cli test-telegram

Add ``--synthetic`` to any data-driven command to run without network access.
That path is for exercising the plumbing; it says nothing about real edge.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import ARTIFACT_DIR, Config
from .labeling import LONG, SHORT


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)


def _apply_profile(args, cfg: Config) -> None:
    """Let --profile override the configured timeframe stack."""
    profile = getattr(args, "profile", None)
    if profile:
        cfg.data.profile = profile
    if cfg.data.profile == "wide":
        # An hourly base needs an hourly-scale horizon: 24 five-minute bars is
        # two hours, but 24 hourly bars is a day and a half.
        if getattr(args, "horizon", None) is None:
            cfg.labels.horizon_bars = 12
            cfg.labels.fwd_return_bars = 6
            cfg.model.embargo_bars = 18
    if getattr(args, "horizon", None):
        cfg.labels.horizon_bars = args.horizon
        cfg.model.embargo_bars = int(args.horizon * 1.5)
    if getattr(args, "no_context", False):
        cfg.context.enabled = False


def _load_frames(
    args, cfg: Config
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Return ``(price_frames, context_frames)``.

    Context is empty when disabled or unavailable; the pipeline degrades to
    price-only features rather than failing.
    """
    _apply_profile(args, cfg)

    if args.synthetic:
        from .data.synthetic import generate_context, generate_frames, generate_minute_bars

        days = getattr(args, "synthetic_days", 200)
        logging.warning(
            "using SYNTHETIC data (%d days) — results validate the pipeline, "
            "not the strategy", days,
        )
        minutes = generate_minute_bars(n_minutes=60 * 24 * days, seed=args.seed)
        from .data.yahoo import resample_ohlcv

        if cfg.data.profile == "wide":
            frames = {
                "1h": resample_ohlcv(minutes, "1h"),
                "4h": resample_ohlcv(minutes, "4h"),
                "1d": resample_ohlcv(minutes, "1D"),
            }
            ctx_interval = "1h"
        else:
            frames = {
                "5m": resample_ohlcv(minutes, "5min"),
                "15m": resample_ohlcv(minutes, "15min"),
                "4h": resample_ohlcv(minutes, "4h"),
            }
            ctx_interval = "15min"
        context = (
            generate_context(minutes, interval=ctx_interval, seed=args.seed + 1)
            if cfg.context.enabled
            else {}
        )
        return frames, context

    from .data.yahoo import fetch_all

    frames = fetch_all(cfg, use_cache=True, refresh=not args.no_refresh)

    context: dict[str, pd.DataFrame] = {}
    if cfg.context.enabled:
        from .data.context import fetch_context

        # Context is pulled hourly and, for the intraday profile, left hourly:
        # a slower context bar is not a problem, and it keeps the request count
        # (and Yahoo's patience) low.
        try:
            context = fetch_context(
                cfg.context,
                cfg.path(cfg.context.cache_dir),
                cfg.data.tz,
                refresh=not args.no_refresh,
            )
        except Exception as exc:  # noqa: BLE001 - price-only is still a valid run
            logging.warning("context unavailable (%s); continuing price-only", exc)

    return frames, context


# ---------------------------------------------------------------- commands


def cmd_fetch(args, cfg: Config) -> int:
    """Download MNQ bars and, unless disabled, the cross-asset basket."""
    from .data.yahoo import fetch_all

    _apply_profile(args, cfg)

    frames = fetch_all(cfg, use_cache=True, refresh=True)
    print(f"\nMNQ bars cached under {cfg.path(cfg.data.cache_dir)}")
    for name, df in frames.items():
        print(f"  {name:>4}: {len(df):>7,} bars   {df.index.min()} .. {df.index.max()}")

    if cfg.context.enabled:
        from .data.context import describe_batch_result, fetch_context

        print(f"\n{'-'*70}\nCross-asset context (free, same Yahoo source)\n{'-'*70}")
        try:
            context = fetch_context(
                cfg.context, cfg.path(cfg.context.cache_dir), cfg.data.tz, refresh=True
            )
            print(describe_batch_result(cfg.context, context))
        except Exception as exc:  # noqa: BLE001 - price-only is still usable
            print(f"  context unavailable: {exc}")
            print("  Training will fall back to price-only features.")

    return 0


def cmd_train(args, cfg: Config) -> int:
    from .models.train import (
        prepare, save_bundle, summarise_labels, train_final, walk_forward_evaluate,
    )

    frames, context = _load_frames(args, cfg)
    data = prepare(frames, cfg, context)

    print("\n--- label balance ---")
    print(json.dumps(summarise_labels(data), indent=2, default=float))

    print("\n--- purged walk-forward evaluation ---")
    preds, metrics = walk_forward_evaluate(data, cfg)

    out = ARTIFACT_DIR
    out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out / "wf_predictions.csv")
    (out / "wf_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    for side in ("long", "short"):
        auc = metrics.get(f"{side}_oos_auc")
        n = metrics.get(f"{side}_oos_n", 0)
        base = metrics.get(f"{side}_oos_base_rate", 0.0)
        if auc is not None:
            verdict = "no edge" if auc < 0.52 else ("weak" if auc < 0.55 else "usable")
            print(f"  {side:>5}: OOS AUC {auc:.4f} on {n:,} bars "
                  f"(base rate {base:.1%}) -> {verdict}")

    if not args.skip_final:
        print("\n--- fitting final models on all history ---")
        bundle = train_final(data, cfg)
        save_bundle(bundle, cfg)

    print(f"\nPredictions -> {out / 'wf_predictions.csv'}")
    print("Run `python -m mnq.cli backtest` next to price these predictions.")
    return 0


def _predictions_for_backtest(args, cfg: Config):
    """Reuse cached walk-forward predictions when they exist, else regenerate."""
    from .models.train import prepare, walk_forward_evaluate

    frames, context = _load_frames(args, cfg)
    data = prepare(frames, cfg, context)

    cached = ARTIFACT_DIR / "wf_predictions.csv"
    if cached.exists() and not args.retrain:
        preds = pd.read_csv(cached, index_col=0, parse_dates=True)
        preds.index = pd.to_datetime(preds.index, utc=True)
        # Stale predictions against a newer matrix would silently mis-price.
        overlap = preds.index.intersection(data.matrix.index)
        if len(overlap) >= 100:
            logging.info("using cached predictions (%d overlapping bars)", len(overlap))
            return data, preds.loc[overlap]
        logging.warning("cached predictions do not match current data; retraining")

    preds, _ = walk_forward_evaluate(data, cfg)
    return data, preds


def cmd_backtest(args, cfg: Config) -> int:
    from .backtest.engine import run_backtest

    data, preds = _predictions_for_backtest(args, cfg)
    result = run_backtest(data.matrix, preds, cfg, verbose=args.verbose)
    print()
    print(result.report("MNQ walk-forward backtest"))

    if result.metrics["n_trades"] == 0:
        # An empty backtest is ambiguous: it can mean the gates are too tight or
        # that something is broken. Show the distribution so it is neither.
        print("\nWhy no trades fired:")
        print(f"  min_probability = {cfg.trade.min_probability:.2f}, "
              f"min_edge_points = {cfg.trade.min_edge_points:.0f}")
        for side in ("long", "short"):
            col = f"{side}_p_meta"
            if col not in preds:
                continue
            s = preds[col].dropna()
            if s.empty:
                continue
            over = (s >= cfg.trade.min_probability).sum()
            print(f"  {side:>5} probability: max {s.max():.3f}  p99 {s.quantile(0.99):.3f} "
                  f" p95 {s.quantile(0.95):.3f}   bars over threshold: {over}")
        atr = data.matrix["atr"].dropna()
        if len(atr):
            target = atr * cfg.labels.tp_atr_mult
            print(f"  target size: median {target.median():.0f} pts, "
                  f"{(target >= cfg.trade.min_edge_points).mean():.0%} of bars clear "
                  f"min_edge_points")
        print("\n  Lower trade.min_probability toward the p95 above, or run "
              "`sweep` to choose it against both sample halves.")

    if not result.frame.empty:
        path = ARTIFACT_DIR / "backtest_trades.csv"
        result.frame.to_csv(path, index=False)
        print(f"\nTrade log -> {path}")
    return 0


def cmd_sweep(args, cfg: Config) -> int:
    from .backtest.sweep import sweep_barrier_geometry, sweep_decision_params

    if args.barriers:
        frames, _ = _load_frames(args, cfg)
        print("Barrier sweep (retrains per combination; this is slow)...")
        df = sweep_barrier_geometry(frames, cfg)
        out = ARTIFACT_DIR / "sweep_barriers.csv"
    else:
        data, preds = _predictions_for_backtest(args, cfg)
        df = sweep_decision_params(data.matrix, preds, cfg, min_trades=args.min_trades)
        out = ARTIFACT_DIR / "sweep_decision.csv"

    if df.empty:
        print("No configuration produced enough trades. Loosen the gates or add data.")
        return 1

    df.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    print(f"\nTop 12 of {len(df)} configurations (ranked on the weaker half, "
          "not on peak profit):\n")
    print(df.head(12).to_string(index=False))
    print(f"\nFull results -> {out}")
    print(
        "\nA configuration is only credible if is_net_usd and oos_net_usd are "
        "BOTH positive. Anything else is a curve fit."
    )
    return 0


def cmd_discover(args, cfg: Config) -> int:
    from .backtest.sweep import discover_patterns, validate_patterns
    from .models.train import prepare

    frames, context = _load_frames(args, cfg)
    data = prepare(frames, cfg, context)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)

    for direction, name in ((LONG, "LONG"), (SHORT, "SHORT")):
        labels = data.labels[direction]
        cut = int(len(data.features) * args.split)
        print(f"\n=== mining {name} patterns on the first {args.split:.0%} of history ===")
        pats = discover_patterns(
            data.features.iloc[:cut], labels.iloc[:cut], direction,
            min_samples=args.min_samples,
        )
        if pats.empty:
            print("  nothing found")
            continue

        val = validate_patterns(pats, data.features, labels, split_frac=args.split, top_n=30)
        held = val[val["holds"]] if "holds" in val else val.iloc[:0]

        print(f"  {len(pats)} candidates; {len(held)} survived the holdout\n")
        cols = ["kind", "condition", "n", "win_rate", "lift", "oos_n", "oos_lift", "holds"]
        show = held if len(held) else val
        print(show[cols].head(12).to_string(index=False))
        if not len(held):
            print("\n  No pattern kept its edge out of sample — that is the "
                  "expected result on data without exploitable structure.")

        val.to_csv(ARTIFACT_DIR / f"patterns_{name.lower()}.csv", index=False)

    print(f"\nSaved to {ARTIFACT_DIR}/patterns_*.csv")
    return 0


def cmd_validate(args, cfg: Config) -> int:
    """Attack the chosen configuration rather than celebrate it."""
    from .backtest.validate import format_report, validate

    cfg.trade.min_probability = args.min_probability
    if args.trail is not None:
        cfg.trade.trail_atr_mult = args.trail

    data, preds = _predictions_for_backtest(args, cfg)

    sweep_path = ARTIFACT_DIR / "sweep_decision.csv"
    sweep = pd.read_csv(sweep_path) if sweep_path.exists() else None
    if sweep is None:
        logging.info("no sweep_decision.csv found; skipping the multiple-comparison check")

    report = validate(data.matrix, preds, cfg, sweep, n_permutations=args.permutations)
    print()
    print(format_report(report, cfg))

    report.threshold_curve.to_csv(ARTIFACT_DIR / "validation_thresholds.csv", index=False)
    report.quarterly.to_csv(ARTIFACT_DIR / "validation_quarterly.csv", index=False)
    print(f"\nDetail written to {ARTIFACT_DIR}")
    return 0


def cmd_experiment(args, cfg: Config) -> int:
    """Run the decisive comparison: timeframe profile x cross-asset context.

    Four configurations, one table. This answers two questions at once - does a
    regime-diverse sample change the verdict, and do cross-asset features add
    anything - without letting either be confounded with the other.
    """
    from argparse import Namespace

    from .models.train import prepare, walk_forward_evaluate

    combos = [
        ("intraday", False, "5m/60d, price only  (your original run)"),
        ("intraday", True, "5m/60d, + cross-asset"),
        ("wide", False, "1h/730d, price only"),
        ("wide", True, "1h/730d, + cross-asset"),
    ]
    if args.wide_only:
        combos = [c for c in combos if c[0] == "wide"]

    rows = []
    for profile, use_ctx, label in combos:
        print(f"\n{'='*72}\n  {label}\n{'='*72}")
        trial = Config.load(args.config)
        trial.data.profile = profile
        trial.context.enabled = use_ctx

        trial_args = Namespace(**{**vars(args), "profile": profile,
                                  "no_context": not use_ctx, "horizon": args.horizon})
        try:
            frames, context = _load_frames(trial_args, trial)
            data = prepare(frames, trial, context)
            _, metrics = walk_forward_evaluate(data, trial)
        except Exception as exc:  # noqa: BLE001 - one combo failing is informative
            logging.error("configuration failed: %s", exc)
            rows.append({"config": label, "long_auc": None, "short_auc": None,
                         "bars": 0, "features": 0, "note": str(exc)[:60]})
            continue

        rows.append(
            {
                "config": label,
                "long_auc": metrics.get("long_oos_auc"),
                "short_auc": metrics.get("short_oos_auc"),
                "bars": metrics.get("long_oos_n", 0),
                "features": len(data.feature_names),
                "note": "",
            }
        )

    print(f"\n\n{'='*78}\n  RESULTS\n{'='*78}\n")
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    df.to_csv(ARTIFACT_DIR / "experiment_results.csv", index=False)

    best = max(
        (r for r in rows if r["long_auc"] is not None),
        key=lambda r: max(r["long_auc"], r["short_auc"]),
        default=None,
    )
    print("\n" + "-" * 78)
    if best is None:
        print("Every configuration failed. Check the errors above.")
        return 1

    peak = max(best["long_auc"], best["short_auc"])
    print(f"Best: {best['config']}  (AUC {peak:.4f})")
    if peak < 0.52:
        print(
            "\nVERDICT: no edge in any configuration.\n"
            "  Nothing here is tradeable. Do NOT sweep or tune - with this many\n"
            "  configurations something will always look good by chance.\n"
            "  The honest conclusion is that these inputs do not predict MNQ."
        )
    elif peak < 0.55:
        print(
            "\nVERDICT: marginal.\n"
            "  Worth a backtest to see whether it survives costs, but expect\n"
            "  most of it to be eaten. Do not trade on this alone."
        )
    else:
        print(
            "\nVERDICT: worth pursuing.\n"
            "  Run:  python -m mnq.cli backtest --profile <the winning profile>\n"
            "  Require positive results in BOTH sample halves before believing it."
        )
    print("-" * 78)
    return 0


def cmd_serve(args, cfg: Config) -> int:
    from .server.app import run

    if args.seed_yahoo:
        from .data.store import BarStore
        from .data.yahoo import download

        print("Seeding the bar store with recent Yahoo history...")
        minutes = download(cfg.data.symbol, "1m", "7d", cfg.data.tz)
        store = BarStore(cfg.path(cfg.server.bar_store_path), cfg.server.max_bars_retained)
        n = store.seed(minutes)
        store.flush()
        print(f"  seeded {n:,} minute bars")

    if not cfg.server.webhook_secret:
        print(
            "\nWARNING: TV_WEBHOOK_SECRET is not set. The webhook will accept "
            "unauthenticated posts, and an ngrok URL is public.\n"
        )
    print(f"Listening on {cfg.server.host}:{cfg.server.port}")
    print(f"Signals every {cfg.server.signal_interval_minutes} minutes.")
    print("Expose with:  ngrok http", cfg.server.port)
    run(cfg)
    return 0


def cmd_status(args, cfg: Config) -> int:
    print("=== data cache ===")
    cache = cfg.path(cfg.data.cache_dir)
    if cache.exists():
        for f in sorted(cache.glob("*.csv")):
            try:
                df = pd.read_csv(f, index_col=0, parse_dates=True)
                print(f"  {f.name:<24} {len(df):>8,} bars  {df.index.min()} .. {df.index.max()}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {f.name:<24} unreadable ({exc})")
    else:
        print("  none — run `fetch`")

    print("\n=== models ===")
    meta = cfg.path("model_dir") / "ensemble_meta.json"
    if meta.exists():
        print(json.dumps(json.loads(meta.read_text()), indent=2)[:1600])
    else:
        print("  none — run `train`")

    print("\n=== live state ===")
    state = cfg.path(cfg.server.state_path)
    print(state.read_text()[:900] if state.exists() else "  none")

    print("\n=== secrets ===")
    for var, value in (
        ("TELEGRAM_BOT_TOKEN", cfg.telegram.bot_token),
        ("TELEGRAM_CHAT_ID", cfg.telegram.chat_id),
        ("TV_WEBHOOK_SECRET", cfg.server.webhook_secret),
    ):
        print(f"  {var:<20} {'set' if value else 'NOT SET'}")
    return 0


def cmd_test_telegram(args, cfg: Config) -> int:
    from datetime import datetime, timezone

    from .notify.telegram import TelegramNotifier
    from .trade.signals import Signal

    notifier = TelegramNotifier(cfg.telegram)
    if not notifier.configured:
        print("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set.")
        return 1

    demo = Signal(
        timestamp=datetime.now(timezone.utc),
        direction=LONG,
        entry=21050.25, stop=21025.75, target=21099.25,
        probability=0.643, atr=24.5,
        expected_points=49.0, risk_points=24.5, reward_risk=2.0,
        components={"p_xgb": 0.61, "p_lgbm": 0.66, "p_meta": 0.643, "fwd_pred": 0.18},
        context={"note": "connectivity test — not a real signal"},
    )
    ok = notifier.send_signal(demo)
    print("sent" if ok else "failed — check the token, the chat id, and that you "
          "have messaged the bot at least once")
    return 0 if ok else 1


# ------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mnq", description="MNQ futures signal system",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__,
    )
    p.add_argument("--config", type=Path, help="YAML config overriding the defaults")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def add_data_args(sp, synthetic_default_days: int = 200):
        sp.add_argument("--synthetic", action="store_true",
                        help="use generated data instead of Yahoo (offline)")
        sp.add_argument("--synthetic-days", type=int, default=synthetic_default_days)
        sp.add_argument("--seed", type=int, default=5)
        sp.add_argument("--no-refresh", action="store_true",
                        help="use only cached bars; do not call Yahoo")
        sp.add_argument("--profile", choices=["intraday", "wide"], default=None,
                        help="intraday = 5m/15m/4h over ~60 days (one regime); "
                             "wide = 1h/4h/1d over ~730 days (many regimes)")
        sp.add_argument("--horizon", type=int, default=None,
                        help="label horizon in base bars (overrides the profile default)")
        sp.add_argument("--no-context", action="store_true",
                        help="skip cross-asset features; price-only")

    sp = sub.add_parser("fetch", help="download and cache Yahoo bars + context")
    add_data_args(sp)
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser("train", help="walk-forward evaluate and fit final models")
    add_data_args(sp)
    sp.add_argument("--skip-final", action="store_true",
                    help="evaluate only; do not fit or save production models")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("backtest", help="simulate on out-of-sample predictions")
    add_data_args(sp)
    sp.add_argument("--retrain", action="store_true", help="ignore cached predictions")
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("sweep", help="search decision or barrier parameters")
    add_data_args(sp)
    sp.add_argument("--barriers", action="store_true",
                    help="sweep TP/SL/horizon instead (slow: retrains each combo)")
    sp.add_argument("--retrain", action="store_true")
    sp.add_argument("--min-trades", type=int, default=20)
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("discover", help="mine the feature space for patterns")
    add_data_args(sp)
    sp.add_argument("--split", type=float, default=0.6,
                    help="fraction of history to mine; the rest validates")
    sp.add_argument("--min-samples", type=int, default=150)
    sp.set_defaults(func=cmd_discover)

    sp = sub.add_parser(
        "validate",
        help="stress-test a chosen configuration (permutation test and more)",
    )
    add_data_args(sp)
    sp.add_argument("--min-probability", type=float, default=0.62,
                    help="the entry threshold to validate (default: 0.62)")
    sp.add_argument("--trail", type=float, default=None,
                    help="override trail_atr_mult")
    sp.add_argument("--permutations", type=int, default=200,
                    help="number of null runs (more is slower but sharper)")
    sp.add_argument("--retrain", action="store_true")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser(
        "experiment",
        help="compare timeframe profiles x cross-asset context (the decisive test)",
    )
    add_data_args(sp)
    sp.add_argument("--wide-only", action="store_true",
                    help="skip the 60-day intraday configurations")
    sp.set_defaults(func=cmd_experiment)

    sp = sub.add_parser("serve", help="run the webhook and signal loop")
    sp.add_argument("--seed-yahoo", action="store_true",
                    help="preload 7 days of 1m Yahoo bars so signals start immediately")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("status", help="show cached data, models and live state")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("test-telegram", help="send a sample signal")
    sp.set_defaults(func=cmd_test_telegram)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    cfg = Config.load(args.config)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        return args.func(args, cfg)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        # These are the expected operational failures - no data, no model, bad
        # config. A stack trace helps nobody; the message already says what to
        # do. Use -v to see the trace when actually debugging.
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
