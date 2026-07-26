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


def _load_frames(args, cfg: Config) -> dict[str, pd.DataFrame]:
    """Real Yahoo bars, or synthetic ones when asked."""
    if args.synthetic:
        from .data.synthetic import generate_frames

        days = getattr(args, "synthetic_days", 200)
        logging.warning(
            "using SYNTHETIC data (%d days) — results validate the pipeline, "
            "not the strategy", days,
        )
        return generate_frames(n_minutes=60 * 24 * days, seed=args.seed)

    from .data.yahoo import fetch_all

    return fetch_all(cfg, use_cache=True, refresh=not args.no_refresh)


# ---------------------------------------------------------------- commands


def cmd_fetch(args, cfg: Config) -> int:
    from .data.yahoo import fetch_all

    frames = fetch_all(cfg, use_cache=True, refresh=True)
    print(f"\nCached under {cfg.path(cfg.data.cache_dir)}")
    for name, df in frames.items():
        print(f"  {name:>4}: {len(df):>7,} bars   {df.index.min()} .. {df.index.max()}")
    return 0


def cmd_train(args, cfg: Config) -> int:
    from .models.train import (
        prepare, save_bundle, summarise_labels, train_final, walk_forward_evaluate,
    )

    frames = _load_frames(args, cfg)
    data = prepare(frames, cfg)

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

    frames = _load_frames(args, cfg)
    data = prepare(frames, cfg)

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

    if not result.frame.empty:
        path = ARTIFACT_DIR / "backtest_trades.csv"
        result.frame.to_csv(path, index=False)
        print(f"\nTrade log -> {path}")
    return 0


def cmd_sweep(args, cfg: Config) -> int:
    from .backtest.sweep import sweep_barrier_geometry, sweep_decision_params

    if args.barriers:
        frames = _load_frames(args, cfg)
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

    frames = _load_frames(args, cfg)
    data = prepare(frames, cfg)

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

    sp = sub.add_parser("fetch", help="download and cache Yahoo bars")
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


if __name__ == "__main__":
    sys.exit(main())
