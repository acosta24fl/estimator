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
from .data.vendor import SPECS as VENDOR_SPECS
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
    if getattr(args, "pooled", False):
        return _train_pooled(args, cfg)

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


def _train_pooled(args, cfg: Config) -> int:
    """Train on every index future at once, predict MNQ.

    The cross-instrument test showed the pattern is a property of index futures
    rather than of MNQ, which makes MNQ's own ~13,700 bars an arbitrary
    restriction. Pooling lifts the training sample roughly fourfold using data
    already on disk.
    """
    from .models.pooled import DEFAULT_POOL, format_report, pooled_walk_forward

    # Pooling only makes sense on the wide profile: 60 days of intraday history
    # per symbol is far too short to be worth combining.
    args.profile = "wide"
    _apply_profile(args, cfg)

    context: dict[str, pd.DataFrame] = {}
    if cfg.context.enabled:
        from .data.context import fetch_context

        try:
            context = fetch_context(
                cfg.context, cfg.path(cfg.context.cache_dir), cfg.data.tz,
                refresh=not args.no_refresh,
            )
        except Exception as exc:  # noqa: BLE001 - price-only is still valid
            logging.warning("context unavailable (%s); continuing price-only", exc)

    pool = tuple(args.pool.split(",")) if getattr(args, "pool", None) else DEFAULT_POOL
    print(f"\n--- pooled walk-forward: {', '.join(pool)} ---")
    matrix, preds, metrics = pooled_walk_forward(
        cfg, pool=pool, context=context,
        n_folds=cfg.model.wf_folds, refresh=not args.no_refresh,
    )

    out = ARTIFACT_DIR
    out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out / "wf_predictions.csv")
    (out / "pooled_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float)
    )

    print()
    print(format_report(metrics))
    print(f"\nPredictions -> {out / 'wf_predictions.csv'}")
    print("Run `python -m mnq.cli backtest` next to price them after costs.")
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

    # The backtest total says what happened; this says whether it means
    # anything. A positive sum whose confidence interval spans zero is the
    # most common way a backtest misleads, and it is invisible in the total.
    from .backtest.profit import expectancy, format_report, target_coverage

    exp = expectancy(result.frame, cfg)
    cov = target_coverage(data.matrix, cfg)
    print()
    print(format_report(exp, cov))
    (ARTIFACT_DIR / "profitability.json").write_text(
        json.dumps({"expectancy": exp, "coverage": cov}, indent=2, default=float)
    )

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


def cmd_crossval(args, cfg: Config) -> int:
    """Train on other index futures, test on MNQ."""
    from .models.crossval import (
        DEFAULT_TEST_SYMBOL, DEFAULT_TRAIN_SYMBOLS, cross_instrument_evaluate,
        format_report,
    )

    # Cross-instrument only makes sense on the wide profile: intraday history is
    # capped at 60 days per symbol, which is far too short to pool.
    args.profile = "wide"
    _apply_profile(args, cfg)

    context: dict[str, pd.DataFrame] = {}
    if cfg.context.enabled:
        from .data.context import fetch_context

        try:
            context = fetch_context(
                cfg.context, cfg.path(cfg.context.cache_dir), cfg.data.tz,
                refresh=not args.no_refresh,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("context unavailable (%s); price-only", exc)

    train = tuple(args.train.split(",")) if args.train else DEFAULT_TRAIN_SYMBOLS
    results = cross_instrument_evaluate(
        cfg, train_symbols=train, test_symbol=args.test or DEFAULT_TEST_SYMBOL,
        context=context, n_folds=args.folds, refresh=not args.no_refresh,
    )

    print()
    print(format_report(results))
    (ARTIFACT_DIR / "crossval_results.json").write_text(
        json.dumps(results, indent=2, default=float)
    )
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


def cmd_ingest(args, cfg: Config) -> int:
    """Turn purchased contract files into an adjusted continuous series.

    Yahoo's ``NQ=F`` is a naive front-month splice with an unadjusted gap at
    every quarterly roll. This path replaces it: read contract-level bars, find
    where liquidity actually moved, and back-adjust so the gaps stop
    masquerading as returns.
    """
    from .data.archive import Archive
    from .data.roll import build_continuous, check_point_scale
    from .data.vendor import load_contract_bars
    from .data.yahoo import resample_ohlcv

    archive = Archive(args.archive or cfg.path(cfg.data.archive_dir))
    product = args.root.upper()

    print(f"=== reading {args.vendor} files from {args.source} ===")
    contracts = load_contract_bars(
        args.source, args.vendor, root=product, pattern=args.pattern,
        strict=args.strict,
    )
    total = sum(len(f) for f in contracts.values())
    print(f"  {len(contracts)} contracts, {total:,} bars")
    for code, frame in sorted(contracts.items())[: args.show]:
        print(f"    {code:<8} {len(frame):>9,}  {frame.index[0]:%Y-%m-%d} .. "
              f"{frame.index[-1]:%Y-%m-%d}")
    if len(contracts) > args.show:
        print(f"    ... and {len(contracts) - args.show} more")

    if not args.no_archive:
        archive.write_contracts(product, args.interval, contracts)
        print(f"  archived -> {archive.contract_dir(product, args.interval)}")

    print(f"\n=== building continuous series ({args.method}, {args.adjust}) ===")
    series = build_continuous(
        contracts,
        method=args.method,
        adjustment=args.adjust,
        confirm_sessions=args.confirm_sessions,
        calendar_offset_days=args.calendar_offset,
    )
    print(f"  {series.summary()}")

    rolls = series.roll_frame
    if not rolls.empty:
        print("\n  roll schedule (last 8):")
        for ts, row in rolls.tail(8).iterrows():
            print(f"    {ts:%Y-%m-%d}  {row['from']:>7} -> {row['to']:<7} "
                  f"gap {row['gap']:+8.2f}pt  ratio {row['ratio']:.6f}")

    out_interval = args.interval
    if args.resample:
        from .data.roll import ContinuousSeries

        # Resampling drops the contract column (there is no sensible way to
        # aggregate it), but the roll schedule is kept so the coarser series
        # can still be audited against the gaps that produced it.
        coarse = resample_ohlcv(series.bars.drop(columns=["contract"]), args.resample)
        series = ContinuousSeries(
            coarse, series.rolls, series.adjustment, series.method
        )
        out_interval = args.resample
        print(f"\n  resampled to {args.resample}: {len(coarse):,} bars")

    if not args.no_archive:
        archive.write_continuous(product, out_interval, series)
        print(f"  wrote -> {archive.continuous_path(product, out_interval, args.adjust)}")

    warning = check_point_scale(series.bars, cfg.labels.min_target_points)
    if warning:
        print(f"\n  [!] {warning}")

    print("\nNext: retrain against this archive. The series is now long enough "
          "that a walk-forward result means something.")
    return 0


def cmd_dashboard(args, cfg: Config) -> int:
    """Serve the local dashboard."""
    import threading
    import webbrowser

    import uvicorn

    from .server.app import create_app
    from .server.engine import LiveEngine

    _apply_profile(args, cfg)
    cfg.server.host, cfg.server.port = args.host, args.port

    engine = LiveEngine(cfg)
    # Seed from the cached bars so the chart has history immediately instead of
    # waiting for live alerts to accumulate one minute at a time.
    try:
        frames, _ = _load_frames(args, cfg)
        base_key = cfg.data.timeframes()[0].key
        if base_key in frames and not frames[base_key].empty:
            engine.store.seed(frames[base_key])
            print(f"  seeded {len(engine.store):,} bars from the cache")
    except Exception as exc:  # noqa: BLE001 - an empty chart still serves
        logging.warning("could not seed bars (%s); the chart starts empty", exc)

    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}/"
    print(f"\n  Dashboard: {url}")
    print("  Press Ctrl+C to stop.\n")
    if args.open:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(create_app(cfg, engine), host=args.host, port=args.port,
                log_level="warning")
    return 0


def cmd_auto(args, cfg: Config) -> int:
    """Run unattended: poll, paper-trade, retrain, and serve the dashboard.

    One process. It keeps working while you are not watching, and the dashboard
    is how you check on it.
    """
    import threading
    import webbrowser

    import uvicorn

    from .autopilot import Autopilot, AutopilotConfig
    from .server.app import create_app
    from .server.engine import LiveEngine

    args.profile = "wide"
    _apply_profile(args, cfg)
    cfg.server.host, cfg.server.port = args.host, args.port

    engine = LiveEngine(cfg)
    if engine.bundle is None:
        print("\n  [!] No trained models found. The autopilot will still collect")
        print("      bars and can retrain, but it cannot generate signals until")
        print("      a model exists. Run option 3 first for the full path.\n")

    try:
        frames, _ = _load_frames(args, cfg)
        base_key = cfg.data.timeframes()[0].key
        if base_key in frames and not frames[base_key].empty:
            engine.store.seed(frames[base_key])
            print(f"  seeded {len(engine.store):,} bars from the cache")
    except Exception as exc:  # noqa: BLE001 - it will fill in from polling
        logging.warning("could not seed bars (%s)", exc)

    auto = Autopilot(cfg, engine, AutopilotConfig(
        poll_seconds=args.poll,
        signal_seconds=args.signal_every,
        retrain_hours=args.retrain_hours,
        enabled_retrain=not args.no_retrain,
        retrain_on_start=args.retrain_now,
    ))
    auto.start()

    url = f"http://localhost:{args.port}/"
    print(f"""
  ============================================================
    AUTOPILOT RUNNING - PAPER TRADING ONLY
  ============================================================
    Dashboard    : {url}
    Poll         : every {args.poll}s
    Signal check : every {args.signal_every}s
    Retrain      : {'off' if args.no_retrain else f'every {args.retrain_hours:g}h'}
    Journal      : {ARTIFACT_DIR / 'paper_trades.csv'}

    No orders are placed. There is no broker connected to this
    system. It simulates trades and records what would have
    happened, which is the evidence the next decision needs.

    Leave this window open. Ctrl+C stops it.
  ============================================================
""")
    if args.open:
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()

    try:
        uvicorn.run(create_app(cfg, engine, autopilot=auto),
                    host=args.host, port=args.port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        auto.stop()
        stats = auto.journal.stats()
        if stats.get("n"):
            print(f"\n  Paper trades this session and before: {stats['n']}, "
                  f"win rate {stats['win_rate']:.1%}, net ${stats['net_usd']:+,.2f}")
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

    print("\n=== vendor archive ===")
    try:
        from .data.archive import Archive

        table = Archive(cfg.path(cfg.data.archive_dir)).describe()
        if table.empty:
            print("  none — run `ingest` once you have purchased contract data")
        else:
            for _, row in table.iterrows():
                print(f"  {row['kind']:<11} {row['product']:<5} {row['interval']:<4} "
                      f"{row['items']:>4} item(s)  {row['first']} .. {row['last']}")
    except Exception as exc:  # noqa: BLE001 - status must never fail
        print(f"  unavailable ({exc})")

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
    sp.add_argument("--pooled", action="store_true",
                    help="train on MNQ+ES+YM+RTY together (~4x the sample); "
                         "each fold still trains only on earlier bars")
    sp.add_argument("--pool", type=str, default=None,
                    help="comma-separated pool (default: MNQ=F,ES=F,YM=F,RTY=F)")
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
        "crossval",
        help="train on other index futures and test on MNQ (strongest test)",
    )
    add_data_args(sp)
    sp.add_argument("--train", type=str, default=None,
                    help="comma-separated training symbols (default: ES=F,YM=F,RTY=F)")
    sp.add_argument("--test", type=str, default=None,
                    help="symbol to test on (default: MNQ=F)")
    sp.add_argument("--folds", type=int, default=4)
    sp.set_defaults(func=cmd_crossval)

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

    sp = sub.add_parser(
        "ingest",
        help="build an adjusted continuous series from purchased contract data",
        description=(
            "Read contract-level futures files, detect where volume rolled from "
            "one contract to the next, and back-adjust the splice so roll gaps "
            "stop appearing as returns. This is how you get 20+ years of NQ "
            "instead of 2 years of Yahoo's MNQ=F."
        ),
    )
    sp.add_argument("source", type=Path, help="directory of vendor files")
    sp.add_argument("--vendor", default="firstrate",
                    choices=sorted(VENDOR_SPECS),
                    help="file layout to expect (default: firstrate)")
    sp.add_argument("--root", default="NQ",
                    help="product root to keep, e.g. NQ or ES (default: NQ)")
    sp.add_argument("--interval", default="1m",
                    help="bar interval of the source files (default: 1m)")
    sp.add_argument("--resample", default=None,
                    help="also store a coarser series, e.g. 1h")
    sp.add_argument("--method", default="volume",
                    choices=["volume", "open_interest", "calendar"],
                    help="how to pick the roll date (default: volume)")
    sp.add_argument("--adjust", default="ratio",
                    choices=["ratio", "difference", "none"],
                    help="back-adjustment; ratio keeps returns continuous")
    sp.add_argument("--confirm-sessions", type=int, default=2,
                    help="sessions the next contract must lead before rolling")
    sp.add_argument("--calendar-offset", type=int, default=5,
                    help="days before expiry to roll, for --method calendar")
    sp.add_argument("--pattern", default="*", help="glob to filter source files")
    sp.add_argument("--archive", type=Path, default=None,
                    help="archive root (default: the configured archive_dir)")
    sp.add_argument("--no-archive", action="store_true",
                    help="report only; write nothing")
    sp.add_argument("--strict", action="store_true",
                    help="stop on the first unreadable file instead of skipping")
    sp.add_argument("--show", type=int, default=8,
                    help="how many contracts to list (default: 8)")
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser(
        "dashboard",
        help="serve the local dashboard: chart, projection and system state",
        description=(
            "Opens a localhost page showing the live chart, the model's "
            "direction and expected move in points, the calibration table "
            "behind that projection, and engine state. Everything is served "
            "from this process - no external scripts or keys."
        ),
    )
    add_data_args(sp)
    sp.add_argument("--host", default="127.0.0.1",
                    help="bind address (default: localhost only)")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--open", action="store_true",
                    help="open a browser window once the server is up")
    sp.set_defaults(func=cmd_dashboard)

    sp = sub.add_parser(
        "auto",
        help="run unattended: poll, paper-trade, retrain, serve the dashboard",
        description=(
            "Autonomous local operation. Polls the market, manages simulated "
            "positions through the same rules the backtest uses, journals every "
            "closed trade, tests the live win rate against the backtest, and "
            "retrains on a slow cadence. PAPER ONLY - no broker is connected "
            "and no orders are placed."
        ),
    )
    add_data_args(sp)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--poll", type=int, default=300,
                    help="seconds between market-data refreshes (default: 300)")
    sp.add_argument("--signal-every", type=int, default=600,
                    help="seconds between signal evaluations (default: 600)")
    sp.add_argument("--retrain-hours", type=float, default=168.0,
                    help="hours between retrains (default: 168 = weekly)")
    sp.add_argument("--no-retrain", action="store_true",
                    help="never retrain; keep the current model")
    sp.add_argument("--retrain-now", action="store_true",
                    help="retrain once at startup before settling into the cadence")
    sp.add_argument("--open", action="store_true", help="open the dashboard")
    sp.set_defaults(func=cmd_auto)

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
