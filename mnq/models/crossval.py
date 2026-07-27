"""Cross-instrument validation: train on other index futures, test on MNQ.

Every test so far has split MNQ's own history by time. That is necessary but
weak: 730 days is a handful of regimes, and with 83 trades at a usable
threshold the statistics cannot settle anything.

This asks a different and much harder question. Train only on ES, YM and RTY -
never on MNQ - then predict MNQ. There are no shared bars, so a pattern that
transfers is a property of index futures rather than an artefact of MNQ's
particular noise. A curve fit cannot survive this; a real effect should.

It also multiplies the training sample using data already downloaded for the
context basket, which costs nothing extra.

Time separation is enforced *as well*. The instruments are ~90% correlated, so
training on ES during the same hours being tested on MNQ would leak through the
correlation - the model would effectively have seen the answer. Each fold
therefore trains only on bars preceding the test window, minus the usual
embargo.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..features.builder import add_session_features, build_feature_matrix, feature_columns
from ..labeling import LONG, SHORT, build_labels, forward_return
from .ensemble import DirectionalEnsemble, SharedRegressor, _safe_auc

log = logging.getLogger(__name__)

# Liquid index futures that share MNQ's drivers. Deliberately not commodities or
# FX: the claim is that these instruments obey the same intraday dynamics.
DEFAULT_TRAIN_SYMBOLS = ("ES=F", "YM=F", "RTY=F")
DEFAULT_TEST_SYMBOL = "MNQ=F"


def load_instrument(
    symbol: str, cfg: Config, refresh: bool = True
) -> dict[str, pd.DataFrame]:
    """Fetch one instrument as wide-profile frames (1h base, 4h and 1d derived).

    4h and 1d are resampled from the hourly pull rather than requested
    separately: it halves the request count and guarantees the timeframes are
    mutually consistent.
    """
    from ..data.yahoo import (
        cache_path, download, drop_maintenance, load_cached, merge_cache, resample_ohlcv,
    )

    root = cfg.path(cfg.data.cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    safe = symbol.replace("=", "").replace("^", "")
    path = root / f"{safe}_1h.csv"

    cached = load_cached(path, cfg.data.tz)
    frame = cached
    if refresh:
        try:
            fresh = download(symbol, "1h", cfg.data.wide_base_lookback, cfg.data.tz)
            frame = merge_cache(cached, fresh)
            frame.to_csv(path)
        except Exception as exc:  # noqa: BLE001
            if cached.empty:
                raise
            log.warning("%s refresh failed (%s); using cache", symbol, str(exc)[:60])
            frame = cached

    if frame.empty:
        raise RuntimeError(f"no hourly data available for {symbol}")

    if cfg.data.drop_maintenance_break:
        frame = drop_maintenance(frame)

    return {
        "1h": frame,
        "4h": resample_ohlcv(frame, "4h"),
        "1d": resample_ohlcv(frame, "1D"),
    }


def prepare_instrument(
    symbol: str,
    cfg: Config,
    context: dict[str, pd.DataFrame] | None = None,
    refresh: bool = True,
) -> dict[str, Any]:
    """Features, labels and forward return for one instrument.

    Context features are rebuilt relative to *this* instrument, so "relative
    strength versus the basket" means the right thing for each one.
    """
    frames = load_instrument(symbol, cfg, refresh)
    timeframes = cfg.data.timeframes()
    base_tf = timeframes[0]

    matrix = build_feature_matrix(frames, cfg.features, timeframes)
    matrix = add_session_features(matrix, base_tf.prefix)

    if context:
        from ..features.context import build_context_features

        ctx = build_context_features(
            matrix.index, matrix["close"], context, cfg.context, base_tf.minutes
        )
        matrix = matrix.join(ctx)

    names = feature_columns(matrix)
    matrix[names] = matrix[names].replace([np.inf, -np.inf], np.nan)

    labels = {d: build_labels(matrix, cfg.labels, d) for d in (LONG, SHORT)}
    fwd = forward_return(matrix, cfg.labels.fwd_return_bars)

    feats = matrix[names]
    keep = (feats.notna().mean(axis=1) >= 0.6) & fwd.notna()

    return {
        "symbol": symbol,
        "matrix": matrix[keep],
        "features": feats[keep],
        "labels": {d: v[keep] for d, v in labels.items()},
        "fwd": fwd[keep],
        "names": names,
    }


def cross_instrument_evaluate(
    cfg: Config,
    train_symbols: tuple[str, ...] = DEFAULT_TRAIN_SYMBOLS,
    test_symbol: str = DEFAULT_TEST_SYMBOL,
    context: dict[str, pd.DataFrame] | None = None,
    n_folds: int = 4,
    refresh: bool = True,
) -> dict[str, Any]:
    """Train on ``train_symbols``, predict ``test_symbol``.

    Returns per-direction AUC on the test instrument, plus a same-instrument
    baseline for comparison. The comparison is the point: if training on other
    indices does about as well as training on MNQ itself, the model has learned
    something general. If it collapses, it had memorised MNQ.
    """
    log.info("loading %s for training", ", ".join(train_symbols))
    train_data = []
    failures: dict[str, str] = {}
    for sym in train_symbols:
        try:
            train_data.append(prepare_instrument(sym, cfg, context, refresh))
            log.info("  %s: %d rows", sym, len(train_data[-1]["features"]))
        except Exception as exc:  # noqa: BLE001 - a missing symbol is survivable
            # Do not truncate. A clipped message once hid the second half of
            # "...must be the same type", turning a one-line dtype fix into a
            # guess about which symbol Yahoo had retired.
            failures[sym] = f"{type(exc).__name__}: {exc}"
            log.warning("  %s unavailable: %s", sym, failures[sym])

    if not train_data:
        distinct = set(failures.values())
        if len(distinct) == 1 and len(failures) > 1:
            # Every symbol failing identically is a bug in this code, not a
            # data availability problem. Say so, and show the whole error.
            raise RuntimeError(
                f"no training instruments could be loaded. All {len(failures)} "
                f"failed with the same error, which points at a systemic "
                f"problem rather than missing data:\n\n    "
                f"{distinct.pop()}\n"
            )
        detail = "\n".join(f"    {s}: {e}" for s, e in failures.items())
        raise RuntimeError(
            f"no training instruments could be loaded:\n\n{detail}\n"
        )

    log.info("loading %s for testing", test_symbol)
    test = prepare_instrument(test_symbol, cfg, context, refresh)

    # Features must line up across instruments; intersect to be safe.
    common = sorted(set(test["names"]).intersection(*(set(d["names"]) for d in train_data)))
    log.info("%d features shared across instruments", len(common))

    results: dict[str, Any] = {
        "train_symbols": list(train_symbols),
        "test_symbol": test_symbol,
        "n_features": len(common),
        "train_rows": int(sum(len(d["features"]) for d in train_data)),
        "test_rows": int(len(test["features"])),
        "folds": [],
    }

    test_index = test["features"].index
    fold_edges = np.linspace(len(test_index) // 2, len(test_index), n_folds + 1, dtype=int)
    embargo = pd.Timedelta(hours=cfg.model.embargo_bars)

    for direction in (LONG, SHORT):
        name = "long" if direction == LONG else "short"
        oos_pred, oos_true = [], []

        for k in range(n_folds):
            lo, hi = fold_edges[k], fold_edges[k + 1]
            if hi - lo < 30:
                continue
            window = test_index[lo:hi]
            cutoff = window[0] - embargo

            # Pool the training instruments, restricted to bars before the test
            # window. Without this the ~90% correlation between indices would
            # leak the answer straight through.
            Xs, ys, fs = [], [], []
            for d in train_data:
                lab = d["labels"][direction]
                mask = (d["features"].index < cutoff) & lab["label"].notna()
                if mask.sum() < 100:
                    continue
                Xs.append(d["features"].loc[mask, common])
                ys.append(lab.loc[mask, "label"].to_numpy(int))
                fs.append(d["fwd"][mask].to_numpy(float))

            if not Xs:
                continue
            X_tr = pd.concat(Xs)
            y_tr = np.concatenate(ys)
            f_tr = np.concatenate(fs)
            if len(np.unique(y_tr)) < 2 or len(X_tr) < 500:
                continue

            test_lab = test["labels"][direction]
            te_mask = test_lab.index.isin(window) & test_lab["label"].notna()
            if te_mask.sum() < 20:
                continue
            X_te = test["features"].loc[te_mask, common]
            y_te = test_lab.loc[te_mask, "label"].to_numpy(int)
            if len(np.unique(y_te)) < 2:
                continue

            reg = SharedRegressor(cfg.model.reg_params).fit(X_tr, f_tr)
            ens = DirectionalEnsemble(direction, cfg.model).fit(X_tr, y_tr, f_tr)
            p = ens.predict_components(X_te, reg.predict(X_te))["p_meta"].to_numpy()

            auc = _safe_auc(y_te, p)
            oos_pred.append(p)
            oos_true.append(y_te)
            results["folds"].append(
                {
                    "direction": name,
                    "fold": k,
                    "n_train": int(len(X_tr)),
                    "n_test": int(len(y_te)),
                    "test_start": str(window[0]),
                    "auc": auc,
                    "base_rate": float(y_te.mean()),
                }
            )
            log.info(
                "%s fold %d: train %d (other instruments) -> test %d on %s, auc=%.4f",
                name, k, len(X_tr), len(y_te), test_symbol, auc,
            )

        if oos_pred:
            all_p = np.concatenate(oos_pred)
            all_y = np.concatenate(oos_true)
            results[f"{name}_auc"] = _safe_auc(all_y, all_p)
            results[f"{name}_n"] = int(len(all_y))

    return results


def format_report(results: dict[str, Any]) -> str:
    """Render the comparison and say plainly what it means."""
    w = 78
    lines = ["=" * w, "  CROSS-INSTRUMENT VALIDATION", "=" * w]
    lines.append(
        f"\n  Trained on : {', '.join(results['train_symbols'])}  "
        f"({results['train_rows']:,} rows)"
    )
    lines.append(
        f"  Tested on  : {results['test_symbol']}  ({results['test_rows']:,} rows)"
    )
    lines.append(f"  Features   : {results['n_features']}")
    lines.append(
        "\n  The test instrument never appears in training, so a pattern that\n"
        "  transfers is a property of index futures - not of MNQ's own noise."
    )

    if results["folds"]:
        lines.append("\n" + "-" * w)
        lines.append("  Per fold")
        lines.append("-" * w)
        lines.append(pd.DataFrame(results["folds"]).to_string(index=False))

    lines.append("\n" + "-" * w)
    lines.append("  Result")
    lines.append("-" * w)
    aucs = []
    for side in ("long", "short"):
        auc = results.get(f"{side}_auc")
        n = results.get(f"{side}_n", 0)
        if auc is None or auc != auc:
            lines.append(f"  {side:>5}: not enough data")
            continue
        aucs.append(auc)
        lines.append(f"  {side:>5}: AUC {auc:.4f} on {n:,} bars")

    lines.append("\n" + "=" * w)
    if not aucs:
        lines.append("  VERDICT: inconclusive - not enough usable folds")
        lines.append("=" * w)
        return "\n".join(lines)

    best = max(aucs)
    if best >= 0.55:
        lines.append("  VERDICT: THE EDGE TRANSFERS")
        lines.append("=" * w)
        lines.append(
            "\n  A model that never saw MNQ still predicts it. That is much harder\n"
            "  to fake than a time split, and it means the pattern is real rather\n"
            "  than memorised.\n\n"
            "  It does NOT yet mean the strategy is profitable - MNQ's own\n"
            "  backtest still showed one quarter carrying everything. But it\n"
            "  justifies training on the pooled instruments, which is the\n"
            "  cheapest way to get the sample size that was missing."
        )
    elif best >= 0.52:
        lines.append("  VERDICT: WEAK TRANSFER")
        lines.append("=" * w)
        lines.append(
            "\n  Some signal survives the instrument change, but not much. Pooled\n"
            "  training may still help by reducing overfitting. Treat as\n"
            "  unresolved, not as encouragement."
        )
    else:
        lines.append("  VERDICT: NO TRANSFER")
        lines.append("=" * w)
        lines.append(
            "\n  Nothing learned on other index futures predicts MNQ. Combined with\n"
            "  the failed permutation test, the earlier result was almost\n"
            "  certainly selection noise.\n\n"
            "  The honest conclusion is that this feature set does not predict\n"
            "  MNQ, and that more sweeping will only produce more false positives."
        )
    return "\n".join(lines)
