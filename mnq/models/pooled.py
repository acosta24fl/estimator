"""Pooled multi-instrument training.

The cross-instrument test settled the question this module acts on: a model
trained only on ES, YM and RTY predicted MNQ at AUC 0.578 long / 0.552 short,
with every one of eight folds above chance. The pattern is a property of index
futures, not of MNQ's own noise.

That makes MNQ's ~13,700 hourly bars an arbitrary restriction. Pooling all four
instruments gives ~55,000 training rows from data already downloaded, and
sample size was the binding constraint all along - ~180 features and a 270-cell
sweep against 83 trades is what produced a configuration that validated at
p=0.18 with one quarter carrying 97% of the profit.

The one rule that makes pooling legitimate: **every fold trains only on bars
that precede its test window**, across all instruments including MNQ itself.
Index futures are ~90% correlated, so training on ES during the hours being
tested on MNQ would leak the answer straight through that correlation - and it
would look like a spectacular result rather than a bug. The embargo is applied
on top, for the label horizon.

Predictions land in the same frame shape :mod:`mnq.models.train` produces, so
``backtest``, ``sweep`` and ``validate`` consume them unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..labeling import LONG, SHORT
from .crossval import _safe_auc, prepare_instrument
from .ensemble import DirectionalEnsemble, SharedRegressor
from .train import save_bundle

log = logging.getLogger(__name__)

#: MNQ first: it is both a training instrument and the prediction target.
DEFAULT_POOL: tuple[str, ...] = ("MNQ=F", "ES=F", "YM=F", "RTY=F")
DEFAULT_TARGET = "MNQ=F"

DIRECTION_NAMES = {LONG: "long", SHORT: "short"}


def pooled_walk_forward(
    cfg: Config,
    pool: tuple[str, ...] = DEFAULT_POOL,
    target: str = DEFAULT_TARGET,
    context: dict[str, pd.DataFrame] | None = None,
    n_folds: int = 5,
    refresh: bool = True,
    fit_final: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Expanding walk-forward on ``target``, trained on every instrument.

    Returns ``(target_matrix, predictions, metrics)``. The predictions frame is
    indexed by target bar time with ``long_p_meta`` / ``short_p_meta`` columns,
    which is exactly what :func:`mnq.backtest.engine.run_backtest` expects.
    """
    if target not in pool:
        pool = (target, *pool)

    loaded: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    for sym in pool:
        try:
            loaded[sym] = prepare_instrument(sym, cfg, context, refresh)
            log.info("  %s: %d rows", sym, len(loaded[sym]["features"]))
        except Exception as exc:  # noqa: BLE001 - one bad symbol is survivable
            failures[sym] = f"{type(exc).__name__}: {exc}"
            log.warning("  %s unavailable: %s", sym, failures[sym])

    if target not in loaded:
        raise RuntimeError(
            f"the prediction target {target} could not be loaded, so there is "
            f"nothing to predict:\n\n    {failures.get(target, 'unknown error')}\n"
        )

    common = sorted(set.intersection(*(set(d["names"]) for d in loaded.values())))
    tgt = loaded[target]
    log.info(
        "pooled: %d instruments, %d shared features, %d total rows",
        len(loaded), len(common),
        sum(len(d["features"]) for d in loaded.values()),
    )

    metrics: dict[str, Any] = {
        "pool": sorted(loaded),
        "target": target,
        "unavailable": failures,
        "n_features": len(common),
        "pooled_rows": int(sum(len(d["features"]) for d in loaded.values())),
        "target_rows": int(len(tgt["features"])),
        "folds": [],
    }

    index = tgt["features"].index
    edges = np.linspace(len(index) // 2, len(index), n_folds + 1, dtype=int)
    embargo = pd.Timedelta(hours=cfg.model.embargo_bars)

    collected: dict[int, list[pd.DataFrame]] = {LONG: [], SHORT: []}

    for direction in (LONG, SHORT):
        name = DIRECTION_NAMES[direction]
        for k in range(n_folds):
            lo, hi = edges[k], edges[k + 1]
            if hi - lo < 30:
                continue
            window = index[lo:hi]
            cutoff = window[0] - embargo

            X_parts, y_parts, f_parts, per_symbol = [], [], [], {}
            for sym, d in loaded.items():
                lab = d["labels"][direction]
                # The whole legitimacy of pooling rests on this line. It applies
                # to the target instrument too: MNQ's own earlier bars are fair
                # training data, its later ones are not.
                mask = (d["features"].index < cutoff) & lab["label"].notna()
                if mask.sum() < 100:
                    continue
                X_parts.append(d["features"].loc[mask, common])
                y_parts.append(lab.loc[mask, "label"].to_numpy(int))
                f_parts.append(d["fwd"][mask].to_numpy(float))
                per_symbol[sym] = int(mask.sum())

            if not X_parts:
                continue
            X_tr = pd.concat(X_parts)
            y_tr = np.concatenate(y_parts)
            f_tr = np.concatenate(f_parts)
            if len(np.unique(y_tr)) < 2 or len(X_tr) < 500:
                continue

            tgt_lab = tgt["labels"][direction]
            te_mask = tgt_lab.index.isin(window) & tgt_lab["label"].notna()
            if te_mask.sum() < 20:
                continue
            X_te = tgt["features"].loc[te_mask, common]
            y_te = tgt_lab.loc[te_mask, "label"].to_numpy(int)
            if len(np.unique(y_te)) < 2:
                continue

            reg = SharedRegressor(cfg.model.reg_params).fit(X_tr, f_tr)
            ens = DirectionalEnsemble(direction, cfg.model).fit(X_tr, y_tr, f_tr)
            comp = ens.predict_components(X_te, reg.predict(X_te))
            comp["y"] = y_te
            comp["fold"] = k
            comp.index = X_te.index
            collected[direction].append(comp)

            auc = _safe_auc(y_te, comp["p_meta"].to_numpy())
            metrics["folds"].append(
                {
                    "direction": name,
                    "fold": k,
                    "n_train": int(len(X_tr)),
                    "n_test": int(len(y_te)),
                    "test_start": str(window[0]),
                    "auc": auc,
                    "base_rate": float(y_te.mean()),
                    "train_by_symbol": per_symbol,
                }
            )
            log.info(
                "%s fold %d: train %d pooled -> test %d on %s, auc=%.4f",
                name, k, len(X_tr), len(y_te), target, auc,
            )

    out: dict[str, pd.Series] = {}
    for direction, chunks in collected.items():
        if not chunks:
            continue
        name = DIRECTION_NAMES[direction]
        df = pd.concat(chunks).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        for col in ("p_meta", "p_xgb", "p_lgbm", "fwd_pred", "y"):
            if col in df:
                out[f"{name}_{col}"] = df[col]
        out[f"{name}_fold"] = df["fold"]

    predictions = pd.DataFrame(out).sort_index()

    for direction in (LONG, SHORT):
        name = DIRECTION_NAMES[direction]
        col, ycol = f"{name}_p_meta", f"{name}_y"
        if col in predictions and ycol in predictions:
            sub = predictions[[col, ycol]].dropna()
            metrics[f"{name}_auc"] = _safe_auc(
                sub[ycol].to_numpy(int), sub[col].to_numpy()
            )
            metrics[f"{name}_n"] = int(len(sub))

    if fit_final:
        try:
            bundle = fit_pooled_bundle(loaded, common, cfg)
            path = save_bundle(bundle, cfg)
            metrics["final_model"] = str(path)
            log.info("pooled production model saved -> %s", path)
        except Exception as exc:  # noqa: BLE001 - the evaluation still stands
            # Losing the production fit must not discard the walk-forward
            # numbers that took an hour to produce.
            metrics["final_model_error"] = f"{type(exc).__name__}: {exc}"
            log.exception("could not fit the pooled production model")

    return tgt["matrix"], predictions, metrics


def fit_pooled_bundle(
    loaded: dict[str, dict[str, Any]], common: list[str], cfg: Config
) -> dict[str, Any]:
    """Fit the production models on every pooled row, for live scoring.

    This is the step that was missing, and its absence was silent in the worst
    way. ``pooled_walk_forward`` fits a model per fold to *measure* the edge and
    throws each one away, which is correct - a fold model has only seen data up
    to its own cutoff. But nothing then fitted the model that actually gets
    used. Pooled training wrote its predictions and metrics, reported a
    profitable backtest, and left no ``ensemble.joblib`` behind, so the
    dashboard kept saying "no trained model" however many times it was run.

    Unlike the fold models this one trains on *all* history with no cutoff.
    That is right for a production model and wrong for measurement, which is
    exactly why the two are separate: every number quoted about this system
    comes from the fold models above, never from this one.
    """
    features = pd.concat([d["features"][common] for d in loaded.values()])
    fwd = pd.concat([d["fwd"] for d in loaded.values()])

    reg_mask = fwd.notna()
    if reg_mask.sum() < 500:
        raise RuntimeError(
            f"only {int(reg_mask.sum())} pooled rows have a forward return; "
            f"not enough to fit a production model"
        )
    reg = SharedRegressor(cfg.model.reg_params).fit(
        features[reg_mask], fwd[reg_mask].to_numpy(float)
    )

    bundle: dict[str, Any] = {
        "feature_names": list(common),
        "trained_at": pd.Timestamp.now("UTC").isoformat(),
        "config": cfg.to_dict(),
        "pool": sorted(loaded),
        "pooled_rows": int(len(features)),
        "shared_regressor": reg,
        "directions": {},
    }

    for direction in (LONG, SHORT):
        name = DIRECTION_NAMES[direction]
        X_parts, y_parts, f_parts = [], [], []
        for d in loaded.values():
            lab = d["labels"][direction]
            mask = lab["label"].notna()
            if mask.sum() < 100:
                continue
            X_parts.append(d["features"].loc[mask, common])
            y_parts.append(lab.loc[mask, "label"].to_numpy(int))
            f_parts.append(d["fwd"][mask].to_numpy(float))

        if not X_parts:
            raise RuntimeError(f"no labelled pooled rows for the {name} model")

        X = pd.concat(X_parts)
        y = np.concatenate(y_parts)
        f = np.concatenate(f_parts)
        if len(np.unique(y)) < 2:
            raise RuntimeError(f"the pooled {name} labels are all one class")

        ens = DirectionalEnsemble(direction, cfg.model).fit(X, y, f)
        bundle["directions"][name] = ens
        log.info("pooled final %s model: %d rows", name, len(X))

    return bundle


def format_report(metrics: dict[str, Any]) -> str:
    """Human-readable summary of a pooled run."""
    w = 78
    lines = [
        "=" * w,
        "  POOLED MULTI-INSTRUMENT TRAINING",
        "=" * w,
        "",
        f"  Pool     : {', '.join(metrics['pool'])}",
        f"  Target   : {metrics['target']}",
        f"  Rows     : {metrics['pooled_rows']:,} pooled "
        f"({metrics['target_rows']:,} on the target alone)",
        f"  Features : {metrics['n_features']}",
        "",
        "  Every fold trains only on bars preceding its test window, across",
        "  all instruments. Index futures are ~90% correlated, so overlapping",
        "  in time would leak the answer through that correlation.",
        "",
    ]

    if metrics.get("unavailable"):
        lines.append("  Unavailable:")
        for sym, err in metrics["unavailable"].items():
            lines.append(f"    {sym}: {err}")
        lines.append("")

    folds = pd.DataFrame(metrics["folds"])
    if not folds.empty:
        lines += [
            "-" * w,
            "  Per fold",
            "-" * w,
            folds[["direction", "fold", "n_train", "n_test", "test_start",
                   "auc", "base_rate"]].to_string(index=False),
            "",
        ]

    lines += ["-" * w, "  Result", "-" * w]
    for name in ("long", "short"):
        auc, n = metrics.get(f"{name}_auc"), metrics.get(f"{name}_n")
        if auc is None or n is None:
            continue
        verdict = (
            "strong" if auc >= 0.58 else
            "usable" if auc >= 0.55 else
            "weak" if auc >= 0.52 else
            "no edge"
        )
        lines.append(f"  {name:>5}: AUC {auc:.4f} on {n:,} bars -> {verdict}")

    lines += [
        "",
        "  AUC is not profit. Run `backtest` next: it prices these predictions",
        "  after slippage and commission and reports whether the edge survives.",
        "=" * w,
    ]
    return "\n".join(lines)
