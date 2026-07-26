"""Training and walk-forward evaluation.

Two entry points:

``walk_forward_evaluate`` measures honestly - it refits at every fold and only
ever scores on data the models were fitted without. Those are the numbers worth
believing.

``train_final`` fits one model on all available history for live use. It has no
honest score by construction; the walk-forward run is where the verdict comes
from.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..config import Config
from ..features.builder import add_session_features, build_feature_matrix, feature_columns
from ..labeling import LONG, SHORT, build_labels, forward_return, label_summary
from .ensemble import DirectionalEnsemble, SharedRegressor, _safe_auc
from .validation import PurgedWalkForward

log = logging.getLogger(__name__)

DIRECTION_NAMES = {LONG: "long", SHORT: "short"}


@dataclass
class TrainingData:
    """The aligned matrix everything downstream consumes."""

    matrix: pd.DataFrame
    features: pd.DataFrame
    labels: dict[int, pd.DataFrame]
    fwd_return: pd.Series
    feature_names: list[str]


def prepare(frames: dict[str, pd.DataFrame], cfg: Config) -> TrainingData:
    """Features + labels + regression target, sharing one index."""
    matrix = build_feature_matrix(frames, cfg.features)
    matrix = add_session_features(matrix)
    names = feature_columns(matrix)

    labels = {d: build_labels(matrix, cfg.labels, d) for d in (LONG, SHORT)}
    fwd = forward_return(matrix, cfg.labels.fwd_return_bars)

    # Drop the warm-up block where the slowest indicator is still undefined.
    # Trees tolerate NaN, but a row where nearly everything is missing is noise.
    feats = matrix[names]
    enough = feats.notna().mean(axis=1) >= 0.6
    keep = enough & fwd.notna()

    log.info(
        "prepared %d/%d usable rows (%.1f%%)",
        int(keep.sum()), len(matrix), 100.0 * keep.mean(),
    )
    return TrainingData(
        matrix=matrix[keep],
        features=feats[keep],
        labels={d: v[keep] for d, v in labels.items()},
        fwd_return=fwd[keep],
        feature_names=names,
    )


def _direction_arrays(
    data: TrainingData, direction: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Index]:
    """Rows with a resolved label for this direction."""
    lab = data.labels[direction]
    valid = lab["label"].notna() & data.fwd_return.notna()
    X = data.features[valid]
    y = lab.loc[valid, "label"].to_numpy(int)
    f = data.fwd_return[valid].to_numpy(float)
    return X, y, f, X.index


def walk_forward_evaluate(
    data: TrainingData, cfg: Config
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Refit at each fold and collect out-of-sample predictions.

    Returns a frame of per-bar predictions (indexed by 5m timestamp, one column
    pair per direction) plus a metrics dict. The prediction frame is what the
    backtester consumes, which is what makes the backtest genuinely
    out-of-sample rather than a replay of fitted values.
    """
    splitter = PurgedWalkForward(
        n_folds=cfg.model.wf_folds,
        initial_train_frac=cfg.model.wf_initial_train,
        embargo=cfg.model.embargo_bars,
    )

    predictions: dict[int, list[pd.DataFrame]] = {LONG: [], SHORT: []}
    metrics: dict[str, Any] = {"folds": []}

    for direction in (LONG, SHORT):
        name = DIRECTION_NAMES[direction]
        X, y, f, idx = _direction_arrays(data, direction)
        if len(X) < 500:
            raise RuntimeError(
                f"only {len(X)} labelled rows for {name}; need at least 500. "
                "Fetch more history or loosen labels.min_target_points."
            )

        for fold_id, (tr, te) in enumerate(splitter.split(len(X))):
            if len(np.unique(y[tr])) < 2:
                log.warning("%s fold %d: single-class train block, skipped", name, fold_id)
                continue

            reg = SharedRegressor(cfg.model.reg_params).fit(X.iloc[tr], f[tr])
            ens = DirectionalEnsemble(direction, cfg.model)
            ens.fit(X.iloc[tr], y[tr], f[tr])

            fwd_te = reg.predict(X.iloc[te])
            comp = ens.predict_components(X.iloc[te], fwd_te)
            comp["y"] = y[te]
            comp["fold"] = fold_id
            comp.index = idx[te]
            predictions[direction].append(comp)

            fold_auc = _safe_auc(y[te], comp["p_meta"].to_numpy())
            metrics["folds"].append(
                {
                    "direction": name,
                    "fold": fold_id,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "test_start": str(idx[te][0]),
                    "test_end": str(idx[te][-1]),
                    "base_rate": float(y[te].mean()),
                    "auc_meta": fold_auc,
                    "auc_xgb": _safe_auc(y[te], comp["p_xgb"].to_numpy()),
                    "auc_lgbm": _safe_auc(y[te], comp["p_lgbm"].to_numpy()),
                    "mean_prob": float(comp["p_meta"].mean()),
                }
            )
            log.info(
                "%s fold %d: n_tr=%d n_te=%d auc=%.4f base=%.3f",
                name, fold_id, len(tr), len(te),
                fold_auc if fold_auc == fold_auc else float("nan"),
                y[te].mean(),
            )

    out = {}
    for direction, chunks in predictions.items():
        if not chunks:
            continue
        name = DIRECTION_NAMES[direction]
        df = pd.concat(chunks).sort_index()
        # Folds are disjoint in time, but guard against overlap regardless.
        df = df[~df.index.duplicated(keep="last")]
        for col in ("p_meta", "p_xgb", "p_lgbm", "fwd_pred", "y"):
            out[f"{name}_{col}"] = df[col]
        out[f"{name}_fold"] = df["fold"]

    preds = pd.DataFrame(out).sort_index()

    for direction in (LONG, SHORT):
        name = DIRECTION_NAMES[direction]
        col, ycol = f"{name}_p_meta", f"{name}_y"
        if col not in preds:
            continue
        sub = preds[[col, ycol]].dropna()
        metrics[f"{name}_oos_auc"] = _safe_auc(sub[ycol].to_numpy(int), sub[col].to_numpy())
        metrics[f"{name}_oos_n"] = int(len(sub))
        metrics[f"{name}_oos_base_rate"] = float(sub[ycol].mean())

    return preds, metrics


def train_final(data: TrainingData, cfg: Config) -> dict[str, Any]:
    """Fit production models on all history and persist them."""
    bundle: dict[str, Any] = {
        "feature_names": data.feature_names,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "config": cfg.to_dict(),
        "directions": {},
    }

    # One regressor for both directions, fitted on every row with a target.
    reg_mask = data.fwd_return.notna()
    reg = SharedRegressor(cfg.model.reg_params).fit(
        data.features[reg_mask], data.fwd_return[reg_mask].to_numpy(float)
    )
    bundle["shared_regressor"] = reg

    for direction in (LONG, SHORT):
        name = DIRECTION_NAMES[direction]
        X, y, f, _ = _direction_arrays(data, direction)
        ens = DirectionalEnsemble(direction, cfg.model).fit(X, y, f)
        bundle["directions"][name] = ens
        log.info("final %s model: %s", name, json.dumps(ens.train_stats, default=float))

    return bundle


def save_bundle(bundle: dict[str, Any], cfg: Config) -> Path:
    out_dir = cfg.path("model_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ensemble.joblib"
    joblib.dump(bundle, path)

    meta = {
        "trained_at": bundle["trained_at"],
        "n_features": len(bundle["feature_names"]),
        "stats": {
            name: ens.train_stats for name, ens in bundle["directions"].items()
        },
    }
    (out_dir / "ensemble_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    log.info("saved model bundle -> %s", path)
    return path


def load_bundle(cfg: Config) -> dict[str, Any]:
    path = cfg.path("model_dir") / "ensemble.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"no trained model at {path}; run `python -m mnq.cli train` first"
        )
    return joblib.load(path)


def summarise_labels(data: TrainingData) -> dict[str, Any]:
    return {DIRECTION_NAMES[d]: label_summary(v) for d, v in data.labels.items()}
