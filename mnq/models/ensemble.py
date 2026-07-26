"""The per-direction ensemble.

    XGBoost classifier  -> P(win | direction)
    LightGBM classifier -> P(win | direction)
    XGBoost regressor   -> forward return (shared across both directions)
    Logistic meta       -> final probability from the three above

The regressor is shared because forward return is a directional quantity, not a
per-direction one: a predicted +0.4% is bullish evidence that should raise the
long probability and lower the short one. Training two mirrored copies would
just fit the same target twice.

The meta learner is fitted on **out-of-fold** base predictions. This is the
single most important detail in the file - see :mod:`mnq.models.validation`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ..config import ModelConfig
from .validation import PurgedKFoldInner

log = logging.getLogger(__name__)


def _make_xgb_classifier(params: dict[str, Any], pos_weight: float):
    from xgboost import XGBClassifier

    return XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=pos_weight,
    )


def _make_lgbm_classifier(params: dict[str, Any], pos_weight: float):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(**params, objective="binary", scale_pos_weight=pos_weight)


def _make_xgb_regressor(params: dict[str, Any]):
    from xgboost import XGBRegressor

    return XGBRegressor(**params, objective="reg:squarederror")


def _pos_weight(y: np.ndarray) -> float:
    """Balance the classes. Wins are the minority at 2:1 barriers, and without
    this both boosters happily predict 'loss' for everything."""
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos <= 0:
        return 1.0
    return max(0.2, min(5.0, neg / pos))


@dataclass
class SharedRegressor:
    """Forward-return model, trained once and consumed by both directions."""

    params: dict[str, Any]
    model: Any = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "SharedRegressor":
        self.model = _make_xgb_regressor(self.params)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("SharedRegressor is not fitted")
        return self.model.predict(X)


@dataclass
class DirectionalEnsemble:
    """Ensemble for one direction (long or short)."""

    direction: int
    cfg: ModelConfig
    feature_names: list[str] = field(default_factory=list)
    xgb: Any = None
    lgbm: Any = None
    meta: Any = None
    meta_scaler: Any = None
    train_stats: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        fwd_return: np.ndarray,
        shared_reg_params: dict[str, Any] | None = None,
    ) -> "DirectionalEnsemble":
        """Fit base models, then the meta learner on out-of-fold predictions."""
        if len(X) != len(y):
            raise ValueError("X and y length mismatch")
        self.feature_names = list(X.columns)
        reg_params = shared_reg_params or self.cfg.reg_params

        Xv = X.to_numpy(dtype=float)
        yv = np.asarray(y, dtype=int)
        fv = np.asarray(fwd_return, dtype=float)

        if len(np.unique(yv)) < 2:
            raise ValueError(
                f"direction {self.direction}: training labels are single-class; "
                "widen the sample or loosen the barrier configuration"
            )

        # ---- out-of-fold base predictions for the meta learner -------------
        oof = np.full((len(X), 3), np.nan)
        inner = PurgedKFoldInner(
            n_folds=self.cfg.meta_inner_folds, embargo=self.cfg.embargo_bars
        )
        for tr, te in inner.split(len(X)):
            if len(np.unique(yv[tr])) < 2:
                continue
            w = _pos_weight(yv[tr])
            xgb_f = _make_xgb_classifier(self.cfg.xgb_params, w)
            lgbm_f = _make_lgbm_classifier(self.cfg.lgbm_params, w)
            reg_f = _make_xgb_regressor(reg_params)

            xgb_f.fit(Xv[tr], yv[tr])
            lgbm_f.fit(Xv[tr], yv[tr])
            reg_f.fit(Xv[tr], fv[tr])

            oof[te, 0] = xgb_f.predict_proba(Xv[te])[:, 1]
            oof[te, 1] = lgbm_f.predict_proba(Xv[te])[:, 1]
            oof[te, 2] = reg_f.predict(Xv[te])

        mask = np.isfinite(oof).all(axis=1)
        if mask.sum() < 50:
            raise RuntimeError(
                "not enough out-of-fold rows to fit the meta learner "
                f"({int(mask.sum())}); reduce meta_inner_folds or embargo_bars"
            )

        # ---- meta learner ---------------------------------------------------
        # Forward return is on a different scale to the two probabilities, and
        # unscaled logistic regression would under-weight it.
        self.meta_scaler = StandardScaler().fit(oof[mask])
        self.meta = LogisticRegression(
            max_iter=2000, C=1.0, class_weight="balanced", random_state=self.cfg.random_state
        )
        self.meta.fit(self.meta_scaler.transform(oof[mask]), yv[mask])

        # ---- refit base models on the full training block -------------------
        w_full = _pos_weight(yv)
        self.xgb = _make_xgb_classifier(self.cfg.xgb_params, w_full)
        self.lgbm = _make_lgbm_classifier(self.cfg.lgbm_params, w_full)
        self.xgb.fit(Xv, yv)
        self.lgbm.fit(Xv, yv)

        oof_meta = self.meta.predict_proba(self.meta_scaler.transform(oof[mask]))[:, 1]
        self.train_stats = {
            "n_train": int(len(X)),
            "base_rate": float(yv.mean()),
            "oof_auc_xgb": _safe_auc(yv[mask], oof[mask, 0]),
            "oof_auc_lgbm": _safe_auc(yv[mask], oof[mask, 1]),
            "oof_auc_meta": _safe_auc(yv[mask], oof_meta),
            "oof_brier_meta": float(brier_score_loss(yv[mask], oof_meta)),
            "meta_coef_xgb": float(self.meta.coef_[0][0]),
            "meta_coef_lgbm": float(self.meta.coef_[0][1]),
            "meta_coef_fwd": float(self.meta.coef_[0][2]),
        }
        return self

    # -------------------------------------------------------------- predict

    def predict_proba(self, X: pd.DataFrame, fwd_pred: np.ndarray) -> np.ndarray:
        """Final win probability. ``fwd_pred`` comes from the shared regressor."""
        if self.meta is None:
            raise RuntimeError("ensemble is not fitted")
        X = self._align(X)
        Xv = X.to_numpy(dtype=float)
        stack = np.column_stack(
            [
                self.xgb.predict_proba(Xv)[:, 1],
                self.lgbm.predict_proba(Xv)[:, 1],
                np.asarray(fwd_pred, dtype=float),
            ]
        )
        return self.meta.predict_proba(self.meta_scaler.transform(stack))[:, 1]

    def predict_components(
        self, X: pd.DataFrame, fwd_pred: np.ndarray
    ) -> pd.DataFrame:
        """Per-model output, so a signal can explain where its confidence came
        from instead of being an unauditable number on a phone screen."""
        X = self._align(X)
        Xv = X.to_numpy(dtype=float)
        p_xgb = self.xgb.predict_proba(Xv)[:, 1]
        p_lgbm = self.lgbm.predict_proba(Xv)[:, 1]
        fwd = np.asarray(fwd_pred, dtype=float)
        stack = np.column_stack([p_xgb, p_lgbm, fwd])
        p_meta = self.meta.predict_proba(self.meta_scaler.transform(stack))[:, 1]
        return pd.DataFrame(
            {"p_xgb": p_xgb, "p_lgbm": p_lgbm, "fwd_pred": fwd, "p_meta": p_meta},
            index=X.index,
        )

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reorder/select columns to match training.

        A silent column-order change between training and inference is a class
        of bug that produces plausible-looking garbage, so it is checked here.
        """
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"missing features at prediction time: {missing[:8]}")
        return X[self.feature_names]

    def feature_importance(self, top: int = 25) -> pd.DataFrame:
        if self.xgb is None:
            raise RuntimeError("ensemble is not fitted")
        imp = pd.DataFrame(
            {
                "feature": self.feature_names,
                "xgb": self.xgb.feature_importances_,
                "lgbm": self.lgbm.feature_importances_ / max(1e-9, self.lgbm.feature_importances_.sum()),
            }
        )
        imp["combined"] = imp["xgb"] / max(1e-9, imp["xgb"].sum()) + imp["lgbm"]
        return imp.sort_values("combined", ascending=False).head(top).reset_index(drop=True)


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC-AUC, or NaN when a fold happens to be single-class."""
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))
