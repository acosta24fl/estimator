"""Records every projection and scores it against what actually happened.

A forecast nobody checks is decoration. Each projection is locked once, at the
moment its 5-minute bar opens, using only data available then — so there is no
hindsight — and appended to ``data/predictions.jsonl``. Scoring is derived from
the bar series at read time rather than stored, keeping the same rule as the
rest of the app: persist observations, derive everything else.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..models import Bar
from .forecast import Forecast

log = logging.getLogger(__name__)


@dataclass
class Prediction:
    """A locked projection for the bar starting at ``anchor_ts``."""

    anchor_ts: int
    target_ts: int
    anchor_price: float
    target: float
    low: float
    high: float
    direction: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "anchor_ts": self.anchor_ts,
            "target_ts": self.target_ts,
            "anchor_price": self.anchor_price,
            "target": self.target,
            "low": self.low,
            "high": self.high,
            "direction": self.direction,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Prediction":
        return cls(
            anchor_ts=int(raw["anchor_ts"]),
            target_ts=int(raw["target_ts"]),
            anchor_price=float(raw["anchor_price"]),
            target=float(raw["target"]),
            low=float(raw["low"]),
            high=float(raw["high"]),
            direction=str(raw.get("direction", "flat")),
        )

    @classmethod
    def from_forecast(cls, forecast: Forecast) -> "Prediction":
        return cls(
            anchor_ts=forecast.anchor_ts,
            target_ts=forecast.target_ts,
            anchor_price=forecast.anchor_price,
            target=forecast.target,
            low=forecast.low,
            high=forecast.high,
            direction=forecast.direction,
        )


@dataclass
class ScoredPrediction:
    prediction: Prediction
    actual: float
    error: float  # signed: actual - target
    direction_correct: bool
    within_band: bool

    @property
    def directional(self) -> bool:
        """Whether this prediction actually called a direction.

        A flat projection makes no directional claim, so counting it as a hit
        or a miss would corrupt the direction rate — those are excluded from
        that statistic rather than scored by a different rule.
        """
        return self.prediction.direction in ("up", "down")

    def as_dict(self) -> dict[str, Any]:
        out = self.prediction.as_dict()
        out.update(
            {
                "actual": round(self.actual, 2),
                "error": round(self.error, 2),
                "abs_error": round(abs(self.error), 2),
                "directional": self.directional,
                "direction_correct": self.direction_correct,
                "within_band": self.within_band,
            }
        )
        return out


class PredictionLog:
    def __init__(self, path: Path, max_records: int = 5_000, persist: bool = True) -> None:
        self._path = Path(path)
        self._max = max_records
        #: When False nothing is written to disk — used for dry-run analysis.
        self._persist = persist
        self._lock = threading.RLock()
        self._by_anchor: dict[int, Prediction] = {}

    # -- persistence -------------------------------------------------------

    def load(self) -> int:
        if not self._path.exists():
            return 0
        loaded = 0
        with self._path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    prediction = Prediction.from_dict(json.loads(line))
                except (ValueError, KeyError) as exc:
                    log.warning("skipping bad prediction %s:%d (%s)", self._path, line_no, exc)
                    continue
                self._by_anchor[prediction.anchor_ts] = prediction
                loaded += 1
        self._trim()
        return loaded

    def _append(self, prediction: Prediction) -> None:
        if not self._persist:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(prediction.as_dict(), separators=(",", ":")) + "\n")

    def _trim(self) -> None:
        if len(self._by_anchor) <= self._max:
            return
        for ts in sorted(self._by_anchor)[: len(self._by_anchor) - self._max]:
            self._by_anchor.pop(ts, None)

    # -- recording ---------------------------------------------------------

    def observe(self, forecast: Forecast) -> Prediction | None:
        """Lock this forecast if its bar has not been predicted yet.

        Returns the newly locked prediction, or None if invalid or already
        recorded. Locking once per bar is what keeps the projection honest: it
        cannot be quietly revised as the bar develops.
        """
        if not forecast.valid:
            return None
        with self._lock:
            if forecast.anchor_ts in self._by_anchor:
                return None
            prediction = Prediction.from_forecast(forecast)
            self._by_anchor[prediction.anchor_ts] = prediction
            self._append(prediction)
            self._trim()
            return prediction

    # -- scoring -----------------------------------------------------------

    def scored(self, bars_5m: Sequence[Bar]) -> list[ScoredPrediction]:
        """Score every prediction whose bar has since completed."""
        actuals = {b.ts: b for b in bars_5m if b.complete}
        out: list[ScoredPrediction] = []
        with self._lock:
            predictions = [self._by_anchor[ts] for ts in sorted(self._by_anchor)]

        for prediction in predictions:
            bar = actuals.get(prediction.anchor_ts)
            if bar is None:
                continue  # not matured yet, or no data for that bar
            actual = bar.close
            moved = actual - prediction.anchor_price
            predicted = prediction.target - prediction.anchor_price
            # Flat predictions are excluded from the direction rate entirely
            # (see ScoredPrediction.directional), so False here is never counted.
            direction_correct = (
                prediction.direction in ("up", "down")
                and moved != 0
                and (moved > 0) == (predicted > 0)
            )
            out.append(
                ScoredPrediction(
                    prediction=prediction,
                    actual=actual,
                    error=actual - prediction.target,
                    direction_correct=direction_correct,
                    within_band=prediction.low <= actual <= prediction.high,
                )
            )
        return out

    def accuracy(self, bars_5m: Sequence[Bar], window: int = 100) -> dict[str, Any]:
        recent = self.scored(bars_5m)[-window:]
        if not recent:
            return {
                "count": 0,
                "directional_count": 0,
                "direction_rate": None,
                "band_rate": None,
                "mean_abs_error": None,
                "baseline_abs_error": None,
            }
        n = len(recent)
        directional = [s for s in recent if s.directional]
        # Baseline = "no change": how far price moved anyway. If the model's
        # error is not below this, the projection is adding nothing.
        baseline = sum(abs(s.actual - s.prediction.anchor_price) for s in recent) / n
        return {
            "count": n,
            "directional_count": len(directional),
            "direction_rate": (
                sum(s.direction_correct for s in directional) / len(directional)
                if directional
                else None
            ),
            "band_rate": sum(s.within_band for s in recent) / n,
            "mean_abs_error": sum(abs(s.error) for s in recent) / n,
            "baseline_abs_error": baseline,
        }

    @property
    def count(self) -> int:
        return len(self._by_anchor)
