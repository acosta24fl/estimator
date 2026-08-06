"""Why each simulated trade was taken, recorded at the moment it was taken.

``trades.jsonl`` is a ledger: side, entry, exit, result. It answers *what
happened*. It cannot answer *what the system believed*, and a one-line prose
``reason`` is not an audit trail — you cannot sort by it, group by it, or ask
"do trades taken with the momentum term above 2 points do better".

This module writes ``data/decisions.jsonl``: one record per trade holding every
number that fed the decision, frozen at entry. Same append-only,
last-record-wins convention as the bar, prediction and trade logs, keyed by
``trade_id`` so a decision joins to its trade and, once the trade closes, the
outcome is appended onto the same key.

What is captured, and why each part
-----------------------------------
==============  ==============================================================
outlook         The headline call: direction, confidence, strength, and whether
                confidence was capped by measured skill.
forecast        The projection itself — target, band, volatility unit.
factors         Each feature's score, its **fitted coefficient**, and its
                contribution in points. Coefficients move every poll as the
                ridge refits, so a trade taken on a momentum coefficient of
                +3.1 was taken by a materially different model than one taken
                at +0.2. Without this the model is unreconstructable.
fit             Sample count and in-sample R^2 behind those coefficients.
accuracy        The track record *as it stood at entry* — direction rate with
                its Wilson interval, band coverage, skill score. Recording it
                afterwards would let hindsight in.
envelope        The next-candle high/low band and the sigma it was scaled by.
entry           The measured entry study: fill model, chosen depth, fill rate,
                expectancy per attempt, suggested stop and target.
market          Price, VWAP distance, bar timestamps.
config          Every setting that changed the decision, so a result can be
                attributed to a configuration rather than guessed at.
==============  ==============================================================

Snapshots are taken **only on entry**, never recomputed on read. A parameter
reconstructed later from current history is not what the system believed then,
and a log that quietly re-derives its own past is worse than no log.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..models import Bar
from .trading import DOLLARS_PER_POINT

log = logging.getLogger(__name__)

#: Bumped when the record shape changes, so a reader can tell mixed files apart
#: instead of guessing from which keys happen to be present.
SCHEMA = 1


@dataclass
class Decision:
    """Everything the system believed when it opened one simulated trade."""

    trade_id: str
    ts: int  # the bar timestamp the trade opened on
    schema: int = SCHEMA
    symbol: str = ""
    timeframe: str = ""
    horizon_minutes: int = 0
    direction: int = 0
    entry: float = 0.0

    outlook: dict[str, Any] = field(default_factory=dict)
    forecast: dict[str, Any] = field(default_factory=dict)
    factors: list[dict[str, Any]] = field(default_factory=list)
    fit: dict[str, Any] = field(default_factory=dict)
    accuracy: dict[str, Any] = field(default_factory=dict)
    envelope: dict[str, Any] = field(default_factory=dict)
    entry_study: dict[str, Any] = field(default_factory=dict)
    market: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    #: Filled in when the trade closes. Absent means still open.
    outcome: dict[str, Any] | None = None

    @property
    def side(self) -> str:
        return "long" if self.direction > 0 else "short"

    def factor(self, key: str) -> dict[str, Any] | None:
        return next((f for f in self.factors if f.get("key") == key), None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "trade_id": self.trade_id,
            "ts": self.ts,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "horizon_minutes": self.horizon_minutes,
            "direction": self.direction,
            "side": self.side,
            "entry": round(self.entry, 2),
            "outlook": self.outlook,
            "forecast": self.forecast,
            "factors": self.factors,
            "fit": self.fit,
            "accuracy": self.accuracy,
            "envelope": self.envelope,
            "entry_study": self.entry_study,
            "market": self.market,
            "config": self.config,
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Decision":
        return cls(
            trade_id=str(raw["trade_id"]),
            ts=int(raw["ts"]),
            schema=int(raw.get("schema", 0)),
            symbol=str(raw.get("symbol", "")),
            timeframe=str(raw.get("timeframe", "")),
            horizon_minutes=int(raw.get("horizon_minutes", 0)),
            direction=int(raw.get("direction", 0)),
            entry=float(raw.get("entry", 0.0)),
            outlook=raw.get("outlook") or {},
            forecast=raw.get("forecast") or {},
            factors=raw.get("factors") or [],
            fit=raw.get("fit") or {},
            accuracy=raw.get("accuracy") or {},
            envelope=raw.get("envelope") or {},
            entry_study=raw.get("entry_study") or {},
            market=raw.get("market") or {},
            config=raw.get("config") or {},
            outcome=raw.get("outcome"),
        )


class DecisionLog:
    """Append-only store of entry snapshots, keyed by trade id."""

    def __init__(self, path: Path, persist: bool = True) -> None:
        self._path = Path(path)
        self._persist = persist
        self._lock = threading.RLock()
        self._decisions: dict[str, Decision] = {}

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
                    decision = Decision.from_dict(json.loads(line))
                except (ValueError, KeyError) as exc:
                    log.warning("skipping bad decision %s:%d (%s)", self._path, line_no, exc)
                    continue
                self._decisions[decision.trade_id] = decision  # last record wins
                loaded += 1
        return loaded

    def _append(self, decision: Decision) -> None:
        if not self._persist:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(decision.as_dict(), separators=(",", ":")) + "\n")

    # -- writing -----------------------------------------------------------

    def record(self, decision: Decision) -> Decision:
        """Store an entry snapshot. Re-recording the same trade is ignored.

        A trade's beliefs are fixed the moment it opens. If a later poll tried
        to write the same id again the numbers would have moved on, and the log
        would silently describe a decision that was never made.
        """
        with self._lock:
            existing = self._decisions.get(decision.trade_id)
            if existing is not None:
                return existing
            self._decisions[decision.trade_id] = decision
            self._append(decision)
            return decision

    def close(self, trade) -> Decision | None:
        """Attach a closed trade's result to its decision.

        Only the outcome is written; every belief field keeps the value it had
        at entry.
        """
        with self._lock:
            decision = self._decisions.get(str(trade.trade_id))
            if decision is None or trade.is_open:
                return None
            net = trade.net_points or 0.0
            decision.outcome = {
                "closed_ts": trade.closed_ts,
                "exit": round(trade.exit, 2) if trade.exit is not None else None,
                "gross_points": (
                    None if trade.gross_points is None else round(trade.gross_points, 2)
                ),
                "net_points": round(net, 2),
                "net_dollars": round(net * DOLLARS_PER_POINT, 2),
                "won": net > 0,
            }
            self._append(decision)
            return decision

    # -- reading -----------------------------------------------------------

    def all(self) -> list[Decision]:
        return sorted(self._decisions.values(), key=lambda d: d.ts)

    def get(self, trade_id: str) -> Decision | None:
        return self._decisions.get(str(trade_id))

    def closed(self) -> list[Decision]:
        return [d for d in self.all() if d.outcome is not None]

    def __len__(self) -> int:
        return len(self._decisions)


# ---------------------------------------------------------------------------
# Reading the log back
# ---------------------------------------------------------------------------


def value_at(decision: Decision, path: str) -> Any:
    """Pull a dotted field out of a record, e.g. ``fit.samples``.

    ``factors.momentum.points`` is understood too, since factors are a list
    rather than a mapping in the stored shape.
    """
    parts = path.split(".")
    if parts[0] == "factors" and len(parts) >= 2:
        factor = decision.factor(parts[1])
        node: Any = factor
        parts = parts[2:]
    else:
        node = getattr(decision, parts[0], None)
        parts = parts[1:]
    for part in parts:
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def group_by(
    decisions: Sequence[Decision], path: str, buckets: int = 3
) -> list[tuple[str, list[Decision]]]:
    """Split closed decisions into equal-count buckets of some parameter.

    This is the question the log exists to answer — "do trades taken when this
    number was high do any better" — and equal-count buckets rather than fixed
    thresholds keep each group large enough to mean something.
    """
    scored = [(value_at(d, path), d) for d in decisions if d.outcome is not None]
    usable = [(v, d) for v, d in scored if isinstance(v, (int, float))]
    if len(usable) < buckets:
        return []
    usable.sort(key=lambda pair: pair[0])

    out: list[tuple[str, list[Decision]]] = []
    size = len(usable) / buckets
    for i in range(buckets):
        chunk = usable[int(i * size) : int((i + 1) * size)]
        if not chunk:
            continue
        out.append((f"{chunk[0][0]:.3g} to {chunk[-1][0]:.3g}", [d for _, d in chunk]))
    return out


def outcome_summary(decisions: Sequence[Decision]) -> dict[str, Any]:
    """Win rate, profit factor and net for a set of closed decisions."""
    nets = [d.outcome["net_points"] for d in decisions if d.outcome is not None]
    if not nets:
        return {"trades": 0, "wins": 0, "win_rate": None, "profit_factor": None,
                "net_points": 0.0, "avg_points": None}
    wins = [n for n in nets if n > 0]
    losses = [-n for n in nets if n < 0]
    gross_loss = sum(losses)
    return {
        "trades": len(nets),
        "wins": len(wins),
        "win_rate": len(wins) / len(nets),
        "profit_factor": (sum(wins) / gross_loss) if gross_loss > 0 else None,
        "net_points": round(sum(nets), 2),
        "avg_points": round(sum(nets) / len(nets), 3),
    }
