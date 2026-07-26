"""Turning a model probability into an actual, sized trade instruction.

A probability alone is not tradeable. This module attaches the entry, the stop
and the target, then applies the gates that decide whether the setup is worth
firing at all - chiefly that the target must clear ``min_edge_points`` so the
20-250 point ambition is enforced at signal time rather than hoped for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..config import POINT_VALUE_USD, TICK_SIZE, LabelConfig, TradeConfig
from .manager import LONG, SHORT

DIRECTION_LABEL = {LONG: "LONG", SHORT: "SHORT"}


def round_to_tick(price: float) -> float:
    """Snap to MNQ's 0.25 grid; an unroundable price cannot be an order."""
    return round(price / TICK_SIZE) * TICK_SIZE


@dataclass
class Signal:
    """A complete trade instruction, ready to send or to execute."""

    timestamp: datetime
    direction: int
    entry: float
    stop: float
    target: float
    probability: float
    atr: float
    expected_points: float
    risk_points: float
    reward_risk: float
    components: dict[str, float] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def side(self) -> str:
        return DIRECTION_LABEL[self.direction]

    @property
    def risk_usd(self) -> float:
        return self.risk_points * POINT_VALUE_USD

    @property
    def reward_usd(self) -> float:
        return self.expected_points * POINT_VALUE_USD

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "side": self.side,
            "entry": self.entry,
            "stop": self.stop,
            "target": self.target,
            "probability": round(self.probability, 4),
            "expected_points": round(self.expected_points, 2),
            "risk_points": round(self.risk_points, 2),
            "reward_risk": round(self.reward_risk, 2),
            "atr": round(self.atr, 2),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "context": self.context,
        }


def build_signal(
    timestamp: datetime,
    direction: int,
    entry: float,
    atr: float,
    probability: float,
    lcfg: LabelConfig,
    tcfg: TradeConfig,
    components: dict[str, float] | None = None,
    context: dict[str, Any] | None = None,
) -> Signal | None:
    """Construct a signal, or return ``None`` if it fails a gate.

    Gates, in order: usable ATR, probability threshold, and a target large
    enough to be worth the risk and the costs.
    """
    if atr is None or atr <= 0 or atr != atr:  # NaN-safe
        return None
    if probability < tcfg.min_probability:
        return None

    reward = atr * lcfg.tp_atr_mult
    risk = atr * lcfg.sl_atr_mult

    # The move must be big enough to matter. Below this the spread, slippage and
    # commission eat a correct prediction.
    if reward < max(tcfg.min_edge_points, lcfg.min_target_points):
        return None
    if reward > lcfg.max_target_points:
        return None
    if risk <= 0:
        return None

    if direction == LONG:
        stop = entry - risk
        target = entry + reward
    else:
        stop = entry + risk
        target = entry - reward

    return Signal(
        timestamp=timestamp,
        direction=direction,
        entry=round_to_tick(entry),
        stop=round_to_tick(stop),
        target=round_to_tick(target),
        probability=float(probability),
        atr=float(atr),
        expected_points=float(reward),
        risk_points=float(risk),
        reward_risk=float(reward / risk),
        components=components or {},
        context=context or {},
    )


def select_direction(
    p_long: float, p_short: float, tcfg: TradeConfig
) -> tuple[int, float] | None:
    """Pick a side when both models fire.

    The two ensembles are trained independently, so they can both clear the
    threshold on the same bar. Taking the stronger one and requiring a clear
    margin avoids entering on bars where the models genuinely disagree - which
    is a statement of uncertainty, not of opportunity.
    """
    p_long = 0.0 if p_long != p_long else p_long
    p_short = 0.0 if p_short != p_short else p_short

    long_ok = p_long >= tcfg.min_probability
    short_ok = p_short >= tcfg.min_probability

    if not long_ok and not short_ok:
        return None
    if long_ok and short_ok:
        # Contradictory conviction: require a decisive gap before acting.
        if abs(p_long - p_short) < 0.05:
            return None
        return (LONG, p_long) if p_long > p_short else (SHORT, p_short)
    return (LONG, p_long) if long_ok else (SHORT, p_short)
