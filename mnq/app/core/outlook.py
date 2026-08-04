"""The dashboard's headline directional call.

Turns a projection into one of three states — bullish, bearish, or no call —
for the page to colour itself with.

Two deliberate constraints, because a full-page colour is a very strong claim:

* A projection smaller than a fraction of the typical move is **no call**.
  Noise-sized drift should not paint the screen green.
* Confidence is capped by *measured* skill. If the projection has not been
  beating a "price doesn't move" baseline out of sample, the call can never
  read as high confidence regardless of how large the projected move is.

The colour therefore reflects both what the model says and how much its own
track record justifies saying it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BULLISH = "bullish"
BEARISH = "bearish"
NEUTRAL = "neutral"

#: |projected move| / band half-width -> confidence tier.
_LOW = 0.10
_MEDIUM = 0.35
_HIGH = 0.70


@dataclass
class Outlook:
    direction: str = NEUTRAL
    horizon_minutes: int = 10
    expected_move: float = 0.0
    target: float = 0.0
    anchor: float = 0.0
    strength: float = 0.0  # projected move as a fraction of typical move
    confidence: str = "none"  # none | low | medium | high
    reason: str = ""
    skill: float | None = None  # measured out-of-sample skill score
    capped_by_skill: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "horizon_minutes": self.horizon_minutes,
            "expected_move": round(self.expected_move, 2),
            "target": round(self.target, 2),
            "anchor": round(self.anchor, 2),
            "strength": round(self.strength, 3),
            "confidence": self.confidence,
            "reason": self.reason,
            "skill": None if self.skill is None else round(self.skill, 4),
            "capped_by_skill": self.capped_by_skill,
        }


def build_outlook(
    forecast,
    horizon_minutes: int,
    skill: float | None = None,
    min_ratio: float = _LOW,
) -> Outlook:
    """Classify a forecast into a directional call."""
    if forecast is None or not forecast.valid:
        return Outlook(
            horizon_minutes=horizon_minutes,
            reason=(forecast.reason if forecast is not None else "no forecast"),
            skill=skill,
        )

    band = forecast.band_half
    move = forecast.expected_move
    # Measure the call against the size of a typical move, so the same
    # threshold means the same thing in quiet and busy markets.
    strength = (abs(move) / band) if band > 0 else 0.0

    base = Outlook(
        horizon_minutes=horizon_minutes,
        expected_move=move,
        target=forecast.target,
        anchor=forecast.anchor_price,
        strength=strength,
        skill=skill,
    )

    if strength < min_ratio:
        base.reason = (
            f"projected move {abs(move):.2f} pts is under {min_ratio:.0%} of the "
            f"typical {band:.2f} pt move — too small to call"
        )
        return base

    base.direction = BULLISH if move > 0 else BEARISH
    base.confidence = _tier(strength)

    # A model that is not beating the no-move baseline has not earned a
    # confident presentation, however large the number it produced.
    if skill is not None and skill <= 0 and base.confidence in ("medium", "high"):
        base.confidence = "low"
        base.capped_by_skill = True
        base.reason = (
            f"capped at low confidence: measured skill {skill * 100:+.1f}% "
            "is not beating the no-move baseline"
        )
    else:
        base.reason = (
            f"projected {move:+.2f} pts over {horizon_minutes} min "
            f"({strength:.0%} of a typical move)"
        )
    return base


def _tier(strength: float) -> str:
    if strength >= _HIGH:
        return "high"
    if strength >= _MEDIUM:
        return "medium"
    return "low"
