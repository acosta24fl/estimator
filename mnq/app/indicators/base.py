"""Indicator interface.

An indicator returns two kinds of output:

* **series** — time/value points the chart draws, and
* **stats** — scalar readouts for the metrics strip,

plus optional **markers** pinned to price bars.

Crucially it also declares *how* it should be drawn (:class:`RenderSpec`).  The
browser builds panes, series and colours from that declaration, so a new
indicator needs no JavaScript changes at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..config import Settings
from ..core.features import ema  # re-exported: indicators import it from here
from ..core.timeframes import SessionBucket, Timeframe
from ..models import Bar

__all__ = ["ema"]  # plus the names defined below

# Rendering targets for a RenderSpec.
PANE_PRICE = "price"  # overlay on the candlestick pane
PANE_OWN = "own"  # a dedicated pane underneath


@dataclass(frozen=True)
class SeriesSpec:
    """One drawable line/histogram belonging to an indicator."""

    key: str
    label: str
    type: str = "line"  # line | histogram
    color: str = "#4c9aff"
    line_width: int = 2
    line_style: int = 0  # 0 solid, 1 dotted, 2 dashed
    up_color: str = ""  # histogram only: colour for values >= 0
    down_color: str = ""  # histogram only: colour for values < 0
    visible_in_legend: bool = True
    #: Whether this series may stretch its pane's price scale.  Set False for
    #: reference levels that can sit far from price (they would otherwise
    #: squash the candles to fit an off-screen level).
    autoscale: bool = True
    #: Draw a horizontal line with a price label at this series' last value.
    #: Useful for a target the viewer should be able to read off the axis.
    price_line: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "type": self.type,
            "color": self.color,
            "line_width": self.line_width,
            "line_style": self.line_style,
            "up_color": self.up_color,
            "down_color": self.down_color,
            "visible_in_legend": self.visible_in_legend,
            "autoscale": self.autoscale,
            "price_line": self.price_line,
        }


@dataclass(frozen=True)
class RenderSpec:
    pane: str = PANE_PRICE
    height: int = 130
    precision: int = 2
    zero_line: bool = False
    series: tuple[SeriesSpec, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "pane": self.pane,
            "height": self.height,
            "precision": self.precision,
            "zero_line": self.zero_line,
            "series": [s.as_dict() for s in self.series],
        }


@dataclass
class Stat:
    """A scalar readout rendered in the metrics strip."""

    key: str
    label: str
    value: Any
    unit: str = ""  # "pts", "%", "x", ...
    tone: str = "neutral"  # neutral | up | down | warn
    precision: int = 2
    hint: str = ""
    #: Show an explicit "+" on positive values.  Only meaningful for
    #: quantities that can genuinely go negative (a distance, not a range).
    signed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "value": self.value,
            "unit": self.unit,
            "tone": self.tone,
            "precision": self.precision,
            "hint": self.hint,
            "signed": self.signed,
        }


@dataclass
class Marker:
    time: int
    position: str = "aboveBar"  # aboveBar | belowBar | inBar
    shape: str = "circle"  # circle | square | arrowUp | arrowDown
    color: str = "#8892a6"
    text: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "position": self.position,
            "shape": self.shape,
            "color": self.color,
            "text": self.text,
        }


@dataclass
class IndicatorContext:
    """Everything an indicator is allowed to look at."""

    timeframe: Timeframe
    bars: Sequence[Bar]  # bars of the selected timeframe
    minute_bars: Sequence[Bar]  # the 1-minute base series
    daily_bars: Sequence[Bar]  # daily series (feed history + live session)
    session: SessionBucket
    settings: Settings
    now: float
    #: 5-minute series and the running projection log, for indicators that
    #: reason about the forecast. None when forecasting is unavailable.
    bars_5m: Sequence[Bar] = ()
    predictions: object | None = None
    #: Live paper trader, for indicators that draw simulated trades.
    paper: object | None = None


@dataclass
class IndicatorResult:
    series: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    markers: list[Marker] = field(default_factory=list)
    stats: list[Stat] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "series": self.series,
            "markers": [m.as_dict() for m in self.markers],
            "stats": [s.as_dict() for s in self.stats],
        }


class Indicator(ABC):
    #: Stable identifier; also the key used in the API payload.
    key: str = ""
    #: Display name.
    name: str = ""
    #: Sort order in the UI (lower first).
    order: int = 100
    #: Short description surfaced in the UI.
    description: str = ""
    #: Tunables echoed to the client; override per-instance if needed.
    params: dict[str, Any] = {}

    @property
    @abstractmethod
    def render(self) -> RenderSpec:
        """How the client should draw this indicator."""

    @abstractmethod
    def compute(self, ctx: IndicatorContext) -> IndicatorResult:
        """Produce series/stats/markers for the given context."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "order": self.order,
            "description": self.description,
            "params": dict(self.params),
            "render": self.render.as_dict(),
        }


# ---------------------------------------------------------------------------
# Small numeric helpers shared by indicators.
# ---------------------------------------------------------------------------


def points(value: float | None) -> float | None:
    return None if value is None else round(value, 2)
