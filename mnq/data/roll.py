"""Continuous futures series construction.

Splicing contracts together naively leaves a price discontinuity at every roll.
For NQ those gaps run from a few points to well over a hundred, four times a
year. Nothing downstream can tell the difference between a roll gap and a real
move, so a naive splice quietly corrupts returns, ATR, and any label whose
barrier the gap happens to cross.

Two decisions produce a usable series:

**When to roll.** Volume migrates from the front contract to the next one over
a few days before expiry. Rolling on observed volume dominance (the default)
tracks where the liquidity actually is. Rolling on a fixed calendar offset is
cruder but works when volume data is missing.

**How to adjust.** Back-adjustment removes the gap by shifting all history
before the roll:

``ratio``
    Multiply older bars so percentage returns are continuous. Correct for
    anything return- or volatility-relative, which is most of this system's
    feature set. Historical prices no longer match what printed.
``difference``
    Subtract the gap so point differences are continuous. Keeps point moves
    honest locally, but over a long history the accumulated offset can drive
    early prices to zero or below.
``none``
    Raw splice. Useful only for inspecting the damage.

Ratio is the default. A caveat that matters for this system: with ratio
adjustment a fixed point threshold is not comparable across decades - NQ traded
near 1,500 in 2003 and above 20,000 in 2025, so ``labels.min_target_points =
20`` means 1.3% then and 0.1% now. Over long histories express targets in ATR
or percentage terms. :func:`check_point_scale` flags this.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..timeutil import to_utc_ns
from .contracts import Contract, parse_contract

log = logging.getLogger(__name__)

OHLC = ["open", "high", "low", "close"]
PRICE_COLS = OHLC
ADJUSTMENTS = ("ratio", "difference", "none")
ROLL_METHODS = ("volume", "open_interest", "calendar")

#: CME equity index futures open at 18:00 ET and the session is dated to the
#: following day. Adding six hours to Eastern time and taking the date gives
#: the session a bar belongs to.
_SESSION_SHIFT = pd.Timedelta(hours=6)


@dataclass(frozen=True)
class RollEvent:
    """One transition from the expiring contract to its successor."""

    timestamp: pd.Timestamp
    from_contract: str
    to_contract: str
    old_close: float
    new_close: float

    @property
    def gap(self) -> float:
        """New minus old at the roll, in index points."""
        return self.new_close - self.old_close

    @property
    def ratio(self) -> float:
        return self.new_close / self.old_close if self.old_close else 1.0


@dataclass
class ContinuousSeries:
    """A back-adjusted price series plus the provenance to audit it."""

    bars: pd.DataFrame
    rolls: list[RollEvent] = field(default_factory=list)
    adjustment: str = "ratio"
    method: str = "volume"

    @property
    def roll_frame(self) -> pd.DataFrame:
        """The roll schedule as a table, for eyeballing gap sizes."""
        if not self.rolls:
            return pd.DataFrame(
                columns=["from", "to", "old_close", "new_close", "gap", "ratio"]
            )
        return pd.DataFrame(
            [
                {
                    "timestamp": r.timestamp,
                    "from": r.from_contract,
                    "to": r.to_contract,
                    "old_close": r.old_close,
                    "new_close": r.new_close,
                    "gap": r.gap,
                    "ratio": r.ratio,
                }
                for r in self.rolls
            ]
        ).set_index("timestamp")

    def summary(self) -> str:
        b = self.bars
        if b.empty:
            return "empty continuous series"
        gaps = [abs(r.gap) for r in self.rolls]
        return (
            f"{len(b):,} bars  {b.index[0]:%Y-%m-%d} .. {b.index[-1]:%Y-%m-%d}  "
            f"{len(self.rolls)} rolls  "
            f"median |gap| {np.median(gaps):.1f}pt  max {max(gaps):.1f}pt"
            if gaps
            else f"{len(b):,} bars  {b.index[0]:%Y-%m-%d} .. {b.index[-1]:%Y-%m-%d}  no rolls"
        )


def session_dates(index: pd.DatetimeIndex) -> pd.Series:
    """Map bar timestamps to CME session dates."""
    et = index.tz_convert("America/New_York")
    return pd.Series((et + _SESSION_SHIFT).date, index=index)


def _prepare(
    contract_bars: dict[str, pd.DataFrame], reference: pd.Timestamp | None
) -> list[tuple[Contract, pd.DataFrame]]:
    """Parse keys to contracts, validate frames, return expiry-ordered pairs."""
    pairs: list[tuple[Contract, pd.DataFrame]] = []
    for code, df in contract_bars.items():
        if df is None or df.empty:
            log.debug("skipping %s: no bars", code)
            continue
        contract = parse_contract(code, reference=reference)
        frame = df.copy()
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(f"{code}: index must be a DatetimeIndex")
        frame.index = to_utc_ns(frame.index)
        missing = [c for c in OHLC if c not in frame.columns]
        if missing:
            raise ValueError(f"{code}: missing columns {missing}")
        if "volume" not in frame.columns:
            frame["volume"] = 0.0
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        pairs.append((contract, frame))

    if not pairs:
        raise ValueError("no contracts with data")
    pairs.sort(key=lambda p: (p[0].expiry, p[0].code))
    return pairs


def _dominance_rolls(
    pairs: list[tuple[Contract, pd.DataFrame]],
    column: str,
    confirm_sessions: int,
) -> dict[str, pd.Timestamp]:
    """Session on which each successor takes over, by volume/OI dominance.

    Returns ``{successor_code: first_session_date}``. A roll is only accepted
    after the successor out-trades the front contract for ``confirm_sessions``
    consecutive sessions, which stops a single busy day from flipping the
    series back and forth.
    """
    daily: dict[str, pd.Series] = {}
    for contract, frame in pairs:
        if column not in frame.columns:
            raise ValueError(
                f"{contract.code}: roll method needs a {column!r} column"
            )
        by_session = frame[column].groupby(session_dates(frame.index)).sum()
        daily[contract.code] = by_session

    table = pd.DataFrame(daily).sort_index().fillna(0.0)

    # An all-zero column is what a missing one looks like after normalisation.
    # Left alone it would silently demote every roll to the expiry backstop and
    # produce a plausible-looking series built on no evidence at all.
    if float(table.to_numpy().sum()) <= 0.0:
        raise ValueError(
            f"{column!r} is zero or absent for every contract, so dominance "
            f"cannot be measured. Supply {column} data or use "
            "method='calendar'."
        )
    order = [c.code for c, _ in pairs]
    rolls: dict[str, pd.Timestamp] = {}

    sessions = list(table.index)
    active_idx = 0
    streak = 0
    for position, session in enumerate(sessions):
        if active_idx + 1 >= len(order):
            break
        front, nxt = order[active_idx], order[active_idx + 1]
        front_expiry = pairs[active_idx][0].expiry.date()

        if table.at[session, nxt] > table.at[session, front]:
            streak += 1
        else:
            streak = 0

        # Expiry is a hard backstop: never carry a contract past its last day
        # even if it somehow still out-trades its successor.
        if streak >= confirm_sessions or session >= front_expiry:
            # Effective the *following* session. The trigger is this session's
            # total volume, which is not known until it closes - switching the
            # series on the same session would build it from information that
            # did not exist at the time those bars printed.
            effective = (
                sessions[position + 1]
                if position + 1 < len(sessions)
                else session
            )
            rolls[nxt] = pd.Timestamp(effective)
            active_idx += 1
            streak = 0

    return rolls


def _calendar_rolls(
    pairs: list[tuple[Contract, pd.DataFrame]], offset_days: int
) -> dict[str, pd.Timestamp]:
    """Roll a fixed number of calendar days before each expiry."""
    rolls: dict[str, pd.Timestamp] = {}
    for idx in range(1, len(pairs)):
        prev_expiry = pairs[idx - 1][0].expiry
        rolls[pairs[idx][0].code] = prev_expiry - pd.Timedelta(days=offset_days)
    return rolls


def build_continuous(
    contract_bars: dict[str, pd.DataFrame],
    *,
    method: str = "volume",
    adjustment: str = "ratio",
    confirm_sessions: int = 2,
    calendar_offset_days: int = 5,
    reference: pd.Timestamp | None = None,
) -> ContinuousSeries:
    """Splice contract-level bars into one adjusted series.

    ``contract_bars`` maps contract codes (``"NQZ24"``) to OHLCV frames with a
    UTC DatetimeIndex. Bars outside a contract's active span are discarded, so
    passing overlapping full histories is expected and correct - the overlap is
    what makes an accurate gap measurement possible.
    """
    if method not in ROLL_METHODS:
        raise ValueError(f"method must be one of {ROLL_METHODS}, got {method!r}")
    if adjustment not in ADJUSTMENTS:
        raise ValueError(
            f"adjustment must be one of {ADJUSTMENTS}, got {adjustment!r}"
        )

    pairs = _prepare(contract_bars, reference)
    order = [c.code for c, _ in pairs]
    frames = {c.code: f for c, f in pairs}

    if len(pairs) == 1:
        only = pairs[0]
        bars = only[1].copy()
        bars["contract"] = only[0].code
        return ContinuousSeries(bars, [], adjustment, method)

    if method == "calendar":
        roll_at = _calendar_rolls(pairs, calendar_offset_days)
    else:
        column = "volume" if method == "volume" else "open_interest"
        roll_at = _dominance_rolls(pairs, column, confirm_sessions)

    # ---- carve each contract's active span -------------------------------
    spans: list[tuple[str, pd.Timestamp | None, pd.Timestamp | None]] = []
    starts: dict[str, pd.Timestamp | None] = {order[0]: None}
    for code in order[1:]:
        starts[code] = roll_at.get(code)

    active = [code for code in order if code == order[0] or starts.get(code) is not None]
    for i, code in enumerate(active):
        begin = starts[code]
        end = starts[active[i + 1]] if i + 1 < len(active) else None
        spans.append((code, begin, end))

    # ---- measure the gap at each boundary --------------------------------
    rolls: list[RollEvent] = []
    for i in range(len(spans) - 1):
        old_code, new_code = spans[i][0], spans[i + 1][0]
        boundary = spans[i + 1][1]
        old_f, new_f = frames[old_code], frames[new_code]

        # The last instant both contracts traded at or before the roll. Using
        # the same timestamp in both books is what makes the gap a pure
        # contango/backwardation difference rather than a price move.
        common = old_f.index.intersection(new_f.index)
        cutoff = _to_utc(boundary)
        usable = common[common <= cutoff] if cutoff is not None else common
        if len(usable) == 0:
            log.warning(
                "%s -> %s: no overlapping bars at the roll; leaving gap unadjusted",
                old_code, new_code,
            )
            continue
        at = usable[-1]
        rolls.append(
            RollEvent(
                timestamp=cutoff if cutoff is not None else at,
                from_contract=old_code,
                to_contract=new_code,
                old_close=float(old_f.at[at, "close"]),
                new_close=float(new_f.at[at, "close"]),
            )
        )

    # ---- cumulative back-adjustment, newest span backwards ---------------
    factors = [1.0] * len(spans)
    offsets = [0.0] * len(spans)
    by_target = {r.to_contract: r for r in rolls}
    for i in range(len(spans) - 2, -1, -1):
        event = by_target.get(spans[i + 1][0])
        ratio, gap = (event.ratio, event.gap) if event else (1.0, 0.0)
        factors[i] = factors[i + 1] * ratio
        offsets[i] = offsets[i + 1] + gap

    # ---- splice ----------------------------------------------------------
    pieces = []
    for i, (code, begin, end) in enumerate(spans):
        frame = frames[code]
        lo, hi = _to_utc(begin), _to_utc(end)
        sel = frame
        if lo is not None:
            sel = sel[sel.index >= lo]
        if hi is not None:
            sel = sel[sel.index < hi]
        if sel.empty:
            continue
        piece = sel.copy()
        if adjustment == "ratio":
            piece[PRICE_COLS] = piece[PRICE_COLS] * factors[i]
        elif adjustment == "difference":
            piece[PRICE_COLS] = piece[PRICE_COLS] + offsets[i]
        piece["contract"] = code
        pieces.append(piece)

    if not pieces:
        raise ValueError("roll schedule produced no bars; check the input frames")

    bars = pd.concat(pieces).sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    bars.index.name = "timestamp"

    if adjustment == "difference" and (bars[PRICE_COLS] <= 0).any().any():
        log.warning(
            "difference adjustment drove %d bars to zero or below; "
            "use adjustment='ratio' for long histories",
            int((bars[PRICE_COLS] <= 0).any(axis=1).sum()),
        )

    series = ContinuousSeries(bars, rolls, adjustment, method)
    log.info("continuous %s: %s", order[0][:2], series.summary())
    return series


def _to_utc(ts: pd.Timestamp | None) -> pd.Timestamp | None:
    if ts is None:
        return None
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def check_point_scale(
    bars: pd.DataFrame, min_target_points: float, tolerance: float = 4.0
) -> str | None:
    """Warn when a fixed point target is not comparable across the history.

    Returns a message if the target means wildly different things at the start
    and end of the series, otherwise ``None``. Over 20 years of NQ the ratio is
    roughly 13x, which makes ``min_target_points`` almost meaningless as a
    constant - it should be expressed in ATR or percentage terms instead.
    """
    if bars.empty or min_target_points <= 0:
        return None
    first, last = float(bars["close"].iloc[0]), float(bars["close"].iloc[-1])
    if first <= 0 or last <= 0:
        return None
    pct_first = 100.0 * min_target_points / first
    pct_last = 100.0 * min_target_points / last
    spread = max(pct_first, pct_last) / max(1e-9, min(pct_first, pct_last))
    if spread < tolerance:
        return None
    return (
        f"min_target_points={min_target_points:g} is {pct_first:.2f}% of price at "
        f"{bars.index[0]:%Y-%m-%d} but {pct_last:.2f}% at {bars.index[-1]:%Y-%m-%d} "
        f"({spread:.1f}x). Express targets in ATR or percent over this history."
    )
