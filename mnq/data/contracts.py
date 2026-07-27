"""Futures contract symbology.

A futures "symbol" is not one instrument. ``NQ`` is a series of quarterly
contracts - NQH24, NQM24, NQU24, NQZ24 - each with its own order book, its own
price, and a fixed expiry. Yahoo's ``NQ=F`` hides this by splicing the front
month together with no adjustment, which puts a fake gap of tens to hundreds of
points into the series four times a year. Those gaps are not returns, but every
momentum feature, every ATR, and every triple-barrier label treats them as if
they were.

This module is the vocabulary needed to avoid that: parsing contract codes,
knowing when each expires, and enumerating the ones that cover a date range.
:mod:`mnq.data.roll` uses it to build a properly adjusted continuous series.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass

import pandas as pd

# The CME month code alphabet. Note the gaps in the obvious mapping: I is
# skipped (reads as 1), and so are L, O, R, S, T, W, Y.
MONTH_CODES = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}
CODE_TO_MONTH = {v: k for k, v in MONTH_CODES.items()}

#: Equity index futures (ES, NQ, YM, RTY and their micros) trade a March
#: quarterly cycle. Energies and grains use other cycles; pass your own if you
#: ever extend beyond index products.
QUARTERLY_CYCLE = (3, 6, 9, 12)

#: Roots this system knows about, mapped to (cycle, dollars per index point).
#: MNQ and NQ track the same underlying at a 1:10 multiplier ratio, which is
#: why NQ's much longer history is usable as MNQ training data.
KNOWN_ROOTS = {
    "NQ": (QUARTERLY_CYCLE, 20.0),
    "MNQ": (QUARTERLY_CYCLE, 2.0),
    "ES": (QUARTERLY_CYCLE, 50.0),
    "MES": (QUARTERLY_CYCLE, 5.0),
    "YM": (QUARTERLY_CYCLE, 5.0),
    "MYM": (QUARTERLY_CYCLE, 0.5),
    "RTY": (QUARTERLY_CYCLE, 50.0),
    "M2K": (QUARTERLY_CYCLE, 5.0),
}

_CONTRACT_RE = re.compile(
    r"^(?P<root>[A-Z0-9]{1,4}?)(?P<code>[FGHJKMNQUVXZ])(?P<year>\d{1,4})$"
)


def third_friday(year: int, month: int) -> pd.Timestamp:
    """The third Friday of a month, as a tz-naive date at midnight.

    Equity index futures stop trading at the opening print on this day. When it
    falls on an exchange holiday the date moves back to Thursday - that is rare
    enough (and the roll happens well before expiry anyway) that this function
    does not model the holiday calendar. Do not use it for settlement pricing.
    """
    fridays = [
        day
        for week in calendar.monthcalendar(year, month)
        if (day := week[calendar.FRIDAY]) != 0
    ]
    return pd.Timestamp(year=year, month=month, day=fridays[2])


@dataclass(frozen=True, order=True)
class Contract:
    """One deliverable futures contract, e.g. NQZ24."""

    year: int
    month: int
    root: str = ""

    def __post_init__(self) -> None:
        if self.month not in MONTH_CODES:
            raise ValueError(f"month must be 1-12, got {self.month}")

    @property
    def month_code(self) -> str:
        return MONTH_CODES[self.month]

    @property
    def code(self) -> str:
        """Two-digit-year form, the most common in vendor data: ``NQZ24``."""
        return f"{self.root}{self.month_code}{self.year % 100:02d}"

    @property
    def code4(self) -> str:
        """Four-digit-year form, unambiguous: ``NQZ2024``."""
        return f"{self.root}{self.month_code}{self.year}"

    @property
    def expiry(self) -> pd.Timestamp:
        """Last trading day (tz-naive midnight)."""
        return third_friday(self.year, self.month)

    @property
    def multiplier(self) -> float:
        """Dollars per index point, or 1.0 for an unknown root."""
        known = KNOWN_ROOTS.get(self.root.upper())
        return known[1] if known else 1.0

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.code


def parse_contract(symbol: str, reference: pd.Timestamp | None = None) -> Contract:
    """Parse ``NQZ24``, ``NQZ2024`` or Databento's single-digit ``NQZ4``.

    A single-digit year is genuinely ambiguous - ``NQZ4`` is 2024 now and 2034
    in eight years - so it is resolved to the decade nearest ``reference``
    (default: today). Prefer two- or four-digit codes in stored data.

    The root is whatever precedes the month code and is *not* validated against
    :data:`KNOWN_ROOTS`, so that unlisted products (CL, GC, ZN) parse without a
    code change. The cost is that a typo which still fits the grammar - say
    ``NQZZ24`` - parses as root ``NQZ`` rather than raising. Filter on
    :data:`KNOWN_ROOTS` yourself when you need that guarantee; ``root=`` on
    :func:`mnq.data.vendor.load_contract_bars` does exactly this.
    """
    s = symbol.strip().upper()
    m = _CONTRACT_RE.match(s)
    if not m:
        raise ValueError(
            f"{symbol!r} is not a futures contract code; expected e.g. NQZ24"
        )

    root = m.group("root")
    month = CODE_TO_MONTH[m.group("code")]
    digits = m.group("year")
    ref = (
        pd.Timestamp(reference)
        if reference is not None
        else pd.Timestamp.now("UTC")
    )
    ref_year = int(ref.year)

    if len(digits) == 4:
        year = int(digits)
    elif len(digits) == 2:
        # Two digits: pivot on a 50-year window around the reference.
        century = (ref_year // 100) * 100
        year = century + int(digits)
        if year - ref_year > 50:
            year -= 100
        elif ref_year - year > 50:
            year += 100
    else:
        # One digit: pick the candidate decade closest to the reference.
        digit = int(digits)
        base = (ref_year // 10) * 10
        year = min(
            (base - 10 + digit, base + digit, base + 10 + digit),
            key=lambda y: abs(y - ref_year),
        )
    return Contract(year=year, month=month, root=root)


def generate_contracts(
    root: str,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    cycle: tuple[int, ...] | None = None,
) -> list[Contract]:
    """Every contract in ``cycle`` whose expiry falls in ``[start, end]``.

    The list is expiry-ordered, which is the order the roll logic walks.
    """
    root = root.upper()
    if cycle is None:
        known = KNOWN_ROOTS.get(root)
        cycle = known[0] if known else QUARTERLY_CYCLE

    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    if lo.tzinfo is not None:
        lo = lo.tz_localize(None)
    if hi.tzinfo is not None:
        hi = hi.tz_localize(None)

    out = [
        c
        for year in range(lo.year - 1, hi.year + 2)
        for month in cycle
        if lo <= (c := Contract(year=year, month=month, root=root)).expiry <= hi
    ]
    return sorted(out)


def front_contract(
    contracts: list[Contract], on: pd.Timestamp, offset_days: int = 0
) -> Contract | None:
    """The nearest contract not yet within ``offset_days`` of expiry.

    This is the *calendar* notion of the front month. Real volume usually
    migrates a few days earlier, which is what the volume-based roll in
    :mod:`mnq.data.roll` detects instead.
    """
    when = pd.Timestamp(on)
    if when.tzinfo is not None:
        when = when.tz_localize(None)
    cutoff = when + pd.Timedelta(days=offset_days)
    live = [c for c in sorted(contracts) if c.expiry >= cutoff]
    return live[0] if live else None
