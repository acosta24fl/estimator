"""Time-of-day structure: how big bars are, and how price behaves, by clock.

Answers the question "what does the 09:30 30-minute bar usually look like, and
what usually happens after it?" - for every slot in the session, at whatever
timeframe you ask for.

**Why this is worth building, when most pattern-hunting is not.** Intraday
volatility seasonality is one of the few genuinely robust effects in index
futures. The cash open, the European close, the settlement window and the last
half hour print systematically larger bars than 13:00, every year, in every
regime. That is a property of when market participants are present, not a
pattern mined out of noise, and it survives out of sample.

**Direction by clock is a different matter entirely and is treated as such.**
Testing thirteen slots for a directional edge will hand you one or two at
p<0.05 from pure chance. Every directional claim here is therefore:

* measured against the instrument's own base rate, not against 50%,
* tested with a binomial test, and
* corrected across all slots with Benjamini-Hochberg.

A slot is only reported as directional if it survives that correction. On real
index futures data, most of the time, none of them do - and the module says so
rather than presenting the raw p-values and letting the reader draw the
flattering conclusion.

**The size ratio is the core measurement.** Raw bar range in points is not
comparable across time: MNQ moved twice as much in 2022 as in 2024. So every
bar's range is divided by the median range of *its own day*:

    size_ratio = bar_range / median(all bar ranges that day)

A ratio of 2.4 at 09:30 means the opening bar is typically two and a half times
the size of a normal bar that day, whatever "normal" happened to be that year.
That is scale-free, regime-free, and directly usable: it says how wide a stop
has to be at that hour relative to any other.

**Stability is measured, not assumed.** History is split in half and the slot
profiles from each half are rank-correlated. A real clock effect reproduces
across halves; a fitted one does not. The number is reported next to the
profile so it can never be read without it.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Timeframes the profile can be built at, in minutes.
TIMEFRAMES: tuple[int, ...] = (5, 15, 30, 60)

#: A slot needs at least this many observations before it is reported. Below
#: it, the median is a description of a handful of days rather than a profile.
MIN_OBSERVATIONS = 30

#: Rank correlation between the two halves of history, below which the profile
#: is not called stable. 0.5 is deliberately demanding: a clock effect that
#: cannot reproduce its own shape across halves is not a clock effect.
STABILITY_FLOOR = 0.5

#: Benjamini-Hochberg false discovery rate for the directional tests.
FDR = 0.10

SESSION_TZ = "America/New_York"


@dataclass
class SlotProfile:
    """What one time-of-day slot looks like, and what follows it."""

    slot: str                     # "09:30"
    minutes: int                  # timeframe this was measured at
    n: int                        # observations

    # --- size: the robust half ---
    median_range_points: float
    median_size_ratio: float      # vs the median bar of the same day
    p25_size_ratio: float
    p75_size_ratio: float
    rank: int = 0                 # 1 = largest slot of the session

    # --- shape ---
    body_fraction: float = 0.0    # |close-open| / range; 1.0 = pure trend bar
    median_volume_ratio: float = 1.0

    # --- direction: the half that is usually noise ---
    p_up: float = 0.5
    base_rate_up: float = 0.5
    edge_vs_base: float = 0.0
    p_value: float = 1.0
    significant: bool = False     # after Benjamini-Hochberg across all slots

    # --- what happens next ---
    continuation_rate: float = 0.5      # next bar keeps this bar's direction
    median_next_size_ratio: float = 1.0  # does the expansion persist?

    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClockProfile:
    """The whole session, slot by slot, plus how much to trust it."""

    minutes: int
    tz: str
    slots: list[SlotProfile] = field(default_factory=list)
    days: int = 0
    stability: float | None = None     # rank correlation across history halves
    stable: bool = False
    base_rate_up: float = 0.5
    n_significant: int = 0
    note: str = ""

    @property
    def available(self) -> bool:
        return bool(self.slots)

    def to_dict(self) -> dict[str, Any]:
        return {
            "minutes": self.minutes,
            "tz": self.tz,
            "days": self.days,
            "stability": self.stability,
            "stable": self.stable,
            "base_rate_up": self.base_rate_up,
            "n_significant": self.n_significant,
            "note": self.note,
            "slots": [s.to_dict() for s in self.slots],
        }

    def slot(self, label: str) -> SlotProfile | None:
        for s in self.slots:
            if s.slot == label:
                return s
        return None


# ------------------------------------------------------------------ shaping


def to_timeframe(bars: pd.DataFrame, minutes: int, tz: str = SESSION_TZ) -> pd.DataFrame:
    """Resample to ``minutes`` and stamp each bar with its session clock slot.

    Resampling happens in the session timezone, not UTC. Bucketing 30-minute
    bars on a UTC boundary puts the cash open at 13:30 in the middle of a bar
    for half the year and at its edge for the other half, because the US
    changes clocks and the exchange session does not move with UTC. The whole
    measurement would then blur across the DST boundary.
    """
    if bars is None or bars.empty:
        return pd.DataFrame()

    local = bars.copy()
    if local.index.tz is None:
        local.index = local.index.tz_localize("UTC")
    local.index = local.index.tz_convert(tz)

    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in local.columns:
        agg["volume"] = "sum"

    out = local.resample(f"{minutes}min", label="left", closed="left").agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"])
    if out.empty:
        return out

    out["slot"] = out.index.strftime("%H:%M")
    out["day"] = out.index.normalize()
    out["range"] = out["high"] - out["low"]
    out["signed"] = out["close"] - out["open"]
    out["body"] = out["signed"].abs()
    return out


def add_size_ratio(frame: pd.DataFrame) -> pd.DataFrame:
    """Divide each bar's range by the median range of its own day.

    This is what makes 2022 and 2024 comparable. Without it the profile is
    dominated by which years happened to be volatile, and a "pattern" at 09:30
    is really a statement about 2022.
    """
    if frame.empty:
        return frame
    out = frame.copy()
    day_median = out.groupby("day", observed=True)["range"].transform("median")
    out["day_median_range"] = day_median
    out["size_ratio"] = out["range"] / day_median.replace(0.0, np.nan)
    if "volume" in out.columns:
        day_vol = out.groupby("day", observed=True)["volume"].transform("median")
        out["volume_ratio"] = out["volume"] / day_vol.replace(0.0, np.nan)
    return out


# ------------------------------------------------------------------- tests


def benjamini_hochberg(p_values: list[float], fdr: float = FDR) -> list[bool]:
    """Which p-values survive an FDR correction, in the input's order.

    Thirteen independent slots tested at p<0.05 yield roughly one false
    positive every time. Reporting those as discoveries is how a clock
    "pattern" gets believed. Benjamini-Hochberg controls the expected share of
    false positives among the ones reported, which is the right question when
    several slots are screened at once.
    """
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    keep = np.zeros(n, dtype=bool)
    threshold_rank = -1
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= fdr * rank / n:
            threshold_rank = rank
    if threshold_rank > 0:
        keep[order[:threshold_rank]] = True
    return [bool(v) for v in keep]


def _binomial_p(successes: int, n: int, rate: float) -> float:
    """Two-sided binomial test of ``successes`` against ``rate``."""
    if n <= 0:
        return 1.0
    try:
        from scipy import stats

        return float(stats.binomtest(successes, n, rate).pvalue)
    except Exception:  # noqa: BLE001 - scipy is optional at runtime
        # Normal approximation. Adequate at the sample sizes involved here,
        # and it keeps the module importable without scipy.
        sd = float(np.sqrt(n * rate * (1 - rate)))
        if sd <= 0:
            return 1.0
        z = abs(successes - n * rate) / sd
        return float(2.0 * (1.0 - _phi(z)))


def _phi(z: float) -> float:
    """Standard normal CDF, via the error function."""
    import math

    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ------------------------------------------------------------------- build


def build_clock_profile(
    bars: pd.DataFrame,
    minutes: int = 30,
    tz: str = SESSION_TZ,
    min_observations: int = MIN_OBSERVATIONS,
    fdr: float = FDR,
) -> ClockProfile:
    """Profile every time-of-day slot at one timeframe.

    ``bars`` is any OHLC frame finer than ``minutes`` - the live 1-minute store
    or a cached 5-minute series both work.
    """
    profile = ClockProfile(minutes=minutes, tz=tz)

    frame = add_size_ratio(to_timeframe(bars, minutes, tz))
    if frame.empty:
        profile.note = "no bars to profile"
        return profile

    profile.days = int(frame["day"].nunique())
    ups = frame["signed"] > 0
    profile.base_rate_up = float(ups.mean())

    # Next-bar behaviour, computed before grouping so it stays aligned.
    frame = frame.sort_index()
    frame["next_signed"] = frame["signed"].shift(-1)
    frame["next_size_ratio"] = frame["size_ratio"].shift(-1)
    frame["continued"] = np.sign(frame["signed"]) == np.sign(frame["next_signed"])

    rows: list[SlotProfile] = []
    p_values: list[float] = []

    for slot, group in frame.groupby("slot", observed=True):
        n = int(len(group))
        if n < min_observations:
            continue
        ratios = group["size_ratio"].dropna()
        if ratios.empty:
            continue

        up = int((group["signed"] > 0).sum())
        p_up = up / n
        p_value = _binomial_p(up, n, profile.base_rate_up)
        p_values.append(p_value)

        with np.errstate(invalid="ignore", divide="ignore"):
            body_frac = float(
                (group["body"] / group["range"].replace(0.0, np.nan)).median()
            )

        rows.append(
            SlotProfile(
                slot=str(slot),
                minutes=minutes,
                n=n,
                median_range_points=float(group["range"].median()),
                median_size_ratio=float(ratios.median()),
                p25_size_ratio=float(ratios.quantile(0.25)),
                p75_size_ratio=float(ratios.quantile(0.75)),
                body_fraction=0.0 if not np.isfinite(body_frac) else body_frac,
                median_volume_ratio=(
                    float(group["volume_ratio"].median())
                    if "volume_ratio" in group and group["volume_ratio"].notna().any()
                    else 1.0
                ),
                p_up=p_up,
                base_rate_up=profile.base_rate_up,
                edge_vs_base=p_up - profile.base_rate_up,
                p_value=p_value,
                continuation_rate=float(group["continued"].mean()),
                median_next_size_ratio=float(
                    group["next_size_ratio"].median()
                    if group["next_size_ratio"].notna().any() else 1.0
                ),
            )
        )

    if not rows:
        profile.note = (
            f"no slot reached {min_observations} observations at {minutes}m; "
            f"only {profile.days} day(s) of history"
        )
        return profile

    survives = benjamini_hochberg(p_values, fdr)
    for row, keep in zip(rows, survives):
        row.significant = keep

    rows.sort(key=lambda r: r.median_size_ratio, reverse=True)
    for rank, row in enumerate(rows, start=1):
        row.rank = rank
        row.note = _slot_note(row)

    rows.sort(key=lambda r: r.slot)
    profile.slots = rows
    profile.n_significant = sum(1 for r in rows if r.significant)
    profile.stability, profile.stable = _stability(frame, min_observations)
    profile.note = _profile_note(profile)
    return profile


def _slot_note(row: SlotProfile) -> str:
    size = (
        f"{row.median_size_ratio:.2f}x a normal bar that day "
        f"({row.median_range_points:.0f} pts typical)"
    )
    if row.significant:
        return (
            f"{size}; closed higher {row.p_up:.0%} of {row.n} times against a "
            f"{row.base_rate_up:.0%} base rate, and that gap survives "
            f"correction for testing every slot"
        )
    return (
        f"{size}; direction {row.p_up:.0%} up vs {row.base_rate_up:.0%} base — "
        f"not distinguishable from chance once every slot is accounted for"
    )


def _profile_note(profile: ClockProfile) -> str:
    biggest = max(profile.slots, key=lambda r: r.median_size_ratio)
    smallest = min(profile.slots, key=lambda r: r.median_size_ratio)
    parts = [
        f"{profile.days} sessions at {profile.minutes}m. Biggest bars at "
        f"{biggest.slot} ({biggest.median_size_ratio:.2f}x), quietest at "
        f"{smallest.slot} ({smallest.median_size_ratio:.2f}x)."
    ]
    if profile.stability is not None:
        parts.append(
            f"Shape reproduces across the two halves of history at "
            f"rho={profile.stability:.2f}"
            + (" — stable." if profile.stable else " — NOT stable, treat as descriptive.")
        )
    if profile.n_significant:
        parts.append(
            f"{profile.n_significant} slot(s) show a directional edge that "
            f"survives multiple-comparison correction."
        )
    else:
        parts.append(
            "No slot shows a directional edge that survives correction, which "
            "is the expected and honest result. Use this for sizing and stop "
            "distance, not for picking a side."
        )
    return " ".join(parts)


def _stability(frame: pd.DataFrame, min_observations: int) -> tuple[float | None, bool]:
    """Rank-correlate the slot size profile between the two halves of history.

    A genuine clock effect has the same shape in both halves. This is the
    single cheapest guard against reporting a profile that is really a
    statement about one unusual period.
    """
    days = np.sort(frame["day"].unique())
    if len(days) < 8:
        return None, False
    cut = days[len(days) // 2]

    def slot_medians(sub: pd.DataFrame) -> pd.Series:
        grouped = sub.groupby("slot", observed=True)["size_ratio"]
        counts = grouped.count()
        medians = grouped.median()
        return medians[counts >= max(5, min_observations // 4)]

    first = slot_medians(frame[frame["day"] < cut])
    second = slot_medians(frame[frame["day"] >= cut])
    shared = first.index.intersection(second.index)
    if len(shared) < 4:
        return None, False

    # Pearson, not Spearman. Most of the session is near-flat, so the ranks of
    # the quiet slots are decided by noise and a rank correlation reads low
    # even when the shape reproduces perfectly - a genuine 3x open next to
    # twelve slots all sitting at 1.0 scored rho=0.39. Correlating the ratios
    # themselves asks the question actually being posed: does the profile have
    # the same shape in both halves?
    rho = float(first[shared].corr(second[shared], method="pearson"))
    if not np.isfinite(rho):
        return None, False
    return rho, rho >= STABILITY_FLOOR


# ------------------------------------------------------------- slot history


def slot_history(
    bars: pd.DataFrame, slot: str, minutes: int = 30, tz: str = SESSION_TZ
) -> pd.DataFrame:
    """One slot's bar size, day by day - "09:30 across time".

    Returns a frame indexed by session date with the raw range, the day-relative
    size ratio, and the signed move. This is the series to look at when asking
    whether a slot's character is changing rather than what it is on average.
    """
    frame = add_size_ratio(to_timeframe(bars, minutes, tz))
    if frame.empty:
        return pd.DataFrame()
    sub = frame[frame["slot"] == slot]
    if sub.empty:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "range_points": sub["range"].to_numpy(float),
            "size_ratio": sub["size_ratio"].to_numpy(float),
            "signed_points": sub["signed"].to_numpy(float),
            "day_median_range": sub["day_median_range"].to_numpy(float),
        },
        index=pd.DatetimeIndex(sub["day"].to_numpy(), name="day"),
    )
    return out.sort_index()


def slot_trend(history: pd.DataFrame, window: int = 20) -> dict[str, Any]:
    """Is this slot's size drifting? Compares the recent window to the rest.

    Reported as a ratio of medians rather than a fitted slope: a slope on 20
    noisy observations is mostly a statement about the two endpoints.
    """
    if history is None or len(history) < window * 2:
        return {"available": False, "note": "not enough sessions to compare"}

    recent = history["size_ratio"].tail(window)
    earlier = history["size_ratio"].iloc[:-window]
    r_med, e_med = float(recent.median()), float(earlier.median())
    change = (r_med / e_med - 1.0) if e_med > 0 else 0.0

    if abs(change) < 0.15:
        verdict = "unchanged"
    else:
        verdict = "expanding" if change > 0 else "contracting"
    return {
        "available": True,
        "recent_median": r_med,
        "earlier_median": e_med,
        "change": change,
        "verdict": verdict,
        "n_recent": int(len(recent)),
        "n_earlier": int(len(earlier)),
        "note": (
            f"last {len(recent)} sessions run {r_med:.2f}x versus {e_med:.2f}x "
            f"before that ({change:+.0%}) — {verdict}"
        ),
    }


# ------------------------------------------------------------------ report


def format_report(profile: ClockProfile, top: int = 0) -> str:
    """Human-readable table, largest slots first."""
    w = 78
    lines = ["=" * w, f"  TIME-OF-DAY PROFILE — {profile.minutes}-MINUTE BARS", "=" * w]
    if not profile.available:
        lines += [f"  {profile.note}", "=" * w]
        return "\n".join(lines)

    lines.append(
        f"  {'slot':<7}{'n':>6}{'size':>8}{'range':>9}{'body':>7}"
        f"{'up%':>7}{'vs base':>9}{'cont%':>7}  {'':<4}"
    )
    lines.append("-" * w)

    ordered = sorted(profile.slots, key=lambda r: r.median_size_ratio, reverse=True)
    if top:
        ordered = ordered[:top]
    for r in ordered:
        mark = " *" if r.significant else ""
        lines.append(
            f"  {r.slot:<7}{r.n:>6}{r.median_size_ratio:>7.2f}x"
            f"{r.median_range_points:>9.0f}{r.body_fraction:>7.2f}"
            f"{100 * r.p_up:>6.1f}%{100 * r.edge_vs_base:>+8.1f}%"
            f"{100 * r.continuation_rate:>6.1f}%{mark}"
        )

    lines += ["-" * w]
    lines.append("  size    = bar range / the median bar of that same day")
    lines.append("  body    = |close-open| / range; 1.00 is a pure trend bar")
    lines.append("  cont%   = the next bar continued this bar's direction")
    lines.append("  *       = directional edge survives correction for testing")
    lines.append("            every slot. No star means treat direction as noise.")
    lines += ["=" * w]
    for line in _wrap(profile.note, w - 4):
        lines.append(f"  {line}")
    lines.append("=" * w)
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words, line, out = text.split(), "", []
    for word in words:
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out
