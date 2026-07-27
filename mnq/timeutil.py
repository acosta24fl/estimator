"""One canonical timestamp representation.

Bars reach this system from several places - a live Yahoo download, a CSV
cache, a Parquet archive, a vendor file, the synthetic generator - and since
pandas 2.0 each of those can pick its own datetime *resolution*. A fresh
yfinance pull may hand back ``datetime64[s, UTC]`` while the CSV cache beside
it parses as ``datetime64[us, UTC]``.

The values are identical; only the unit differs. Most operations paper over
that, which is what makes it dangerous: ``merge_asof`` does not, and it is the
one operation this system relies on for leak-free higher-timeframe and
cross-asset joins. Mixing sources produced

    MergeError: incompatible merge keys [0] datetime64[us, UTC] and
                datetime64[s, UTC], must be the same type

on a real run, after every symbol had already downloaded - so the failure
appeared as "no training instruments could be loaded" rather than as anything
to do with time. It never reproduced in tests because synthetic frames and CSV
fixtures agree on a unit; only mixing a *live* download with a *cached* file
disagrees.

So every timestamp is coerced to UTC nanoseconds the moment it enters the
system, and again defensively at each merge. Nanoseconds because pandas below
2.0 supports nothing else, and because casting is cheap next to a run that
dies an hour in.
"""

from __future__ import annotations

import pandas as pd

#: The single resolution every index in this system uses.
UTC_NS = "datetime64[ns, UTC]"


def to_utc_ns(values) -> pd.DatetimeIndex:
    """Coerce anything date-like to a tz-aware UTC nanosecond index."""
    idx = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    return idx if str(idx.dtype) == UTC_NS else idx.astype(UTC_NS)


def normalise_index(df: pd.DataFrame, name: str = "timestamp") -> pd.DataFrame:
    """Put a frame's index on the canonical resolution, in place-ish.

    Returns the same object when it is already canonical, so this is safe to
    call on every load path without copying large frames needlessly.
    """
    if df is None or len(df) == 0:
        return df
    if str(df.index.dtype) != UTC_NS:
        df.index = to_utc_ns(df.index)
    df.index.name = name
    return df


def align_merge_keys(
    left: pd.DataFrame, left_on: str, right: pd.DataFrame, right_on: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Force two ``merge_asof`` key columns onto the same resolution.

    The boundary coercion should make this redundant. It is here anyway
    because the failure mode is a hard error an hour into a run, and the
    cost of being sure is one dtype comparison.
    """
    if str(left[left_on].dtype) != UTC_NS:
        left = left.assign(**{left_on: to_utc_ns(left[left_on])})
    if str(right[right_on].dtype) != UTC_NS:
        right = right.assign(**{right_on: to_utc_ns(right[right_on])})
    return left, right
