"""Vendor data ingestion.

Every historical futures vendor ships a different CSV. Databento gives you
nanosecond epochs, fixed-point prices with nine implied decimals, and a ``symbol`` column
carrying the contract code. FirstRate gives you Eastern-time strings, decimal
prices, and the contract encoded in the filename. Neither is wrong; they just
need normalising to one shape before anything downstream can use them.

That shape is the same one :mod:`mnq.data.yahoo` already produces - a UTC
DatetimeIndex over ``open/high/low/close/volume`` - so contract-level history
flows into the existing feature builder without touching it.

Add a vendor by describing it, not by writing a parser::

    spec = VendorSpec(
        name="acme",
        timestamp_column="date_time",
        columns={"o": "open", "h": "high", "l": "low", "c": "close"},
        tz="America/Chicago",
    )
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import pandas as pd

from .contracts import parse_contract

log = logging.getLogger(__name__)

CANONICAL = ["open", "high", "low", "close", "volume"]
OPTIONAL = ["open_interest"]


@dataclass(frozen=True)
class VendorSpec:
    """How to read one vendor's files."""

    name: str
    timestamp_column: str = "timestamp"
    #: Source column name -> canonical name. Columns already named canonically
    #: need no entry.
    columns: Mapping[str, str] = field(default_factory=dict)
    #: Timezone of naive timestamps. Ignored when the column is epoch-based.
    tz: str = "UTC"
    #: ``"ns"``/``"us"``/``"ms"``/``"s"`` when timestamps are integer epochs.
    timestamp_unit: str | None = None
    #: Divide price columns by this. Databento ships fixed-point integers
    #: with nine implied decimals, so its divisor is 1e9. Division keeps
    #: the result exact; multiplying by 1e-9 does not.
    price_divisor: float = 1.0
    #: Column holding the contract code, when one file covers many contracts.
    symbol_column: str | None = None
    #: Regex with a ``contract`` group, when the code is in the filename.
    filename_pattern: str | None = None


#: Databento CSV as downloaded raw (integer epochs, fixed-point prices).
DATABENTO = VendorSpec(
    name="databento",
    timestamp_column="ts_event",
    timestamp_unit="ns",
    price_divisor=1e9,
    symbol_column="symbol",
)

#: Databento exported with ``--pretty-px --pretty-ts``.
DATABENTO_PRETTY = VendorSpec(
    name="databento_pretty",
    timestamp_column="ts_event",
    symbol_column="symbol",
)

#: FirstRate Data per-contract files, e.g. ``NQ_2024Z_1min.csv``.
FIRSTRATE = VendorSpec(
    name="firstrate",
    timestamp_column="timestamp",
    tz="America/New_York",
    filename_pattern=r"(?P<contract>[A-Z]{1,4}[FGHJKMNQUVXZ]\d{2,4})",
)

#: Plain UTC CSV with canonical column names.
GENERIC = VendorSpec(name="generic")

SPECS: dict[str, VendorSpec] = {
    s.name: s
    for s in (DATABENTO, DATABENTO_PRETTY, FIRSTRATE, GENERIC)
}


def get_spec(name: str) -> VendorSpec:
    try:
        return SPECS[name]
    except KeyError:
        raise ValueError(
            f"unknown vendor {name!r}; known: {sorted(SPECS)}"
        ) from None


# ---------------------------------------------------------------- normalise


def normalise(df: pd.DataFrame, spec: VendorSpec) -> pd.DataFrame:
    """Coerce one vendor frame to the canonical UTC OHLCV shape."""
    if df.empty:
        return pd.DataFrame(columns=CANONICAL)

    out = df.rename(columns={**spec.columns})
    out.columns = [str(c).strip().lower() for c in out.columns]

    ts_col = spec.timestamp_column.lower()
    if ts_col not in out.columns:
        raise ValueError(
            f"{spec.name}: timestamp column {spec.timestamp_column!r} not found; "
            f"got {list(out.columns)}"
        )

    if spec.timestamp_unit:
        idx = pd.to_datetime(out[ts_col], unit=spec.timestamp_unit, utc=True)
    else:
        idx = pd.to_datetime(out[ts_col], utc=False, format="mixed")
        idx = pd.DatetimeIndex(idx)
        if idx.tz is None:
            # Ambiguous DST times are resolved to the first occurrence and
            # nonexistent ones shifted forward; both beat crashing on one bar.
            idx = idx.tz_localize(
                spec.tz, ambiguous="NaT", nonexistent="shift_forward"
            )
        idx = idx.tz_convert("UTC")

    out.index = pd.DatetimeIndex(idx, name="timestamp")
    out = out[out.index.notna()]

    missing = [c for c in CANONICAL if c not in out.columns]
    if missing:
        raise ValueError(f"{spec.name}: missing columns {missing}")

    keep = CANONICAL + [c for c in OPTIONAL if c in out.columns]
    out = out[keep].astype(float)

    if spec.price_divisor != 1.0:
        price_cols = [c for c in ("open", "high", "low", "close") if c in out]
        out[price_cols] = out[price_cols] / spec.price_divisor

    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]


def validate_bars(df: pd.DataFrame, name: str = "bars") -> list[str]:
    """Structural problems that would silently corrupt features.

    Returns human-readable issues rather than raising: a vendor file with three
    bad prints out of a million is worth logging and keeping, not discarding.
    """
    issues: list[str] = []
    if df.empty:
        return [f"{name}: empty"]

    if not df.index.is_monotonic_increasing:
        issues.append(f"{name}: index is not sorted")
    if df.index.has_duplicates:
        issues.append(f"{name}: {int(df.index.duplicated().sum())} duplicate timestamps")
    if getattr(df.index, "tz", None) is None:
        issues.append(f"{name}: index is timezone-naive")

    bad_range = df["high"] < df["low"]
    if bad_range.any():
        issues.append(f"{name}: {int(bad_range.sum())} bars with high < low")

    body_high = df[["open", "close"]].max(axis=1) > df["high"] + 1e-9
    body_low = df[["open", "close"]].min(axis=1) < df["low"] - 1e-9
    if body_high.any() or body_low.any():
        issues.append(
            f"{name}: {int(body_high.sum() + body_low.sum())} bars where "
            "open/close fall outside the high-low range"
        )

    if "volume" in df.columns and (df["volume"] < 0).any():
        issues.append(f"{name}: negative volume")

    return issues


# -------------------------------------------------------------------- load


def _contract_from_filename(path: Path, spec: VendorSpec) -> str | None:
    if not spec.filename_pattern:
        return None
    m = re.search(spec.filename_pattern, path.stem.upper())
    return m.group("contract") if m else None


def read_file(path: str | Path, spec: VendorSpec) -> dict[str, pd.DataFrame]:
    """Read one vendor file into ``{contract_code: bars}``.

    A file may hold many contracts (Databento) or exactly one (FirstRate); both
    return the same mapping so callers do not care which.
    """
    path = Path(path)
    raw = (
        pd.read_parquet(path)
        if path.suffix.lower() in (".parquet", ".pq")
        else pd.read_csv(path)
    )

    if spec.symbol_column and spec.symbol_column.lower() in [
        str(c).strip().lower() for c in raw.columns
    ]:
        col = spec.symbol_column.lower()
        raw.columns = [str(c).strip().lower() for c in raw.columns]
        out: dict[str, pd.DataFrame] = {}
        for code, group in raw.groupby(col):
            frame = normalise(group.drop(columns=[col]), spec)
            if not frame.empty:
                out[str(code).strip().upper()] = frame
        return out

    code = _contract_from_filename(path, spec)
    if code is None:
        raise ValueError(
            f"{path.name}: no symbol column and filename does not match "
            f"{spec.filename_pattern!r}; cannot tell which contract this is"
        )
    frame = normalise(raw, spec)
    return {code: frame} if not frame.empty else {}


def load_contract_bars(
    directory: str | Path,
    spec: VendorSpec | str,
    root: str | None = None,
    pattern: str = "*",
    reference: pd.Timestamp | None = None,
    strict: bool = False,
) -> dict[str, pd.DataFrame]:
    """Read every vendor file in a directory into contract-level frames.

    Files for the same contract are merged, so a vendor that ships one file per
    month per contract works without pre-concatenation. Pass ``root="NQ"`` to
    keep only that product when a directory holds several.
    """
    spec = get_spec(spec) if isinstance(spec, str) else spec
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")

    files = sorted(
        p
        for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in (".csv", ".txt", ".parquet", ".pq")
    )
    if not files:
        raise FileNotFoundError(f"no data files under {directory} matching {pattern!r}")

    merged: dict[str, pd.DataFrame] = {}
    for path in files:
        try:
            for code, frame in read_file(path, spec).items():
                merged[code] = (
                    frame
                    if code not in merged
                    else _merge(merged[code], frame)
                )
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the load
            if strict:
                raise
            log.warning("skipping %s: %s", path.name, exc)

    # Drop anything that is not a parseable contract of the requested root.
    keep: dict[str, pd.DataFrame] = {}
    for code, frame in merged.items():
        try:
            contract = parse_contract(code, reference=reference)
        except ValueError:
            log.warning("skipping %r: not a contract code", code)
            continue
        if root and contract.root.upper() != root.upper():
            continue
        keep[contract.code] = frame

    if not keep:
        raise ValueError(
            f"no {root or 'futures'} contracts found under {directory}"
        )

    for code, frame in sorted(keep.items()):
        for issue in validate_bars(frame, code):
            log.warning("%s", issue)

    log.info(
        "loaded %d contracts from %s (%s)",
        len(keep), directory, spec.name,
    )
    return keep


def _merge(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([a, b])
    return combined[~combined.index.duplicated(keep="last")].sort_index()
