"""Long-term Parquet archive for contract-level and continuous history.

The live path keeps bars in CSV because it appends one row a minute and a human
occasionally opens the file. History is the opposite problem: tens of millions
of rows, written once, read repeatedly, and CSV loses the timezone and the
float precision on every round trip. Parquet keeps both, and reads an order of
magnitude faster.

Layout under the archive root::

    contracts/NQ/1m/NQZ24.parquet      per-contract raw bars
    continuous/NQ_1m_ratio.parquet     spliced, back-adjusted series
    continuous/NQ_1m_ratio.rolls.csv   the roll schedule, human-readable

Contract files are the source of truth: the continuous series is derived and
can always be rebuilt with different roll or adjustment settings.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .contracts import parse_contract
from .roll import ContinuousSeries, RollEvent

log = logging.getLogger(__name__)


def _require_parquet() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "reading and writing the archive needs pyarrow. "
            "Install it with: pip install pyarrow"
        ) from None


class Archive:
    """File-backed store of futures history."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    # ------------------------------------------------------------ contracts

    def contract_dir(self, product: str, interval: str) -> Path:
        return self.root / "contracts" / product.upper() / interval

    def write_contracts(
        self, product: str, interval: str, frames: dict[str, pd.DataFrame]
    ) -> int:
        """Persist per-contract bars, merging into anything already stored."""
        _require_parquet()
        target = self.contract_dir(product, interval)
        target.mkdir(parents=True, exist_ok=True)

        written = 0
        for code, frame in frames.items():
            if frame is None or frame.empty:
                continue
            path = target / f"{code.upper()}.parquet"
            if path.exists():
                frame = _merge(pd.read_parquet(path), frame)
            frame = frame.sort_index()
            frame.index.name = "timestamp"
            frame.to_parquet(path, index=True)
            written += len(frame)
        log.info(
            "archived %d contracts (%d bars) -> %s",
            len(frames), written, target,
        )
        return written

    def read_contracts(
        self,
        product: str,
        interval: str,
        codes: list[str] | None = None,
        reference: pd.Timestamp | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Load per-contract bars back, keyed by contract code."""
        _require_parquet()
        target = self.contract_dir(product, interval)
        if not target.is_dir():
            raise FileNotFoundError(
                f"no archived {product} {interval} contracts at {target}"
            )

        wanted = {c.upper() for c in codes} if codes else None
        out: dict[str, pd.DataFrame] = {}
        for path in sorted(target.glob("*.parquet")):
            code = path.stem.upper()
            if wanted and code not in wanted:
                continue
            try:
                parse_contract(code, reference=reference)
            except ValueError:
                log.warning("skipping %s: not a contract code", path.name)
                continue
            frame = pd.read_parquet(path)
            frame.index = pd.DatetimeIndex(frame.index).tz_convert("UTC")
            out[code] = frame.sort_index()

        if not out:
            raise FileNotFoundError(f"no matching contracts under {target}")
        return out

    def list_contracts(self, product: str, interval: str) -> list[str]:
        target = self.contract_dir(product, interval)
        if not target.is_dir():
            return []
        return sorted(p.stem.upper() for p in target.glob("*.parquet"))

    # ----------------------------------------------------------- continuous

    def continuous_path(
        self, product: str, interval: str, adjustment: str = "ratio"
    ) -> Path:
        return (
            self.root
            / "continuous"
            / f"{product.upper()}_{interval}_{adjustment}.parquet"
        )

    def write_continuous(
        self, product: str, interval: str, series: ContinuousSeries
    ) -> Path:
        """Persist a built series plus its roll schedule.

        The schedule is written as CSV alongside deliberately: it is short, and
        being able to open it and see the gap sizes is how you catch a bad roll
        before it costs you a month of training runs.
        """
        _require_parquet()
        path = self.continuous_path(product, interval, series.adjustment)
        path.parent.mkdir(parents=True, exist_ok=True)

        bars = series.bars.copy()
        bars.index.name = "timestamp"
        bars.to_parquet(path, index=True)

        rolls = series.roll_frame
        if not rolls.empty:
            rolls.to_csv(path.with_suffix(".rolls.csv"))

        log.info("wrote continuous series -> %s (%s)", path, series.summary())
        return path

    def read_continuous(
        self, product: str, interval: str, adjustment: str = "ratio"
    ) -> ContinuousSeries:
        _require_parquet()
        path = self.continuous_path(product, interval, adjustment)
        if not path.exists():
            raise FileNotFoundError(
                f"no continuous series at {path}; build it with `ingest` first"
            )
        bars = pd.read_parquet(path)
        bars.index = pd.DatetimeIndex(bars.index).tz_convert("UTC")
        bars = bars.sort_index()

        rolls: list[RollEvent] = []
        sidecar = path.with_suffix(".rolls.csv")
        if sidecar.exists():
            table = pd.read_csv(sidecar, index_col=0, parse_dates=True)
            for ts, row in table.iterrows():
                rolls.append(
                    RollEvent(
                        timestamp=pd.Timestamp(ts),
                        from_contract=str(row["from"]),
                        to_contract=str(row["to"]),
                        old_close=float(row["old_close"]),
                        new_close=float(row["new_close"]),
                    )
                )
        return ContinuousSeries(bars, rolls, adjustment=adjustment)

    # ---------------------------------------------------------------- misc

    def describe(self) -> pd.DataFrame:
        """What is in the archive, for the status command."""
        rows = []
        contracts = self.root / "contracts"
        if contracts.is_dir():
            for product_dir in sorted(contracts.iterdir()):
                if not product_dir.is_dir():
                    continue
                for interval_dir in sorted(product_dir.iterdir()):
                    if not interval_dir.is_dir():
                        continue
                    files = sorted(interval_dir.glob("*.parquet"))
                    if not files:
                        continue
                    rows.append(
                        {
                            "kind": "contracts",
                            "product": product_dir.name,
                            "interval": interval_dir.name,
                            "items": len(files),
                            "first": files[0].stem,
                            "last": files[-1].stem,
                        }
                    )
        cont = self.root / "continuous"
        if cont.is_dir():
            for path in sorted(cont.glob("*.parquet")):
                parts = path.stem.split("_")
                rows.append(
                    {
                        "kind": "continuous",
                        "product": parts[0] if parts else path.stem,
                        "interval": parts[1] if len(parts) > 1 else "",
                        "items": 1,
                        "first": parts[2] if len(parts) > 2 else "",
                        "last": f"{path.stat().st_size / 1e6:.1f} MB",
                    }
                )
        return pd.DataFrame(rows)


def _merge(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return fresh
    if fresh.empty:
        return existing
    combined = pd.concat([existing, fresh])
    return combined[~combined.index.duplicated(keep="last")].sort_index()
