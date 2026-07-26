"""Cross-asset context data.

MNQ does not trade in isolation. Nasdaq futures move with the rest of the index
complex, against bonds and the dollar, and with the market's appetite for risk.
That relationship carries information which pure price-derived indicators on MNQ
itself do not - and which is far less picked over than an EMA crossover.

Every symbol here comes from the same free Yahoo endpoint the main loader
already uses, so this adds capability without adding a credential, a paid feed,
or a rate limit to manage.

Robustness matters more than completeness. Yahoo silently changes and retires
tickers, and a symbol that works today may 404 next month. Every fetch is
therefore individually guarded: a failed symbol is logged and skipped, never
fatal. A partial basket still produces a usable feature set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .yahoo import OHLCV, download, load_cached, merge_cache, resample_ohlcv

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextSymbol:
    """One instrument in the context basket."""

    ticker: str
    name: str          # short, used as the feature suffix
    why: str           # why it is expected to carry information
    rth_only: bool = False   # ETFs stop trading overnight; futures roughly do not


# The basket. Futures are preferred over ETFs because they trade nearly 23
# hours, matching MNQ's session - an ETF's overnight staleness would otherwise
# masquerade as "no change" in the features.
DEFAULT_BASKET: tuple[ContextSymbol, ...] = (
    ContextSymbol("ES=F", "es", "S&P futures: broad market vs tech-heavy Nasdaq"),
    ContextSymbol("YM=F", "ym", "Dow futures: old-economy vs growth rotation"),
    ContextSymbol("RTY=F", "rty", "Russell futures: small-cap risk appetite"),
    ContextSymbol("ZN=F", "zn", "10Y note futures: the core risk-off bid"),
    ContextSymbol("GC=F", "gc", "Gold: safe-haven demand"),
    ContextSymbol("CL=F", "cl", "Crude: growth and inflation impulse"),
    ContextSymbol("6E=F", "eur", "Euro FX: the dollar leg of risk sentiment"),
    ContextSymbol("6J=F", "jpy", "Yen FX: carry unwind signal"),
    ContextSymbol("^VIX", "vix", "Implied volatility: the regime variable", rth_only=True),
    ContextSymbol("HYG", "hyg", "High-yield credit: risk appetite", rth_only=True),
    ContextSymbol("SOXX", "sox", "Semiconductors: historically lead Nasdaq", rth_only=True),
    ContextSymbol("RSP", "rsp", "Equal-weight S&P: a free market-breadth proxy", rth_only=True),
)


@dataclass
class ContextConfig:
    """Which context symbols to pull and how."""

    enabled: bool = True
    symbols: tuple[ContextSymbol, ...] = DEFAULT_BASKET
    interval: str = "1h"
    lookback: str = "730d"
    cache_dir: str = "artifacts/data/context"
    # A basket this small is useless if most of it fails; fail loudly instead of
    # silently training on two symbols.
    min_symbols: int = 4
    correlation_window: int = 60
    return_windows: tuple[int, ...] = (1, 4, 12, 24)
    vix_percentile_window: int = 500
    fetched: dict[str, str] = field(default_factory=dict)


def fetch_context(
    cfg: ContextConfig,
    cache_root: Path,
    tz: str = "UTC",
    refresh: bool = True,
    resample_to: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Download (or load) every symbol in the basket.

    Returns a mapping of short name -> OHLCV frame. Symbols that fail are
    omitted rather than raising, so one dead ticker cannot break a training run.
    """
    if not cfg.enabled:
        return {}

    cache_root.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}
    failures: list[tuple[str, str]] = []

    for sym in cfg.symbols:
        safe = sym.ticker.replace("=", "").replace("^", "").replace("-", "").replace(".", "")
        path = cache_root / f"{safe}_{cfg.interval}.csv"
        cached = load_cached(path, tz)

        frame = cached
        if refresh:
            try:
                fresh = download(sym.ticker, cfg.interval, cfg.lookback, tz)
                frame = merge_cache(cached, fresh)
                frame.to_csv(path)
            except Exception as exc:  # noqa: BLE001 - one bad ticker is survivable
                if cached.empty:
                    failures.append((sym.ticker, str(exc)[:90]))
                    log.warning("context symbol %s unavailable: %s", sym.ticker, str(exc)[:90])
                    continue
                log.warning("context %s refresh failed (%s); using cache", sym.ticker, str(exc)[:60])
                frame = cached

        if frame.empty:
            failures.append((sym.ticker, "no rows"))
            continue

        if resample_to:
            frame = resample_ohlcv(frame, resample_to)

        out[sym.name] = frame
        log.info("context %-10s %-6s %6d bars", sym.ticker, sym.name, len(frame))

    if len(out) < cfg.min_symbols:
        raise RuntimeError(
            f"only {len(out)} of {len(cfg.symbols)} context symbols loaded "
            f"(minimum {cfg.min_symbols}). Failures: {failures}. "
            "Check network access, or lower context.min_symbols to proceed "
            "with a reduced basket."
        )

    if failures:
        log.warning("context basket incomplete, continuing without: %s",
                    [t for t, _ in failures])
    return out


def available_names(frames: dict[str, pd.DataFrame]) -> list[str]:
    return sorted(frames)


def describe_batch_result(
    cfg: ContextConfig, frames: dict[str, pd.DataFrame]
) -> str:
    """Report which basket symbols loaded and which did not.

    Yahoo retires and renames tickers without warning, so a missing symbol is a
    routine event rather than a bug - but it should be visible, not silent.
    """
    lines = []
    for sym in cfg.symbols:
        df = frames.get(sym.name)
        if df is None or df.empty:
            lines.append(f"  {sym.ticker:<12} {'--':>8}   unavailable")
        else:
            lines.append(
                f"  {sym.ticker:<12} {len(df):>8,} bars   "
                f"{df.index.min():%Y-%m-%d} .. {df.index.max():%Y-%m-%d}"
            )
    got, total = len(frames), len(cfg.symbols)
    lines.append(f"\n  {got}/{total} symbols loaded.")
    if got < total:
        lines.append("  Missing symbols are skipped; the rest still produce features.")
    return "\n".join(lines)


def describe_basket(cfg: ContextConfig) -> str:
    """Human-readable basket listing, for the CLI."""
    lines = ["Context basket:"]
    for s in cfg.symbols:
        tag = " (RTH only)" if s.rth_only else ""
        lines.append(f"  {s.ticker:<12} {s.name:<5} {s.why}{tag}")
    return "\n".join(lines)
