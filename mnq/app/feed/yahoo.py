"""Yahoo Finance feed (free, unauthenticated chart endpoint).

Endpoint
    https://query1.finance.yahoo.com/v8/finance/chart/MNQ%3DF?range=5d&interval=1m

It returns parallel arrays — ``timestamp[]`` alongside
``indicators.quote[0].{open,high,low,close,volume}`` — where each timestamp is
the bar's *start* in UTC epoch seconds.  Gaps come back as ``null`` and are
skipped.  ``meta`` carries the live last price used for the header ticker.

Yahoo caps 1-minute history at roughly 7 days; deeper intraday history is built
up by this app's own append-only minute log as it runs.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from ..models import Bar, FeedSnapshot, Quote
from . import register
from .base import PriceFeed

log = logging.getLogger(__name__)

_HOSTS = ("https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com")

# Yahoo rejects requests without a browser-like User-Agent.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}


class YahooError(RuntimeError):
    pass


@register
class YahooFeed(PriceFeed):
    key = "yahoo"
    name = "Yahoo Finance"

    def __init__(self, settings) -> None:
        super().__init__(settings)
        self._client = httpx.AsyncClient(
            headers=_HEADERS,
            timeout=settings.request_timeout,
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- public API --------------------------------------------------------

    async def fetch_intraday(self) -> FeedSnapshot:
        payload = await self._chart(
            interval="1m", range_=self.settings.intraday_range
        )
        return self._parse(payload, bar_seconds=60)

    async def fetch_daily(self) -> FeedSnapshot:
        payload = await self._chart(interval="1d", range_=self.settings.daily_range)
        return self._parse(payload, bar_seconds=24 * 3600)

    # -- transport ---------------------------------------------------------

    async def _chart(self, interval: str, range_: str) -> dict[str, Any]:
        params = {
            "range": range_,
            "interval": interval,
            "includePrePost": "true",
            "events": "div,splits",
        }
        last_error: Exception | None = None
        for host in _HOSTS:
            url = f"{host}/v8/finance/chart/{self.settings.symbol}"
            try:
                resp = await self._client.get(url, params=params)
                resp.raise_for_status()
                body = resp.json()
            except Exception as exc:  # network, HTTP, or JSON failure
                last_error = exc
                log.warning("yahoo request failed on %s: %s", host, exc)
                continue

            chart = body.get("chart") or {}
            if chart.get("error"):
                raise YahooError(str(chart["error"]))
            results = chart.get("result") or []
            if not results:
                last_error = YahooError("empty chart result")
                continue
            return results[0]

        raise YahooError(f"all Yahoo hosts failed: {last_error}")

    # -- parsing -----------------------------------------------------------

    def _parse(self, result: dict[str, Any], bar_seconds: int) -> FeedSnapshot:
        timestamps = result.get("timestamp") or []
        quote_blocks = (result.get("indicators") or {}).get("quote") or [{}]
        q = quote_blocks[0] if quote_blocks else {}
        opens = q.get("open") or []
        highs = q.get("high") or []
        lows = q.get("low") or []
        closes = q.get("close") or []
        volumes = q.get("volume") or []

        now = time.time()
        bars: list[Bar] = []
        for i, ts in enumerate(timestamps):
            o = _at(opens, i)
            h = _at(highs, i)
            low = _at(lows, i)
            c = _at(closes, i)
            if c is None:
                continue  # no trade printed in this slot
            # Yahoo occasionally nulls a single leg; fall back to the close so
            # the bar stays usable instead of dropping it entirely.
            o = c if o is None else o
            h = max(x for x in (h, o, c) if x is not None)
            low = min(x for x in (low, o, c) if x is not None)
            ts = int(ts)
            bars.append(
                Bar(
                    ts=ts,
                    open=float(o),
                    high=float(h),
                    low=float(low),
                    close=float(c),
                    volume=float(_at(volumes, i) or 0.0),
                    count=1,
                    complete=(ts + bar_seconds) <= now,
                )
            )

        bars.sort(key=lambda b: b.ts)
        return FeedSnapshot(bars=bars, quote=self._parse_quote(result, bars))

    @staticmethod
    def _parse_quote(result: dict[str, Any], bars: list[Bar]) -> Quote | None:
        meta = result.get("meta") or {}
        price = meta.get("regularMarketPrice")
        if price is None:
            price = bars[-1].close if bars else None
        if price is None:
            return None
        return Quote(
            price=float(price),
            ts=int(meta.get("regularMarketTime") or (bars[-1].ts if bars else time.time())),
            prev_close=_float_or_none(
                meta.get("previousClose", meta.get("chartPreviousClose"))
            ),
            day_high=_float_or_none(meta.get("regularMarketDayHigh")),
            day_low=_float_or_none(meta.get("regularMarketDayLow")),
            currency=str(meta.get("currency") or "USD"),
            exchange=str(meta.get("fullExchangeName") or meta.get("exchangeName") or ""),
        )


def _at(seq: list, i: int):
    if i < len(seq):
        return seq[i]
    return None


def _float_or_none(value) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
