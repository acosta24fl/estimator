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

import asyncio
import logging
import random
import time
from typing import Any

import httpx

from ..models import Bar, FeedSnapshot, Quote
from . import register
from .base import PriceFeed

log = logging.getLogger(__name__)

_HOSTS = ("https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com")

# Pages visited purely to be handed session cookies. fc.yahoo.com answers 404
# but still sets them, which is the cheapest route; the finance page is the
# fallback for regions where that host misbehaves.
_COOKIE_URLS = ("https://fc.yahoo.com", "https://finance.yahoo.com")
_CRUMB_URL = "https://query1.finance.yahoo.com/v1/test/getcrumb"

# Yahoo rejects or throttles requests that do not look like a browser.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://finance.yahoo.com",
    "Referer": "https://finance.yahoo.com/",
    "Connection": "keep-alive",
}

_MAX_ATTEMPTS = 3
_BACKOFF_CAP = 20.0


class YahooError(RuntimeError):
    pass


class YahooRateLimited(YahooError):
    """Yahoo answered 429. Carries the server's Retry-After when given."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


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
        self._crumb: str | None = None
        self._has_session = False

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- session -----------------------------------------------------------

    async def _ensure_session(self, force: bool = False) -> None:
        """Pick up the cookies (and crumb) Yahoo expects from a browser.

        Since 2024 the chart endpoint throttles cookie-less clients with 429
        almost immediately. Visiting a Yahoo page first yields the session
        cookies; ``/v1/test/getcrumb`` then returns the token that goes with
        them. The crumb is optional for charts, so a failure there is not
        fatal — the cookies alone lift the throttling.
        """
        if self._has_session and not force:
            return

        self._crumb = None
        self._client.cookies.clear()

        for url in _COOKIE_URLS:
            try:
                await self._client.get(url)
            except Exception as exc:
                log.debug("cookie bootstrap via %s failed: %s", url, exc)
                continue
            if len(self._client.cookies):
                break

        if not len(self._client.cookies):
            log.warning("Yahoo returned no session cookies; requests may be throttled")

        try:
            resp = await self._client.get(_CRUMB_URL)
            crumb = resp.text.strip()
            # A valid crumb is a short opaque token, never an HTML error page.
            if resp.status_code == 200 and crumb and "<" not in crumb and len(crumb) < 40:
                self._crumb = crumb
        except Exception as exc:
            log.debug("crumb fetch failed: %s", exc)

        self._has_session = True
        log.info(
            "Yahoo session ready (%d cookies, crumb=%s)",
            len(self._client.cookies),
            "yes" if self._crumb else "no",
        )

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
        await self._ensure_session()

        params: dict[str, Any] = {
            "range": range_,
            "interval": interval,
            "includePrePost": "true",
            "events": "div,splits",
        }
        if self._crumb:
            params["crumb"] = self._crumb

        last_error: Exception | None = None
        throttled = False

        for attempt in range(_MAX_ATTEMPTS):
            for host in _HOSTS:
                url = f"{host}/v8/finance/chart/{self.settings.symbol}"
                try:
                    resp = await self._client.get(url, params=params)
                except Exception as exc:  # transport failure
                    last_error = exc
                    log.debug("yahoo request failed on %s: %s", host, exc)
                    continue

                if resp.status_code in (401, 403, 429):
                    throttled = resp.status_code == 429
                    last_error = YahooRateLimited(
                        f"{resp.status_code} from {host}", _retry_after(resp)
                    )
                    # A stale or missing session is the usual cause; rebuild it
                    # once and let the retry use the fresh cookies.
                    self._has_session = False
                    continue

                try:
                    resp.raise_for_status()
                    body = resp.json()
                except Exception as exc:
                    last_error = exc
                    log.debug("bad response from %s: %s", host, exc)
                    continue

                chart = body.get("chart") or {}
                if chart.get("error"):
                    raise YahooError(str(chart["error"]))
                results = chart.get("result") or []
                if not results:
                    last_error = YahooError("empty chart result")
                    continue
                return results[0]

            if attempt + 1 < _MAX_ATTEMPTS:
                delay = _backoff_delay(attempt, last_error)
                log.debug("retrying Yahoo in %.1fs (%s)", delay, last_error)
                await asyncio.sleep(delay)
                await self._ensure_session()
                if self._crumb:
                    params["crumb"] = self._crumb
                else:
                    params.pop("crumb", None)

        if throttled:
            raise YahooRateLimited(
                "Yahoo is rate limiting this client (429). It usually clears on "
                "its own; raise MNQ_POLL_SECONDS if it persists.",
                getattr(last_error, "retry_after", None),
            )
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


def _retry_after(resp) -> float | None:
    raw = resp.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None  # HTTP-date form; the default backoff covers it


def _backoff_delay(attempt: int, error: Exception | None) -> float:
    """Exponential backoff with jitter, honouring Retry-After when present."""
    retry_after = getattr(error, "retry_after", None)
    if retry_after:
        return min(float(retry_after), _BACKOFF_CAP)
    base = min(2.0 * (2**attempt), _BACKOFF_CAP)
    return base * (0.6 + random.random() * 0.4)


def _at(seq: list, i: int):
    if i < len(seq):
        return seq[i]
    return None


def _float_or_none(value) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
