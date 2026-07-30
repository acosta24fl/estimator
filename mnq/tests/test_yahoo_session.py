"""Cookie/crumb bootstrap and 429 handling.

Yahoo is unreachable from CI, so these drive the real client through an
httpx MockTransport — the request/response contract is exercised for real, only
the network is faked.
"""

import asyncio
import time

import httpx
import pytest

from app.config import load_settings
from app.feed.yahoo import YahooError, YahooFeed, YahooRateLimited, _backoff_delay


def chart_body(now=None):
    now = now or int(time.time()) // 60 * 60
    return {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {"regularMarketPrice": 20_000.0, "currency": "USD"},
                    "timestamp": [now - 120, now - 60],
                    "indicators": {
                        "quote": [
                            {
                                "open": [1.0, 2.0],
                                "high": [1.0, 2.0],
                                "low": [1.0, 2.0],
                                "close": [1.0, 2.0],
                                "volume": [1, 1],
                            }
                        ]
                    },
                }
            ],
        }
    }


class Recorder:
    """Scripted Yahoo stand-in that records every request it receives."""

    def __init__(self, chart_statuses=None, crumb="abc123", cookie=True):
        self.requests: list[httpx.Request] = []
        self.chart_statuses = list(chart_statuses or [])
        self.crumb = crumb
        self.cookie = cookie

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        host, path = request.url.host, request.url.path

        if host in ("fc.yahoo.com", "finance.yahoo.com"):
            headers = {"Set-Cookie": "A1=token; Path=/; Domain=.yahoo.com"} if self.cookie else {}
            return httpx.Response(404, headers=headers, text="")

        if path == "/v1/test/getcrumb":
            return httpx.Response(200, text=self.crumb)

        status = self.chart_statuses.pop(0) if self.chart_statuses else 200
        if status == 429:
            return httpx.Response(429, headers={"Retry-After": "1"}, json={})
        if status != 200:
            return httpx.Response(status, json={})
        return httpx.Response(200, json=chart_body())

    def paths(self, needle):
        return [r for r in self.requests if needle in str(r.url)]


def feed_with(recorder) -> YahooFeed:
    feed = YahooFeed(load_settings())
    feed._client = httpx.AsyncClient(
        transport=httpx.MockTransport(recorder),
        headers=dict(feed._client.headers),
        follow_redirects=True,
    )
    return feed


class TestSessionBootstrap:
    def test_cookies_are_collected_before_the_first_chart_call(self):
        rec = Recorder()
        feed = feed_with(rec)
        asyncio.run(feed.fetch_intraday())

        order = [str(r.url) for r in rec.requests]
        assert "fc.yahoo.com" in order[0], "cookies must be fetched first"
        assert any("getcrumb" in u for u in order)
        assert any("/v8/finance/chart/" in u for u in order)

    def test_crumb_is_sent_with_the_chart_request(self):
        rec = Recorder(crumb="Xy7Qz")
        feed = feed_with(rec)
        asyncio.run(feed.fetch_intraday())
        chart = rec.paths("/v8/finance/chart/")[-1]
        assert chart.url.params.get("crumb") == "Xy7Qz"

    def test_session_is_reused_across_polls(self):
        rec = Recorder()
        feed = feed_with(rec)
        asyncio.run(feed.fetch_intraday())
        asyncio.run(feed.fetch_daily())
        # Bootstrap happens once, not per request.
        assert len(rec.paths("getcrumb")) == 1

    def test_an_html_error_page_is_not_used_as_a_crumb(self):
        rec = Recorder(crumb="<!DOCTYPE html><html>error</html>")
        feed = feed_with(rec)
        asyncio.run(feed.fetch_intraday())
        assert feed._crumb is None
        assert "crumb" not in str(rec.paths("/v8/finance/chart/")[-1].url)

    def test_missing_cookies_do_not_prevent_the_request(self):
        rec = Recorder(cookie=False)
        feed = feed_with(rec)
        snap = asyncio.run(feed.fetch_intraday())
        assert snap.bars

    def test_browser_headers_are_sent(self):
        rec = Recorder()
        feed = feed_with(rec)
        asyncio.run(feed.fetch_intraday())
        headers = rec.paths("/v8/finance/chart/")[-1].headers
        assert "Chrome" in headers["user-agent"]
        assert headers["referer"].startswith("https://finance.yahoo.com")


class TestRateLimiting:
    def test_a_429_is_retried_and_can_succeed(self, monkeypatch):
        monkeypatch.setattr("app.feed.yahoo.asyncio.sleep", _no_sleep)
        # Both hosts throttle on the first pass, then recover.
        rec = Recorder(chart_statuses=[429, 429, 200])
        feed = feed_with(rec)
        snap = asyncio.run(feed.fetch_intraday())
        assert snap.bars

    def test_the_session_is_rebuilt_after_a_429(self, monkeypatch):
        monkeypatch.setattr("app.feed.yahoo.asyncio.sleep", _no_sleep)
        rec = Recorder(chart_statuses=[429, 429, 200])
        feed = feed_with(rec)
        asyncio.run(feed.fetch_intraday())
        # Bootstrapped once at the start, then again after being throttled.
        assert len(rec.paths("getcrumb")) >= 2

    def test_persistent_429_raises_a_clear_error(self, monkeypatch):
        monkeypatch.setattr("app.feed.yahoo.asyncio.sleep", _no_sleep)
        rec = Recorder(chart_statuses=[429] * 12)
        feed = feed_with(rec)
        with pytest.raises(YahooRateLimited) as excinfo:
            asyncio.run(feed.fetch_intraday())
        assert "rate limiting" in str(excinfo.value)

    def test_a_500_is_reported_as_a_plain_error(self, monkeypatch):
        monkeypatch.setattr("app.feed.yahoo.asyncio.sleep", _no_sleep)
        rec = Recorder(chart_statuses=[500] * 12)
        feed = feed_with(rec)
        with pytest.raises(YahooError) as excinfo:
            asyncio.run(feed.fetch_intraday())
        assert not isinstance(excinfo.value, YahooRateLimited)

    def test_it_fails_over_to_the_second_host(self, monkeypatch):
        monkeypatch.setattr("app.feed.yahoo.asyncio.sleep", _no_sleep)
        rec = Recorder(chart_statuses=[500, 200])
        feed = feed_with(rec)
        assert asyncio.run(feed.fetch_intraday()).bars
        hosts = [r.url.host for r in rec.paths("/v8/finance/chart/")]
        assert hosts[0] == "query1.finance.yahoo.com"
        assert hosts[1] == "query2.finance.yahoo.com"


class TestBackoffDelay:
    def test_retry_after_is_honoured(self):
        err = YahooRateLimited("429", retry_after=7)
        assert _backoff_delay(0, err) == 7

    def test_retry_after_is_capped(self):
        err = YahooRateLimited("429", retry_after=9999)
        assert _backoff_delay(0, err) <= 20.0

    def test_delay_grows_with_attempts(self):
        first = min(_backoff_delay(0, None) for _ in range(50))
        later = max(_backoff_delay(2, None) for _ in range(50))
        assert later > first

    def test_delay_is_always_positive_and_capped(self):
        for attempt in range(6):
            delay = _backoff_delay(attempt, None)
            assert 0 < delay <= 20.0


async def _no_sleep(_seconds):
    return None
