"""End-to-end tests: real app, real engine, synthetic feed, temp data dir."""

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, load_settings
from app.core import timeframes
from app.main import create_app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    defaults = load_settings()
    settings = Settings(
        feed="synthetic",
        data_dir=tmp_path_factory.mktemp("data"),
        poll_seconds=3600,  # one startup poll is enough for the tests
        web_dir=defaults.web_dir,
    )
    with TestClient(create_app(settings)) as c:
        yield c


class TestConfig:
    def test_lists_every_timeframe(self, client):
        body = client.get("/api/config").json()
        assert [tf["key"] for tf in body["timeframes"]] == [
            "1m", "5m", "10m", "15m", "30m", "1h", "4h", "1d"
        ]

    def test_lists_indicators_with_render_specs(self, client):
        body = client.get("/api/config").json()
        keys = {i["key"] for i in body["indicators"]}
        assert {"vwap", "bar_size", "macd", "daily_structure"} <= keys
        for indicator in body["indicators"]:
            assert indicator["render"]["pane"] in ("price", "own")

    def test_default_timeframe_is_registered(self, client):
        body = client.get("/api/config").json()
        assert body["default_timeframe"] in {tf["key"] for tf in body["timeframes"]}


class TestSnapshot:
    @pytest.mark.parametrize("tf", ["1m", "5m", "10m", "15m", "30m", "1h", "4h", "1d"])
    def test_every_timeframe_returns_bars_and_indicators(self, client, tf):
        body = client.get(f"/api/snapshot?tf={tf}").json()
        assert body["timeframe"] == tf
        assert body["bars"], f"no bars for {tf}"
        for key in ("vwap", "bar_size", "macd", "daily_structure"):
            assert key in body["indicators"]
            assert "error" not in body["indicators"][key]

    def test_bars_are_ordered_and_well_formed(self, client):
        bars = client.get("/api/snapshot?tf=15m").json()["bars"]
        assert [b["time"] for b in bars] == sorted(b["time"] for b in bars)
        for bar in bars:
            assert bar["high"] >= max(bar["open"], bar["close"])
            assert bar["low"] <= min(bar["open"], bar["close"])

    def test_only_the_last_bar_may_be_incomplete(self, client):
        bars = client.get("/api/snapshot?tf=5m").json()["bars"]
        assert all(b["complete"] for b in bars[:-1])

    def test_higher_timeframe_has_fewer_bars(self, client):
        counts = {}
        for tf in ("1m", "5m", "30m"):
            counts[tf] = len(client.get(f"/api/snapshot?tf={tf}&limit=5000").json()["bars"])
        assert counts["1m"] > counts["5m"] > counts["30m"]

    def test_limit_is_respected(self, client):
        bars = client.get("/api/snapshot?tf=5m&limit=50").json()["bars"]
        assert len(bars) == 50

    def test_indicator_points_stay_inside_the_visible_window(self, client):
        body = client.get("/api/snapshot?tf=5m&limit=60").json()
        first = body["bars"][0]["time"]
        for result in body["indicators"].values():
            for points in result["series"].values():
                assert all(p["time"] >= first for p in points)

    def test_unknown_timeframe_is_a_404(self, client):
        assert client.get("/api/snapshot?tf=7m").status_code == 404

    def test_out_of_range_limit_is_rejected(self, client):
        assert client.get("/api/snapshot?tf=5m&limit=1").status_code == 422


class TestStatus:
    def test_reports_the_feed_and_counters(self, client):
        body = client.get("/api/status").json()
        assert body["feed"] == "synthetic"
        assert body["minute_bars"] > 0
        assert body["bars_logged"] > 0

    def test_health_is_ok(self, client):
        assert client.get("/api/health").json()["ok"] is True


class TestWebsocket:
    def test_pushes_a_snapshot_on_connect(self, client):
        with client.websocket_connect("/ws") as ws:
            message = ws.receive_json()
        assert message["type"] == "snapshot"
        assert message["data"]["bars"]

    def test_subscribe_switches_timeframe(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # initial push
            ws.send_json({"type": "subscribe", "tf": "1h"})
            message = ws.receive_json()
        assert message["data"]["timeframe"] == "1h"

    def test_unknown_timeframe_is_ignored(self, client):
        with client.websocket_connect("/ws") as ws:
            first = ws.receive_json()
            ws.send_json({"type": "subscribe", "tf": "bogus"})
            message = ws.receive_json()
        assert message["data"]["timeframe"] == first["data"]["timeframe"]


class TestStaticSite:
    def test_index_is_served(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "MNQ Live Dashboard" in resp.text

    def test_chart_library_is_served_when_vendored(self, client):
        # The library is fetched on first run rather than committed, so skip
        # rather than fail on a checkout where it has not been downloaded yet.
        from app.vendor import CHART_FILENAME, is_present

        if not is_present(load_settings().web_dir):
            pytest.skip("chart library not vendored yet; run `python -m app.vendor`")
        resp = client.get(f"/static/vendor/{CHART_FILENAME}")
        assert resp.status_code == 200
        assert len(resp.content) > 100_000

    def test_app_js_is_served(self, client):
        assert client.get("/static/app.js").status_code == 200
