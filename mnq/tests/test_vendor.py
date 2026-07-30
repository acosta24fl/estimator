from pathlib import Path

import pytest

from app.vendor import (
    CHART_FILENAME,
    VendorError,
    ensure_chart_library,
    is_present,
    vendor_path,
)


class TestVendor:
    def test_path_is_under_web_vendor(self, tmp_path):
        assert vendor_path(tmp_path) == tmp_path / "vendor" / CHART_FILENAME

    def test_missing_file_is_not_present(self, tmp_path):
        assert not is_present(tmp_path)

    def test_a_truncated_file_is_not_accepted(self, tmp_path):
        path = vendor_path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("// oops, interrupted download")
        assert not is_present(tmp_path)

    def test_an_existing_file_is_left_alone(self, tmp_path):
        """No network call when the library is already vendored."""
        path = vendor_path(tmp_path)
        path.parent.mkdir(parents=True)
        body = "window.LightweightCharts={};" + "/*pad*/" * 10_000
        path.write_text(body)
        assert ensure_chart_library(tmp_path) == path
        assert path.read_text() == body  # untouched

    def test_download_failure_explains_the_manual_fix(self, tmp_path, monkeypatch):
        import httpx

        def boom(*args, **kwargs):
            raise httpx.ConnectError("no network")

        monkeypatch.setattr(httpx, "get", boom)
        with pytest.raises(VendorError) as excinfo:
            ensure_chart_library(tmp_path)
        assert "manually" in str(excinfo.value)
        assert CHART_FILENAME in str(excinfo.value)

    def test_a_bogus_download_is_rejected(self, tmp_path, monkeypatch):
        import httpx

        class FakeResponse:
            text = "<!doctype html><html>404 not found</html>"

            def raise_for_status(self):
                return None

        monkeypatch.setattr(httpx, "get", lambda *a, **k: FakeResponse())
        with pytest.raises(VendorError):
            ensure_chart_library(tmp_path)
        assert not vendor_path(tmp_path).exists()

    def test_successful_download_is_saved(self, tmp_path, monkeypatch):
        import httpx

        body = "window.LightweightCharts={version:()=>'5.2.0'};" + "x" * 60_000

        class FakeResponse:
            text = body

            def raise_for_status(self):
                return None

        monkeypatch.setattr(httpx, "get", lambda *a, **k: FakeResponse())
        path = ensure_chart_library(tmp_path)
        assert path.read_text() == body
        assert is_present(tmp_path)
