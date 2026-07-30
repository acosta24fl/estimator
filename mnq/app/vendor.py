"""First-run fetch of the charting library.

The dashboard serves the chart library from ``web/vendor/`` rather than a CDN,
so the page keeps working offline and is not at the mercy of a third-party host
at load time. The file itself is not kept in git — it is a 190 KB minified
build — so it is downloaded once on first run and reused from then on.

Run it manually with::

    python -m app.vendor

or drop the file in yourself; if it is already present nothing is downloaded.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

CHART_VERSION = "5.2.0"
CHART_FILENAME = "lightweight-charts.standalone.production.js"

#: The npm registry is tried first: it is the canonical source and tends to be
#: reachable from locked-down networks that block general CDN traffic. The CDN
#: is the fallback for the reverse case.
NPM_URL = (
    f"https://registry.npmjs.org/lightweight-charts/-/lightweight-charts-{CHART_VERSION}.tgz"
)
CDN_URL = (
    f"https://unpkg.com/lightweight-charts@{CHART_VERSION}/dist/{CHART_FILENAME}"
)

#: The API surface web/app.js relies on; also a sanity check on the download.
_EXPECTED_MARKER = "LightweightCharts"
_MIN_BYTES = 50_000


class VendorError(RuntimeError):
    pass


def vendor_path(web_dir: Path) -> Path:
    return Path(web_dir) / "vendor" / CHART_FILENAME


def is_present(web_dir: Path) -> bool:
    path = vendor_path(web_dir)
    return path.is_file() and path.stat().st_size >= _MIN_BYTES


def _fetch_from_npm() -> str:
    """Pull the package tarball and read the one file we need out of it."""
    import io
    import tarfile

    import httpx

    response = httpx.get(NPM_URL, timeout=90.0, follow_redirects=True)
    response.raise_for_status()
    member = f"package/dist/{CHART_FILENAME}"
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
        extracted = tar.extractfile(member)
        if extracted is None:
            raise VendorError(f"{member} missing from {NPM_URL}")
        return extracted.read().decode("utf-8")


def _fetch_from_cdn() -> str:
    import httpx

    response = httpx.get(CDN_URL, timeout=60.0, follow_redirects=True)
    response.raise_for_status()
    return response.text


def ensure_chart_library(web_dir: Path) -> Path:
    """Download the chart library if it is not already vendored."""
    path = vendor_path(web_dir)
    if is_present(web_dir):
        return path

    log.info("fetching charting library %s -> %s", CHART_VERSION, path)
    failures: list[str] = []
    for label, fetch in (("npm registry", _fetch_from_npm), ("unpkg CDN", _fetch_from_cdn)):
        try:
            body = fetch()
        except Exception as exc:
            log.warning("%s unavailable: %s", label, exc)
            failures.append(f"{label}: {exc}")
            continue

        if len(body) < _MIN_BYTES or _EXPECTED_MARKER not in body:
            failures.append(f"{label}: response was not the library ({len(body)} bytes)")
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        log.info("charting library saved from %s (%d KB)", label, len(body) // 1024)
        return path

    raise VendorError(
        "could not download the charting library.\n  "
        + "\n  ".join(failures)
        + f"\nFetch it manually and save it as {path}\n"
        f"  npm pack lightweight-charts@{CHART_VERSION}  # then copy dist/{CHART_FILENAME}"
    )


if __name__ == "__main__":  # pragma: no cover
    import sys

    from .config import load_settings

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        print(ensure_chart_library(load_settings().web_dir))
    except VendorError as exc:
        sys.exit(str(exc))
