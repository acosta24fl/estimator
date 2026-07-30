#!/usr/bin/env python3
"""Start the local MNQ dashboard.

    python run.py                 # live Yahoo Finance data
    MNQ_FEED=synthetic python run.py   # offline demo data
"""

from __future__ import annotations

import logging
import sys

import uvicorn

from app.config import load_settings
from app.vendor import VendorError, ensure_chart_library


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    settings = load_settings()

    # Fetched once on first run, then reused; see app/vendor.py.
    try:
        ensure_chart_library(settings.web_dir)
    except VendorError as exc:
        sys.exit(f"\n  {exc}\n")

    print(f"\n  MNQ dashboard -> http://{settings.host}:{settings.port}\n")
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
