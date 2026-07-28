"""FastAPI application: TradingView webhook + 10-minute signal scheduler.

TradingView posts one alert per minute to ``/webhook``, reached from the public
internet through ngrok. A background task fires the signal evaluation every
``signal_interval_minutes``.

Security note: an ngrok URL is public and guessable enough to be found. The
shared secret is mandatory whenever one is configured, and the endpoint rejects
anything that does not carry it. Without that, anyone who finds the URL can
inject fake prices and drive the trade monitor.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..config import Config
from . import dashboard
from .engine import LiveEngine

log = logging.getLogger(__name__)

# Populated by create_app so route handlers can reach them.
_engine: LiveEngine | None = None
_config: Config | None = None
_autopilot: Any = None


class AlertPayload(BaseModel):
    """A TradingView alert.

    Every field except ``close`` is optional so a minimal alert
    (``{"close": {{close}}}``) still works; missing OHLC collapses to the close.
    """

    secret: str | None = None
    symbol: str | None = None
    time: str | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float = Field(..., description="last traded price (required)")
    volume: float | None = None
    action: str | None = None  # optional: "flatten" etc.


def _check_secret(cfg: Config, supplied: str | None) -> None:
    expected = cfg.server.webhook_secret
    if not expected:
        return
    # Constant-time compare: a naive == leaks the secret to a timing attack.
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="invalid or missing secret")


def _parse_time(raw: str | None) -> datetime:
    """TradingView's ``{{timenow}}`` is ISO-ish; fall back to arrival time."""
    if not raw:
        return datetime.now(timezone.utc)
    try:
        import pandas as pd

        ts = pd.Timestamp(raw)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return ts.to_pydatetime()
    except Exception:  # noqa: BLE001
        log.debug("unparseable alert time %r; using arrival time", raw)
        return datetime.now(timezone.utc)


def create_app(
    cfg: Config, engine: LiveEngine | None = None, autopilot: Any = None
) -> FastAPI:
    global _engine, _config, _autopilot
    _config = cfg
    _engine = engine or LiveEngine(cfg)
    _autopilot = autopilot

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(_signal_loop(cfg))
        log.info(
            "signal loop started (every %d min); webhook ready on %s:%d",
            cfg.server.signal_interval_minutes, cfg.server.host, cfg.server.port,
        )
        try:
            yield
        finally:
            task.cancel()
            if _engine:
                _engine.store.flush()
            log.info("shutdown complete")

    app = FastAPI(title="MNQ Signal Engine", version="1.0", lifespan=lifespan)

    # --------------------------------------------------------- dashboard
    # Cached because rebuilding the feature matrix on every poll would make the
    # page's own refresh the heaviest thing the process does.
    _cache: dict[str, Any] = {"at": None, "payload": None}

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_page() -> str:
        return dashboard.read_page()

    @app.get("/api/snapshot")
    async def snapshot(bars: int = dashboard.DEFAULT_BARS) -> dict[str, Any]:
        from ..models.projection import build_calibration, project

        now = datetime.now(timezone.utc)
        fresh = (
            _cache["at"] is not None
            and (now - _cache["at"]).total_seconds() < 10
            and _cache["payload"] is not None
        )
        if fresh:
            return _cache["payload"]

        matrix = score = None
        if _engine is not None:
            try:
                matrix = _engine.build_matrix()
                score = _engine.score_latest()
            except Exception as exc:  # noqa: BLE001 - the page must still render
                log.warning("snapshot: live scoring failed (%s)", exc)

        calibration = _calibration_cache(cfg, matrix)
        payload = dashboard.build_snapshot(
            cfg, matrix, score, project(score, calibration, cfg),
            engine_status=_engine.status() if _engine is not None else None,
            calibration=calibration, limit=bars,
            autopilot=_autopilot.status() if _autopilot is not None else None,
        )
        _cache["at"], _cache["payload"] = now, payload
        return payload

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}

    @app.get("/status")
    async def status() -> dict[str, Any]:
        assert _engine is not None
        return _engine.status()

    @app.post("/webhook")
    async def webhook(
        payload: AlertPayload,
        background: BackgroundTasks,
        x_webhook_secret: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Ingest a price alert and immediately re-check open trades."""
        assert _engine is not None and _config is not None
        _check_secret(_config, payload.secret or x_webhook_secret)

        if payload.action == "flatten":
            closed = _engine.flatten_all("webhook_flatten")
            return {"ok": True, "flattened": len(closed)}

        ts = _parse_time(payload.time)
        close = float(payload.close)
        has_full_bar = None not in (payload.open, payload.high, payload.low)

        if has_full_bar:
            result = _engine.on_minute_bar(
                ts,
                {
                    "open": float(payload.open),
                    "high": float(payload.high),
                    "low": float(payload.low),
                    "close": close,
                    "volume": float(payload.volume or 0.0),
                },
            )
        else:
            result = _engine.on_price(ts, close, float(payload.volume or 0.0))

        # Flushing is I/O; keep it off the request path so TradingView's alert
        # does not time out waiting on a disk write.
        background.add_task(_engine.store.flush)
        return {"ok": True, **result}

    @app.post("/webhook/raw")
    async def webhook_raw(
        request: Request,
        background: BackgroundTasks,
        x_webhook_secret: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Accept a plain-text alert body.

        TradingView cannot always be coaxed into valid JSON, so this tolerates
        ``LONG 21050.25`` or a bare number.
        """
        assert _engine is not None and _config is not None
        body = (await request.body()).decode("utf-8", errors="replace").strip()

        parsed: dict[str, Any] | None = None
        try:
            decoded = json.loads(body)
            # A bare number is valid JSON, so the result is not necessarily a
            # dict — "21123.75" decodes to a float.
            if isinstance(decoded, dict):
                parsed = decoded
            elif isinstance(decoded, (int, float)):
                parsed = {"close": float(decoded)}
        except json.JSONDecodeError:
            pass

        if parsed is None:
            # Pull the first number out of free text like "LONG 21050.25".
            for token in body.replace(",", " ").split():
                try:
                    parsed = {"close": float(token)}
                    break
                except ValueError:
                    continue

        if not parsed or "close" not in parsed:
            raise HTTPException(status_code=400, detail=f"no price found in body: {body[:80]!r}")

        _check_secret(_config, parsed.get("secret") or x_webhook_secret)
        result = _engine.on_price(_parse_time(parsed.get("time")), float(parsed["close"]))
        background.add_task(_engine.store.flush)
        return {"ok": True, **result}

    @app.post("/evaluate")
    async def evaluate(x_webhook_secret: str | None = Header(default=None)) -> dict[str, Any]:
        """Force an evaluation now, bypassing cooldown. For testing."""
        assert _engine is not None and _config is not None
        _check_secret(_config, x_webhook_secret)
        signal = _engine.evaluate(force=True)
        return {"signal": signal.to_dict() if signal else None}

    @app.post("/flatten")
    async def flatten(x_webhook_secret: str | None = Header(default=None)) -> dict[str, Any]:
        assert _engine is not None and _config is not None
        _check_secret(_config, x_webhook_secret)
        closed = _engine.flatten_all()
        return {"flattened": [t.to_dict() for t in closed]}

    return app


async def _signal_loop(cfg: Config) -> None:
    """Fire an evaluation on the configured cadence, aligned to the clock."""
    interval = max(1, cfg.server.signal_interval_minutes)
    while True:
        try:
            # Align to the wall clock so signals land at :00, :10, :20 ... which
            # makes them reproducible against a chart.
            now = datetime.now(timezone.utc)
            minutes_past = now.minute % interval
            wait = (interval - minutes_past) * 60 - now.second
            if wait <= 0:
                wait += interval * 60
            await asyncio.sleep(wait)

            if _engine is None:
                continue
            # evaluate() is synchronous and does model inference; a thread keeps
            # the event loop free to accept the next minute's webhook.
            await asyncio.to_thread(_engine.evaluate)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the loop must survive anything
            log.exception("signal loop iteration failed: %s", exc)
            await asyncio.sleep(30)


def run(cfg: Config) -> None:
    import uvicorn

    app = create_app(cfg)
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level="info")


# ------------------------------------------------------------- calibration

_CALIBRATION: dict[str, Any] = {"built_at": None, "value": None}


def _calibration_cache(cfg: Config, matrix):
    """Build the confidence-to-outcome table once, then reuse it.

    It depends on the walk-forward predictions rather than on live bars, so it
    only needs rebuilding when those change - which is on a `train` run, not on
    a dashboard poll.
    """
    from ..config import ARTIFACT_DIR
    from ..models.projection import Calibration, build_calibration

    path = ARTIFACT_DIR / "wf_predictions.csv"
    if not path.exists() or matrix is None or matrix.empty:
        return _CALIBRATION.get("value") or Calibration()

    stamp = path.stat().st_mtime
    if _CALIBRATION["built_at"] == stamp and _CALIBRATION["value"] is not None:
        return _CALIBRATION["value"]

    try:
        import pandas as pd

        preds = pd.read_csv(path, index_col=0, parse_dates=True)
        preds.index = pd.to_datetime(preds.index, utc=True)
        value = build_calibration(preds, matrix, cfg)
    except Exception as exc:  # noqa: BLE001 - the page renders without it
        log.warning("calibration unavailable (%s)", exc)
        value = Calibration()

    _CALIBRATION["built_at"], _CALIBRATION["value"] = stamp, value
    return value
