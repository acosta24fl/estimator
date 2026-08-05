"""HTTP + WebSocket surface.

The contract is deliberately small:

``GET  /api/config``    what exists — timeframes and indicator render specs
``GET  /api/snapshot``  everything needed to draw one timeframe
``GET  /api/status``    feed health and logging counters
``WS   /ws``            the same snapshot, pushed on every poll

Because ``/api/config`` describes indicators structurally, the front end never
hard-codes one.  New indicators show up on their own.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect

from ..core import timeframes
from ..indicators import all_indicators

log = logging.getLogger(__name__)
router = APIRouter()


def _engine(request: Request):
    return request.app.state.engine


@router.get("/api/config")
async def get_config(request: Request):
    engine = _engine(request)
    return {
        "symbol": engine.settings.symbol,
        "display_name": engine.settings.display_name,
        "poll_seconds": engine.settings.poll_seconds,
        "session": {
            "tz": engine.settings.session_tz,
            "open_hour": engine.settings.session_open_hour,
        },
        "timeframes": [tf.as_dict() for tf in timeframes.ordered()],
        "default_timeframe": "5m",
        "indicators": [i.as_dict() for i in all_indicators()],
    }


@router.get("/api/snapshot")
async def get_snapshot(
    request: Request,
    tf: str = Query("5m", description="Timeframe key, e.g. 5m"),
    limit: int | None = Query(None, ge=10, le=5000),
):
    try:
        return _engine(request).snapshot(tf, limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None


@router.get("/api/trades")
async def get_trades(request: Request, limit: int = Query(200, ge=1, le=5000)):
    """Simulated trade history, newest last, with the running tally."""
    engine = _engine(request)
    if not engine.settings.paper_trading:
        return {"enabled": False, "trades": [], "summary": None}
    price = engine.quote.price if engine.quote else None
    return {
        "enabled": True,
        "trades": [t.as_dict() for t in engine.paper.trades()[-limit:]],
        "summary": engine.paper.summary(price),
    }


@router.get("/api/status")
async def get_status(request: Request):
    return _engine(request).status()


@router.get("/api/health")
async def get_health(request: Request):
    engine = _engine(request)
    return {"ok": engine.last_error is None, "error": engine.last_error}


@router.websocket("/ws")
async def stream(websocket: WebSocket):
    await websocket.accept()
    engine = websocket.app.state.engine
    state = {"tf": "5m", "limit": None}
    lock = asyncio.Lock()

    async def push() -> None:
        async with lock:
            try:
                payload = engine.snapshot(state["tf"], state["limit"])
                await websocket.send_json({"type": "snapshot", "data": payload})
            except (WebSocketDisconnect, RuntimeError):
                pass
            except Exception:
                log.exception("failed to push snapshot")

    engine.add_listener(push)
    try:
        await push()
        while True:
            message = await websocket.receive_json()
            if message.get("type") == "subscribe":
                tf = message.get("tf")
                if tf in timeframes.TIMEFRAMES:
                    state["tf"] = tf
                limit = message.get("limit")
                state["limit"] = int(limit) if limit else None
                await push()
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("websocket error")
    finally:
        engine.remove_listener(push)
