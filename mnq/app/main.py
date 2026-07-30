"""Application assembly — the only place the pieces are wired together."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router
from .config import Settings, load_settings
from .core import timeframes
from .core.engine import Engine
from .core.store import BarStore
from .feed import create as create_feed

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    timeframes.configure_session(settings.session_tz, settings.session_open_hour)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        feed = create_feed(settings)
        store = BarStore(settings.bars_dir, settings.max_1m_bars)
        engine = Engine(settings, feed, store)
        app.state.engine = engine
        log.info(
            "starting %s on %s via %s feed",
            settings.symbol,
            f"http://{settings.host}:{settings.port}",
            feed.name,
        )
        await engine.start()
        try:
            yield
        finally:
            await engine.stop()

    app = FastAPI(title="MNQ Live Dashboard", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.include_router(router)

    web_dir = settings.web_dir
    if web_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

        @app.get("/")
        async def index():
            return FileResponse(str(web_dir / "index.html"))

    return app


app = create_app()
