from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import routes as api_routes
from .api import ingestion_routes as ingestion_api_routes
from .api.routes import router
from .core.config import Settings, get_settings
from .core.logging import configure_logging, new_request_id, request_id_var
from .domain.schemas import APIError
from .importer import import_workbooks
from .repositories.database import SQLiteDatabase

configure_logging()


def load_normalized_catalog(settings: Settings) -> list[tuple[str, str, dict]]:
    directory = settings.normalized_data_dir
    if not (directory / "rally.json").exists():
        import_workbooks(settings.source_dir, directory)
    rally = json.loads((directory / "rally.json").read_text(encoding="utf-8"))
    workouts = json.loads((directory / "workouts.json").read_text(encoding="utf-8"))
    roadmap = json.loads((directory / "mens_wellness_roadmap.json").read_text(encoding="utf-8"))
    items: list[tuple[str, str, dict]] = []
    for metric in rally["metrics"]:
        items.append(("metric", metric["metric_code"], metric))
    for rule in rally["supplemental_rules"]:
        items.append(("supplemental_rule", rule["rule_id"], rule))
    for sex in ("female", "male"):
        for workout in workouts[sex]:
            items.append((f"workout_{sex}", workout["exercise_id"], workout))
    for index, topic in enumerate(roadmap):
        items.append(("roadmap_backlog", f"{topic['category']}:{topic['source_id']}:{index}", topic))
    return items


def create_app(settings_provider: Callable[[], Settings] = get_settings) -> FastAPI:
    initial_settings = settings_provider()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        current_settings = settings_provider()
        db = SQLiteDatabase(current_settings.database_path)
        db.sync_catalog(load_normalized_catalog(current_settings))
        api_routes.set_runtime(current_settings, db)
        ingestion_api_routes.set_runtime(current_settings, db)
        yield

    app = FastAPI(title=initial_settings.app_name, version="0.2.0", lifespan=lifespan)
    app.state.settings_provider = settings_provider
    if initial_settings.allowed_origins:
        app.add_middleware(CORSMiddleware, allow_origins=initial_settings.allowed_origins,
                           allow_credentials=True, allow_methods=["GET", "POST", "PATCH", "DELETE"], allow_headers=["*"])

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or new_request_id()
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_var.reset(token)

    @app.exception_handler(Exception)
    async def internal_error_handler(request: Request, exc: Exception):
        request_id = request_id_var.get()
        return JSONResponse(status_code=500, content=APIError(code="INTERNAL_ERROR", message="Unexpected server error", request_id=request_id).model_dump())

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "service": initial_settings.app_name, "version": app.version}

    @app.get("/readyz")
    def readyz() -> dict[str, str]:
        catalog = initial_settings.normalized_data_dir / "rally.json"
        return {"status": "ready" if catalog.exists() else "not_ready", "catalog_loaded": str(catalog.exists()).lower()}

    app.include_router(router)
    app.include_router(ingestion_api_routes.router)
    return app


app = create_app()
