from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import __version__
from .config import Settings
from .embeddings import EmbeddingAdapter
from .errors import classify_error
from .media import MediaStore
from .schemas import AddRequest, AddResponse, SearchRequest, SearchResponse
from .service import MemoryService
from .state_store import StateStore
from .vector_store import VectorStore

logger = logging.getLogger("aml_adapter")


def _default_route(name: str) -> str:
    return {"health": "/health", "add": "/add", "search": "/search"}[name]


def create_app(
    settings: Settings | None = None,
    service: MemoryService | None = None,
) -> FastAPI:
    service_lock = Lock()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        runtime_settings = settings or Settings.from_env()
        runtime_settings.data_dir.mkdir(parents=True, exist_ok=True)
        application.state.settings = runtime_settings
        application.state.memory_service = service
        if service is None:
            application.state.state_store = StateStore(runtime_settings.state_db)
            application.state.media_store = MediaStore(runtime_settings.media_dir)
            application.state.vector_store = VectorStore(
                runtime_settings.vectors_dir, runtime_settings.track
            )
        yield

    application = FastAPI(
        title="MemPalace AML Adapter",
        version=__version__,
        lifespan=lifespan,
    )

    def runtime_service() -> MemoryService:
        current = application.state.memory_service
        if current is not None:
            return current
        with service_lock:
            current = application.state.memory_service
            if current is None:
                runtime_settings = application.state.settings
                current = MemoryService(
                    runtime_settings.track,
                    application.state.state_store,
                    application.state.media_store,
                    EmbeddingAdapter.for_track(runtime_settings.track),
                    application.state.vector_store,
                )
                application.state.memory_service = current
        return current

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, _error: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"detail": {"reason": "schema_validation_failed"}},
        )

    @application.middleware("http")
    async def safe_request_log(request: Request, call_next):
        started = time.perf_counter()
        request_id = None
        if request.method == "POST" and request.url.path == (settings.add_path if settings else "/add"):
            try:
                payload = json.loads((await request.body()).decode("utf-8"))
                candidate = payload.get("request_id") if isinstance(payload, dict) else None
                if isinstance(candidate, str) and candidate.strip():
                    request_id = candidate
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
        error_category = None
        try:
            response = await call_next(request)
        except Exception as error:  # noqa: BLE001 - transport boundary redacts every failure
            mapping = classify_error(error)
            error_category = mapping.reason
            response = JSONResponse(
                status_code=mapping.status,
                content={"detail": {"reason": mapping.reason}},
            )
        if response.status_code >= 400 and error_category is None:
            error_category = {
                400: "invalid_request",
                409: "idempotency_conflict",
                413: "media_limit_exceeded",
                422: "schema_validation_failed",
                503: "service_unavailable",
            }.get(response.status_code, "internal_error")
        event = {
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        if request_id is not None:
            event["request_id"] = request_id
        result_count = getattr(request.state, "result_count", None)
        if isinstance(result_count, int):
            event["result_count"] = result_count
        if error_category is not None:
            event["error_category"] = error_category
        logger.info(json.dumps(event, ensure_ascii=False, sort_keys=True))
        return response

    health_path = settings.health_path if settings else _default_route("health")
    add_path = settings.add_path if settings else _default_route("add")
    search_path = settings.search_path if settings else _default_route("search")

    @application.get(health_path)
    async def health():
        runtime_settings = application.state.settings
        return {"status": "ok", "version": __version__, "track": runtime_settings.track.value}

    @application.post(add_path, response_model=AddResponse)
    async def add(request: AddRequest):
        return await run_in_threadpool(lambda: runtime_service().add(request))

    @application.post(search_path, response_model=SearchResponse)
    async def search(request: SearchRequest, http_request: Request):
        response = await run_in_threadpool(lambda: runtime_service().search(request))
        http_request.state.result_count = len(response.data)
        return response

    return application


app = create_app()
