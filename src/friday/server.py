"""The FastAPI app: static UI, the NDJSON chat stream, transcribe, CSV, and session end.

``create_app(container, on_ready)`` registers the routes from the design's HTTP table and
three middlewares: a trusted-host guard (127.0.0.1/localhost only), a security-headers and
strict-CSP layer, and a body-size limit. State-changing POSTs reject a foreign ``Origin``.
Every error body is redacted JSON, and ``/api/chat`` streams events one line at a time
under the per-Session lock, always ending with a ``final`` event.

This module imports no SDK: it depends only on the injected ``Container``.
"""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Final

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from friday.constants import MAX_AUDIO_BYTES, MAX_CHAT_BODY_BYTES, MAX_TEXT_CHARS
from friday.data.dataset import to_csv
from friday.errors import AwsCredentialError, FridayError, RequestError, STTError
from friday.events import NdjsonEncoder
from friday.session import parse_session_id
from friday.wiring import Container

STATIC_DIR: Final = Path(__file__).parent / "static"
SESSION_HEADER: Final = "x-friday-session"
NDJSON_MEDIA_TYPE: Final = "application/x-ndjson"
ALLOWED_HOSTS: Final = ["127.0.0.1", "localhost"]

CSP: Final = (
    "default-src 'self'; img-src 'self' data:; media-src 'self' blob: data:; "
    "connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)
SECURITY_HEADERS: Final[dict[str, str]] = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Cross-Origin-Opener-Policy": "same-origin",
}
STATE_CHANGING_PATHS: Final = frozenset({"/api/chat", "/api/transcribe", "/api/session/end"})
_SLUG: Final = re.compile(r"[^a-z0-9]+")

OnReady = Callable[[], None] | Callable[[], Awaitable[None]]


def create_app(container: Container, on_ready: OnReady | None = None) -> FastAPI:
    """Build the Friday FastAPI app from an injected ``Container``."""
    encoder = NdjsonEncoder(container.redactor)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        if on_ready is not None:
            result = on_ready()
            if result is not None:
                await result
        try:
            yield
        finally:
            await container.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

    @app.middleware("http")
    async def guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        origin_error = _check_origin(request)
        if origin_error is not None:
            return origin_error
        size_error = _check_body_size(request)
        if size_error is not None:
            return size_error
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        # Always revalidate (ETag) so the browser never mixes stale and fresh ES modules.
        response.headers.setdefault("Cache-Control", "no-cache")
        return response

    _register_routes(app, container, encoder)

    @app.exception_handler(FridayError)
    async def _on_friday_error(_request: Request, exc: FridayError) -> JSONResponse:
        status = 400 if isinstance(exc, RequestError) else 500
        return _error(status, exc.kind, "request failed")

    return app


def _check_origin(request: Request) -> Response | None:
    """Reject a present cross-origin header on state-changing POSTs (Req 12.5)."""
    if request.url.path not in STATE_CHANGING_PATHS or request.method != "POST":
        return None
    origin = request.headers.get("origin")
    if origin is None:
        return None
    host = request.headers.get("host", "")
    allowed = {f"http://{host}", f"https://{host}"}
    if origin not in allowed:
        return _error(403, "forbidden_origin", "cross-origin request rejected")
    return None


def _check_body_size(request: Request) -> Response | None:
    """Reject an over-limit ``Content-Length`` before reading the body."""
    length = request.headers.get("content-length")
    if length is None or not length.isdigit():
        return None
    size = int(length)
    if request.url.path == "/api/chat" and size > MAX_CHAT_BODY_BYTES:
        return _error(413, "request_too_large", "request body is too large")
    if request.url.path == "/api/transcribe" and size > MAX_AUDIO_BYTES:
        return _error(413, "audio_too_large", "audio is too large")
    return None


def _register_routes(app: FastAPI, container: Container, encoder: NdjsonEncoder) -> None:
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/api/chat")
    async def chat(request: Request) -> Response:
        if "application/json" not in (request.headers.get("content-type") or ""):
            return _error(400, "invalid_text", "Content-Type must be application/json")
        session_id = _session_id(request)
        body: object = await request.json()
        text = body.get("text") if isinstance(body, dict) else None  # type: ignore[union-attr]
        if not isinstance(text, str) or not (1 <= len(text.strip()) <= MAX_TEXT_CHARS):
            return _error(400, "invalid_text", "text must be 1-2000 characters")
        session = container.sessions.get_or_create(session_id)

        async def stream() -> AsyncIterator[bytes]:
            async with session.lock:
                async for event in container.agent.run_turn(session, text.strip()):
                    yield encoder.encode(event)

        return StreamingResponse(stream(), media_type=NDJSON_MEDIA_TYPE)

    @app.post("/api/transcribe")
    async def transcribe(request: Request) -> Response:
        session_id = _session_id(request)
        container.sessions.get_or_create(session_id)
        audio = await request.body()
        if len(audio) > MAX_AUDIO_BYTES:
            return _error(413, "audio_too_large", "audio is too large")
        try:
            text = await container.stt.transcribe(audio)
        except AwsCredentialError:
            return _error(403, "aws_credentials", "AWS credentials expired or lack access")
        except STTError as exc:
            return _error(400, exc.kind, "transcription failed")
        return JSONResponse({"text": text})

    @app.get("/api/datasets/{dataset_id}.csv")
    async def dataset_csv(dataset_id: str, session: str) -> Response:
        session_id = _parse_session(session)
        store = container.sessions.get_or_create(session_id)
        ds = store.datasets.get(dataset_id)
        if ds is None:
            return _error(404, "dataset_not_found", "dataset is no longer available")
        filename = f"{ds.series_id}_{_slug(ds.indicator)}_{ds.dataset_id}.csv"
        body = container.redactor.redact(to_csv(ds))
        return PlainTextResponse(
            body,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.post("/api/session/end")
    async def session_end(request: Request) -> Response:
        raw = request.headers.get(SESSION_HEADER)
        if not raw:
            raw = (await request.body()).decode("ascii", "ignore")
        container.sessions.end(raw.strip())
        return Response(status_code=204)


def _session_id(request: Request) -> str:
    """The validated session ID from the ``X-Friday-Session`` header (Req 12.5)."""
    raw = request.headers.get(SESSION_HEADER)
    if not raw:
        raise RequestError.invalid_session()
    return parse_session_id(raw)


def _parse_session(raw: str) -> str:
    return parse_session_id(raw)


def _slug(text: str) -> str:
    """A filename-safe slug of an Indicator name."""
    return _SLUG.sub("-", text.strip().casefold()).strip("-") or "series"


def _error(status: int, code: str, message: str) -> JSONResponse:
    """A redacted JSON error body with the wire ``code``."""
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)
