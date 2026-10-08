"""Command-line entry point for Friday (``friday`` console script).

``main()`` runs the startup sequence from the design: load ``.env`` (process env wins),
build ``Settings`` (one message naming every missing required variable), load the
Indicator_Map, attach the ``RedactingFilter`` to the root handler, bind ``127.0.0.1:port``
manually (mapping EADDRINUSE/EACCES), build the app and the Container, and serve with
uvicorn. ``friday --check`` runs the credential probes and exits 0 only when all pass.

Exit codes: 0 success, 2 for a configuration error (missing vars, bad port, bad map).
This is the composition entry point, so it may import the adapters and uvicorn.
"""

from __future__ import annotations

import asyncio
import errno
import logging
import socket
import sys
from pathlib import Path
from typing import Final

import uvicorn

from friday.check import run_checks
from friday.config import Settings, load_settings, read_environment, secret_values
from friday.data.indicators import load_indicator_map
from friday.errors import ConfigError, FridayError, PortError
from friday.redact import RedactingFilter, Redactor
from friday.server import create_app
from friday.wiring import Container, build_container

_log: Final = logging.getLogger("friday")
_CONFIG_EXIT: Final = 2
_HOST: Final = "127.0.0.1"
_GRACEFUL_TIMEOUT_S: Final = 3
_QUIET_LOGGERS: Final = ("httpx", "httpcore", "agent_framework", "botocore", "boto3")
_CREDENTIAL_WARNING: Final = "AWS credentials look expired or invalid — refresh .env and restart."


def main(argv: list[str] | None = None) -> int:
    """Run the Friday command line and return a process exit code."""
    args = sys.argv[1:] if argv is None else argv
    try:
        settings = _load_settings()
    except ConfigError as exc:
        print(f"Friday: {exc.message}", file=sys.stderr, flush=True)
        return _CONFIG_EXIT

    _configure_logging(secret_values(settings))

    try:
        load_indicator_map(settings.indicator_map_path)  # fail fast on a bad map (Req 12.7)
    except FridayError as exc:
        print(f"Friday: {exc.message}", file=sys.stderr, flush=True)
        return _CONFIG_EXIT

    if "--check" in args:
        return _run_check(settings)
    return _serve(settings)


def _load_settings() -> Settings:
    """Load ``.env`` (process env wins) and build Settings."""
    env = read_environment(Path(".env"))
    return load_settings(env)


def _configure_logging(secrets: list[str]) -> None:
    """Attach a ``RedactingFilter`` to the root handler and quiet noisy loggers (Req 12.4)."""
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter(Redactor(secrets)))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _bind(port: int) -> socket.socket:
    """Bind a socket to 127.0.0.1:port, mapping EADDRINUSE/EACCES to a PortError."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((_HOST, port))
    except OSError as exc:
        sock.close()
        if exc.errno == errno.EADDRINUSE:
            raise PortError(str(port), "port in use") from exc
        if exc.errno == errno.EACCES:
            raise PortError(str(port), "bind not permitted") from exc
        raise PortError(str(port), "invalid value") from exc
    sock.listen()
    return sock


def _serve(settings: Settings) -> int:
    """Bind the port, build the app, and run uvicorn until shutdown."""
    try:
        sock = _bind(settings.port)
    except ConfigError as exc:
        print(f"Friday: {exc.message}", file=sys.stderr, flush=True)
        return _CONFIG_EXIT

    def on_ready() -> None:
        actual_port = sock.getsockname()[1]
        print(f"Friday is running at http://{_HOST}:{actual_port}", flush=True)

    container = build_container(settings)
    app = create_app(container, on_ready=on_ready)
    _spawn_credential_probe(container)

    config = uvicorn.Config(
        app,
        log_config=None,
        timeout_graceful_shutdown=_GRACEFUL_TIMEOUT_S,
    )
    server = uvicorn.Server(config)
    server.run(sockets=[sock])
    return 0


def _run_check(settings: Settings) -> int:
    """Run the credential checks and print one OK/FAIL line each; exit 0 only if all pass."""
    container = build_container(settings)

    async def run() -> list[object]:
        try:
            results = await run_checks(container)
        finally:
            await container.close()
        return list(results)

    results = asyncio.run(run())
    all_ok = True
    for result in results:
        print(result.line())  # type: ignore[attr-defined]
        all_ok = all_ok and getattr(result, "ok", False)
    return 0 if all_ok else 1


def _spawn_credential_probe(container: Container) -> None:
    """Log a redacted warning if the STS probe shows expired/invalid credentials."""
    # A best-effort background probe; failures only warn, the server still starts.
    from typing import Any, cast

    try:
        cast(Any, container.sts_client).get_caller_identity()
    except Exception:  # noqa: BLE001 - a probe failure must never stop startup
        _log.warning(_CREDENTIAL_WARNING)


if __name__ == "__main__":
    raise SystemExit(main())
