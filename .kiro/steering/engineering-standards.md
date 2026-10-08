# Engineering Standards (Friday)

These apply to all code in this repo. The "Engineering Standards" section of the friday-voice-data-assistant design doc (.kiro/specs/friday-voice-data-assistant/design.md) is the source of truth.

## Structure and layering
- Python package lives in `src/friday/` (src layout). Tests in `tests/`, mirroring module names.
- Domain logic (`data/`, `agent/narration.py`, `agent/templates.py`) is pure: no FastAPI, boto3, httpx, or environment access.
- External services (Bedrock, Polly, Transcribe, FRED) sit behind Protocols in `friday/ports.py` / `llm/base.py`; the Agent depends only on Protocols.
- `wiring.py` is the only place concrete adapters are constructed. No module-level singletons or global mutable state.
- Only `config.read_environment` reads the environment; pass `Settings` explicitly.

## DRY / single sources of truth
- Limits and timeouts live in `friday/constants.py`; colors in `friday/style.py`. Never hardcode them elsewhere. JS/CSS copies are verified by `tests/test_parity.py`.
- User-facing strings live in `agent/templates.py`.
- Tool schemas come from Pydantic models (schema + validation from one definition).
- Errors subclass `FridayError` in `friday/errors.py` with a `kind` code reused in events/HTTP responses.
- Before writing a helper, search for an existing one.

## Code quality
- Type hints everywhere; pyright strict on `src/`. Docstrings on public functions.
- Small, single-purpose functions; keep files around 300 lines or fewer.
- Domain objects are immutable (frozen dataclasses, read-only arrays).
- Async I/O; wrap blocking SDK calls with `asyncio.to_thread`.
- Logging via stdlib `logging` through the RedactingFilter; no `print` outside `cli.py`.
- Run `make check` (ruff lint + format check, pyright, pytest) before marking a task done.

## Security
- Never write credential values into code, docs, tests, fixtures, or logs; reference env var names only. `.env` stays git-ignored.
- Pin dependencies with `==`; lock with `uv.lock`.
- Backend binds to 127.0.0.1 only; strict CSP (`default-src 'self'`); no third-party CDNs.

## Frontend
- Plain ES modules, no build step, no inline scripts/styles. Pure logic (`state.js`, `input.js`) separate from DOM code; tested with `node --test`.
