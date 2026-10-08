"""``friday --check`` credential verification (Req 5.12, 5.13, 12.4, 12.11).

``run_checks(container)`` runs one live probe per AWS/FRED dependency and returns a
``CheckResult`` for each, classified and redacted. ``cli.py`` prints ``OK``/``FAIL`` lines
and exits 0 only when every check passes. Each probe is the smallest call that proves
access: STS ``GetCallerIdentity``, a tiny Converse ping through the Container's chat client
(not the harness), Polly ``synthesize_speech("Check.")``, 1 s of Transcribe silence, and
FRED ``UNRATE`` with ``limit=1``.

This is an adapter-layer module: it may touch boto3 and MAF errors. Every message is passed
through the Container's Redactor, so no Credential value is ever printed or logged.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Final, cast

from friday.aws_errors import classify_aws_error, describe_aws_error
from friday.constants import LLM_TIMEOUT_S, STT_SAMPLE_RATE_HZ
from friday.errors import FridayError
from friday.redact import Redactor
from friday.wiring import Container

_log: Final = logging.getLogger(__name__)
_SILENCE_SECONDS: Final = 1
_PING_PROMPT: Final = "Reply with OK."
_POLLY_TEXT: Final = "Check."
_FRED_PROBE_SERIES: Final = "UNRATE"


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one credential check."""

    name: str
    ok: bool
    detail: str

    def line(self) -> str:
        """A fixed-width ``OK``/``FAIL`` line for the CLI."""
        status = "OK" if self.ok else "FAIL"
        return f"{self.name:<13} {status:<5} {self.detail}".rstrip()


async def run_checks(container: Container) -> list[CheckResult]:
    """Run every credential check in order and return their results."""
    redactor = container.redactor
    return [
        await _aws_identity(container, redactor),
        await _bedrock(container, redactor),
        await _polly(container, redactor),
        await _transcribe(container, redactor),
        await _fred(container, redactor),
    ]


def _failure(name: str, exc: BaseException, redactor: Redactor) -> CheckResult:
    """A redacted FAIL result classifying ``exc``."""
    if isinstance(exc, FridayError):
        detail = redactor.redact(exc.message)
    else:
        detail = describe_aws_error(exc, redactor)
    _log.warning("Check %s failed: %s", name, type(exc).__name__)
    return CheckResult(name, False, detail)


async def _aws_identity(container: Container, redactor: Redactor) -> CheckResult:
    name = "AWS identity"
    try:
        identity = cast(
            "dict[str, Any]",
            await asyncio.to_thread(
                container.sts_client.get_caller_identity  # type: ignore[attr-defined]
            ),
        )
        account = redactor.redact(str(identity.get("Account", "")))
        return CheckResult(name, True, f"account {account}")
    except Exception as exc:  # noqa: BLE001 - every failure is reported, not raised
        return _failure(name, exc, redactor)


async def _bedrock(container: Container, redactor: Redactor) -> CheckResult:
    name = "Bedrock LLM"
    try:
        await asyncio.wait_for(
            container.chat_client.get_response(_PING_PROMPT),  # type: ignore[attr-defined]
            LLM_TIMEOUT_S,
        )
        return CheckResult(name, True, "Converse reachable")
    except Exception as exc:  # noqa: BLE001
        return _failure(name, exc, redactor)


async def _polly(container: Container, redactor: Redactor) -> CheckResult:
    name = "Polly"
    try:
        await container.tts.synthesize(_POLLY_TEXT)
        return CheckResult(name, True, "synthesis reachable")
    except Exception as exc:  # noqa: BLE001
        return _failure(name, exc, redactor)


async def _transcribe(container: Container, redactor: Redactor) -> CheckResult:
    name = "Transcribe"
    silence = b"\x00\x00" * (STT_SAMPLE_RATE_HZ * _SILENCE_SECONDS)
    try:
        await container.stt.transcribe(silence)
        return CheckResult(name, True, "streaming reachable")
    except FridayError as exc:
        if exc.kind == "stt_empty":
            # Silence correctly transcribes to nothing: the service is reachable.
            return CheckResult(name, True, "streaming reachable")
        return _failure(name, exc, redactor)
    except Exception as exc:  # noqa: BLE001
        return _failure(name, exc, redactor)


async def _fred(container: Container, redactor: Redactor) -> CheckResult:
    name = "FRED"
    try:
        await container.fred.observations(_FRED_PROBE_SERIES, None, None)
        return CheckResult(name, True, "API key valid")
    except Exception as exc:  # noqa: BLE001
        return _failure(name, exc, redactor)


__all__ = ["CheckResult", "run_checks", "classify_aws_error"]
