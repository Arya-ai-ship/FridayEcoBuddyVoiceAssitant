"""Classification of AWS SDK failures, shared by every AWS adapter (Req 5.12, 5.13).

:func:`classify_aws_error` is pure: it inspects an exception and everything it
is chained to (``__cause__``, ``__context__``, and the members of an exception
group), so a botocore or smithy error wrapped by the agent framework is still
found. Each link can yield one of three signals:

- ``CREDENTIAL``: an AWS_Credential_Error. That is a botocore ``ClientError``,
  a smithy exception, or an HTTP 401/403 body whose error code (``code`` or
  ``__type``) is one of :data:`CREDENTIAL_CODES`. A 401/403 that carries no
  error code at all also counts. So do botocore ``NoCredentialsError`` and
  ``PartialCredentialsError`` and smithy ``SmithyIdentityError``, because
  missing or partial credentials are fixed the same way (refresh ``.env``).
- ``TIMEOUT``: ``asyncio.TimeoutError`` (the builtin ``TimeoutError``),
  botocore ``ReadTimeoutError``/``ConnectTimeoutError``, or a smithy error
  flagged ``is_timeout_error`` (``ClientTimeoutError``).
- ``OTHER``: anything else.

A credential signal anywhere in the chain wins, because refreshing credentials
is the only action that helps. Otherwise any timeout signal gives ``TIMEOUT``.
Smithy (Transcribe streaming) errors are matched by class name and attributes,
so this module does not import smithy.

The ``to_*_error`` helpers turn a failure into the matching :mod:`friday.errors`
exception. Their messages go through the caller's :class:`~friday.redact.Redactor`,
so no Credential value, in full or in part, reaches a response or a log line.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from enum import Enum
from typing import Final

from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    NoCredentialsError,
    PartialCredentialsError,
    ReadTimeoutError,
)

from friday.errors import AwsCredentialError, LLMUnavailableError, STTError, TTSError
from friday.redact import Redactor


class ErrorKind(Enum):
    """How an AWS failure should be reported."""

    CREDENTIAL = "credential"
    TIMEOUT = "timeout"
    OTHER = "other"


CREDENTIAL_CODES: Final = frozenset(
    {
        "ExpiredToken",
        "ExpiredTokenException",
        "UnrecognizedClientException",
        "InvalidSignatureException",
        "AccessDeniedException",
    }
)
"""AWS error codes that identify an AWS_Credential_Error (Glossary)."""

AUTH_STATUSES: Final = frozenset({401, 403})
"""HTTP statuses that mean the request was not authenticated or not permitted."""

_CREDENTIAL_TYPES: Final = (NoCredentialsError, PartialCredentialsError)
_CREDENTIAL_TYPE_NAMES: Final = frozenset({"SmithyIdentityError"})
_TIMEOUT_TYPES: Final = (TimeoutError, ReadTimeoutError, ConnectTimeoutError)
_TIMEOUT_TYPE_NAMES: Final = frozenset({"ClientTimeoutError"})

# Smithy reports unmodeled errors as "Unknown error for operation <op> - status: 403 - id: <code>".
_SMITHY_STATUS: Final = re.compile(r"status: (\d{3})")
_SMITHY_ID: Final = re.compile(r"- id: (\S+)")


# --- Chain walk ------------------------------------------------------------


def iter_exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield ``exc``, then its causes, contexts, and group members, each once.

    A visited set stops cycles (for example an exception whose ``__context__``
    points back at an exception that wraps it).
    """
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        link = stack.pop()
        if id(link) in seen:
            continue
        seen.add(id(link))
        yield link
        nested: list[BaseException] = []
        if link.__cause__ is not None:
            nested.append(link.__cause__)
        if link.__context__ is not None:
            nested.append(link.__context__)
        if isinstance(link, BaseExceptionGroup):
            nested.extend(link.exceptions)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        stack.extend(reversed(nested))  # Depth-first: cause before context.


def _find(exc: BaseException) -> tuple[ErrorKind, BaseException]:
    """Return the classification and the chain link that best describes it.

    For ``OTHER`` that is the first link carrying an AWS error code (so the
    message names the service error, not a framework wrapper), else ``exc``.
    """
    timeout: BaseException | None = None
    coded: BaseException | None = None
    for link in iter_exception_chain(exc):
        kind = _classify_link(link)
        if kind is ErrorKind.CREDENTIAL:
            return kind, link
        if kind is ErrorKind.TIMEOUT and timeout is None:
            timeout = link
        if coded is None and _code_and_status(link)[0] is not None:
            coded = link
    if timeout is not None:
        return ErrorKind.TIMEOUT, timeout
    return ErrorKind.OTHER, coded if coded is not None else exc


def classify_aws_error(exc: BaseException) -> ErrorKind:
    """Classify an AWS SDK failure (possibly wrapped) as credential, timeout, or other."""
    return _find(exc)[0]


# --- Per-link signals ------------------------------------------------------


def _classify_link(link: BaseException) -> ErrorKind:
    """Classify one exception, ignoring what it is chained to."""
    if isinstance(link, _CREDENTIAL_TYPES) or type(link).__name__ in _CREDENTIAL_TYPE_NAMES:
        return ErrorKind.CREDENTIAL
    code, status = _code_and_status(link)
    if code in CREDENTIAL_CODES or (code is None and status in AUTH_STATUSES):
        return ErrorKind.CREDENTIAL
    if isinstance(link, _TIMEOUT_TYPES) or type(link).__name__ in _TIMEOUT_TYPE_NAMES:
        return ErrorKind.TIMEOUT
    if getattr(link, "is_timeout_error", False) is True:
        return ErrorKind.TIMEOUT
    return ErrorKind.OTHER


def _code_and_status(link: BaseException) -> tuple[str | None, int | None]:
    """Extract the AWS error code and HTTP status a link carries, if any."""
    if isinstance(link, ClientError):
        return _client_error_signal(link)
    status = _http_status(link)
    code = _attr_code(link)
    if code is None and status in AUTH_STATUSES:
        code = _body_code(link)
    if code is None and type(link).__name__ in CREDENTIAL_CODES:
        code = type(link).__name__  # A modeled smithy exception named after its code.
    if code is None or status is None:
        smithy_code, smithy_status = _smithy_message_signal(link)
        code = code if code is not None else smithy_code
        status = status if status is not None else smithy_status
    return code, status


def _client_error_signal(err: ClientError) -> tuple[str | None, int | None]:
    """Read ``Error.Code`` and ``ResponseMetadata.HTTPStatusCode`` from a botocore error."""
    response: Mapping[str, object] = err.response  # pyright: ignore[reportAssignmentType]
    error = response.get("Error")
    meta = response.get("ResponseMetadata")
    code = error.get("Code") if isinstance(error, Mapping) else None
    status = meta.get("HTTPStatusCode") if isinstance(meta, Mapping) else None
    return _normalize_code(code), status if isinstance(status, int) else None


def _http_status(link: BaseException) -> int | None:
    """HTTP status from ``status_code``/``status`` on the exception or its ``response``."""
    for holder in (link, _safe_getattr(link, "response")):
        for name in ("status_code", "status"):
            value = _safe_getattr(holder, name)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


def _attr_code(link: BaseException) -> str | None:
    """Error code from ``code``/``error_code`` attributes (smithy and generic SDK errors)."""
    for name in ("code", "error_code"):
        code = _normalize_code(_safe_getattr(link, name))
        if code is not None:
            return code
    return None


def _body_code(link: BaseException) -> str | None:
    """Error code from the ``code``/``__type`` field of an HTTP error body."""
    for holder in (link, _safe_getattr(link, "response")):
        for name in ("body", "content", "text"):
            code = _code_from_body(_safe_getattr(holder, name))
            if code is not None:
                return code
    return None


def _code_from_body(body: object) -> str | None:
    """Parse a JSON body (bytes, str, or mapping) and return its normalized error code."""
    if isinstance(body, bytes | bytearray):
        body = bytes(body).decode("utf-8", errors="replace")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except ValueError:
            return None
    if not isinstance(body, Mapping):
        return None
    fields: Mapping[object, object] = body  # pyright: ignore[reportUnknownVariableType]
    for key in ("code", "__type", "Code"):
        code = _normalize_code(fields.get(key))
        if code is not None:
            return code
    return None


def _smithy_message_signal(link: BaseException) -> tuple[str | None, int | None]:
    """Code and status from a smithy "Unknown error ... - status: N - id: X" message."""
    text = _safe_str(link)
    if "Unknown error for operation" not in text:
        return None, None
    status_match = _SMITHY_STATUS.search(text)
    id_match = _SMITHY_ID.search(text)
    status = int(status_match.group(1)) if status_match else None
    return _normalize_code(id_match.group(1) if id_match else None), status


def _normalize_code(raw: object) -> str | None:
    """Reduce ``ns#Code`` / ``Code:uri`` forms to the bare code; ``None`` if absent."""
    if not isinstance(raw, str):
        return None
    code = raw.rsplit("#", 1)[-1].split(":", 1)[0].strip()
    return code or None


def _safe_getattr(obj: object, name: str) -> object:
    """``getattr`` that treats a raising property as absent."""
    try:
        return getattr(obj, name, None)
    except Exception:  # A broken attribute just means "no signal".
        return None


def _safe_str(obj: object) -> str:
    """``str`` that tolerates exceptions whose ``__str__`` raises."""
    try:
        return str(obj)
    except Exception:
        return ""


# --- Redacted messages and conversions --------------------------------------


def describe_aws_error(exc: BaseException, redactor: Redactor) -> str:
    """One-line, redacted description of the link that decided the classification."""
    link = _find(exc)[1]
    code, _ = _code_and_status(link)
    label = type(link).__name__ if code is None else f"{type(link).__name__} {code}"
    text = _safe_str(link)
    return redactor.redact(f"{label}: {text}" if text else label)


def to_credential_error(exc: BaseException, redactor: Redactor) -> AwsCredentialError:
    """Build an :class:`AwsCredentialError` with a redacted message."""
    detail = describe_aws_error(exc, redactor)
    return AwsCredentialError(f"AWS credentials have expired or lack access ({detail}).")


def to_llm_error(
    exc: BaseException, redactor: Redactor
) -> AwsCredentialError | LLMUnavailableError:
    """Map a Bedrock failure: credentials → AWS error; timeouts and others → unavailable."""
    if classify_aws_error(exc) is ErrorKind.CREDENTIAL:
        return to_credential_error(exc, redactor)
    detail = describe_aws_error(exc, redactor)
    return LLMUnavailableError(f"The language model is unavailable ({detail}).")


def to_tts_error(exc: BaseException, redactor: Redactor) -> AwsCredentialError | TTSError:
    """Map a Polly failure to ``aws_credentials``, ``tts_timeout``, or ``tts_failed``."""
    kind = classify_aws_error(exc)
    if kind is ErrorKind.CREDENTIAL:
        return to_credential_error(exc, redactor)
    detail = describe_aws_error(exc, redactor)
    if kind is ErrorKind.TIMEOUT:
        return TTSError("tts_timeout", f"Speech synthesis timed out ({detail}).")
    return TTSError("tts_failed", f"Speech synthesis failed ({detail}).")


def to_stt_error(exc: BaseException, redactor: Redactor) -> AwsCredentialError | STTError:
    """Map a Transcribe failure to ``aws_credentials``, ``stt_timeout``, or ``stt_failed``."""
    kind = classify_aws_error(exc)
    if kind is ErrorKind.CREDENTIAL:
        return to_credential_error(exc, redactor)
    detail = describe_aws_error(exc, redactor)
    if kind is ErrorKind.TIMEOUT:
        return STTError("stt_timeout", f"Transcription timed out ({detail}).")
    return STTError("stt_failed", f"Transcription failed ({detail}).")
