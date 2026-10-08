"""Unit tests for AWS error classification and conversion (Req 5.12, 5.13).

Errors are built in-process (botocore ``ClientError`` instances and fake wrapper
chains); nothing touches the network. Every secret is an obviously fake placeholder.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest
from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    NoCredentialsError,
    PartialCredentialsError,
    ReadTimeoutError,
)

from friday.aws_errors import (
    CREDENTIAL_CODES,
    ErrorKind,
    classify_aws_error,
    describe_aws_error,
    iter_exception_chain,
    to_llm_error,
    to_stt_error,
    to_tts_error,
)
from friday.errors import AwsCredentialError, LLMUnavailableError, STTError, TTSError
from friday.redact import REDACTED, Redactor

FAKE_SECRET = "FAKEsecretVALUEforTESTSonly0123456789xyz"  # noqa: S105 - fake placeholder
FAKE_KEY_ID = "AKIAFAKEFAKEFAKEFAKE"


def client_error(code: str, status: int = 400, message: str = "boom") -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"RequestId": "r", "HostId": "h", "HTTPStatusCode": status},
        },
        "Converse",
    )


class ChatClientException(Exception):
    """Stand-in for a MAF wrapper exception."""


class ServiceResponseException(Exception):
    """A second wrapper level."""


def wrap_cause(inner: BaseException, levels: int) -> BaseException:
    """Wrap ``inner`` via ``raise ... from`` ``levels`` times."""
    exc = inner
    for i in range(levels):
        try:
            raise (ChatClientException if i % 2 == 0 else ServiceResponseException)(
                "wrapped"
            ) from exc
        except Exception as wrapper:  # noqa: BLE001
            exc = wrapper
    return exc


def wrap_context(inner: BaseException, levels: int) -> BaseException:
    """Wrap ``inner`` as the implicit ``__context__`` ``levels`` times."""
    exc = inner
    for _ in range(levels):
        try:
            try:
                raise exc
            except BaseException:
                raise ChatClientException("while handling")  # noqa: B904 - implicit context
        except ChatClientException as wrapper:
            assert wrapper.__cause__ is None
            exc = wrapper
    return exc


CASES: list[tuple[str, BaseException, ErrorKind]] = [
    *[(code, client_error(code, 403), ErrorKind.CREDENTIAL) for code in sorted(CREDENTIAL_CODES)],
    ("ExpiredToken-400", client_error("ExpiredToken", 400), ErrorKind.CREDENTIAL),
    ("asyncio-timeout", TimeoutError(), ErrorKind.TIMEOUT),
    ("read-timeout", ReadTimeoutError(endpoint_url="https://example.invalid"), ErrorKind.TIMEOUT),
    (
        "connect-timeout",
        ConnectTimeoutError(endpoint_url="https://example.invalid"),
        ErrorKind.TIMEOUT,
    ),
    ("throttling", client_error("ThrottlingException", 429), ErrorKind.OTHER),
    ("validation", client_error("ValidationException", 400), ErrorKind.OTHER),
    ("runtime", RuntimeError("model exploded"), ErrorKind.OTHER),
]


@pytest.mark.parametrize(("name", "exc", "expected"), CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("levels", [0, 1, 2])
@pytest.mark.parametrize("wrap", [wrap_cause, wrap_context], ids=["cause", "context"])
def test_classification_bare_and_wrapped(
    name: str,
    exc: BaseException,
    expected: ErrorKind,
    levels: int,
    wrap: Callable[[BaseException, int], BaseException],
) -> None:
    assert classify_aws_error(wrap(exc, levels)) is expected


def test_maf_output_config_value_error_chain() -> None:
    # Spike item 7: an outputConfig ValidationException is re-raised as ValueError from e.
    inner = client_error("ValidationException")
    try:
        raise ValueError("outputConfig rejected") from inner
    except ValueError as exc:
        assert classify_aws_error(exc) is ErrorKind.OTHER
    try:
        raise ValueError("outputConfig rejected") from client_error("ExpiredTokenException", 403)
    except ValueError as exc:
        assert classify_aws_error(exc) is ErrorKind.CREDENTIAL


def test_cyclic_chain_terminates() -> None:
    a = ChatClientException("a")
    b = ServiceResponseException("b")
    a.__cause__ = b
    b.__context__ = a
    assert classify_aws_error(a) is ErrorKind.OTHER
    assert [type(e) for e in iter_exception_chain(a)] == [
        ChatClientException,
        ServiceResponseException,
    ]
    b.__cause__ = client_error("UnrecognizedClientException", 403)
    assert classify_aws_error(a) is ErrorKind.CREDENTIAL


def test_credential_wins_over_timeout_in_chain() -> None:
    exc = TimeoutError()
    exc.__context__ = client_error("ExpiredTokenException", 403)
    assert classify_aws_error(exc) is ErrorKind.CREDENTIAL


def test_exception_group_members_are_inspected() -> None:
    group = ExceptionGroup("task group", [RuntimeError("x"), client_error("AccessDeniedException")])
    assert classify_aws_error(group) is ErrorKind.CREDENTIAL


def test_missing_and_partial_credentials_are_credential_errors() -> None:
    assert classify_aws_error(NoCredentialsError()) is ErrorKind.CREDENTIAL
    partial = PartialCredentialsError(provider="explicit", cred_var="aws_secret_access_key")
    assert classify_aws_error(partial) is ErrorKind.CREDENTIAL


# --- HTTP bodies and smithy-style errors ------------------------------------


class HttpStatusError(Exception):
    def __init__(self, status: int, body: object) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.body = body


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (403, b'{"__type":"com.amazon.coral.service#ExpiredTokenException"}', ErrorKind.CREDENTIAL),
        (401, '{"code": "UnrecognizedClientException:http://internal/"}', ErrorKind.CREDENTIAL),
        (403, {"code": "InvalidSignatureException"}, ErrorKind.CREDENTIAL),
        (403, b"not json", ErrorKind.CREDENTIAL),
        (403, b'{"__type":"SomeOtherException"}', ErrorKind.OTHER),
        (500, b'{"__type":"ExpiredTokenException"}', ErrorKind.OTHER),
    ],
)
def test_http_auth_body_codes(status: int, body: object, expected: ErrorKind) -> None:
    assert classify_aws_error(HttpStatusError(status, body)) is expected


@dataclass(kw_only=True)
class CallError(Exception):
    """Mimics smithy_core.exceptions.CallError without importing smithy."""

    message: str = field(default="", kw_only=False)
    is_timeout_error: bool = False

    def __post_init__(self) -> None:
        super().__init__(self.message)


class ClientTimeoutError(CallError):
    pass


class AccessDeniedException(CallError):
    pass


def test_smithy_unmodeled_error_with_credential_id() -> None:
    exc = CallError(
        "Unknown error for operation com.amazonaws.transcribestreaming#StartStreamTranscription"
        " - status: 403 - id: com.amazonaws.transcribestreaming#UnrecognizedClientException"
    )
    assert classify_aws_error(wrap_cause(exc, 2)) is ErrorKind.CREDENTIAL


def test_smithy_status_without_id_and_other_errors() -> None:
    no_id = CallError("Unknown error for operation x#Op - status: 401")
    assert classify_aws_error(no_id) is ErrorKind.CREDENTIAL
    server = CallError("Unknown error for operation x#Op - status: 500 - id: x#InternalFailure")
    assert classify_aws_error(server) is ErrorKind.OTHER


def test_smithy_timeouts_and_modeled_names() -> None:
    assert classify_aws_error(ClientTimeoutError("A timeout error occurred.")) is ErrorKind.TIMEOUT
    assert classify_aws_error(CallError("slow", is_timeout_error=True)) is ErrorKind.TIMEOUT
    assert classify_aws_error(AccessDeniedException("denied")) is ErrorKind.CREDENTIAL


# --- Conversions and redaction ----------------------------------------------


@pytest.fixture
def redactor() -> Redactor:
    return Redactor([FAKE_SECRET])


def leaky_credential_error() -> BaseException:
    message = f"Signature mismatch for Credential={FAKE_KEY_ID}/x secret={FAKE_SECRET[3:20]}"
    return wrap_cause(client_error("InvalidSignatureException", 403, message), 2)


def test_messages_are_redacted(redactor: Redactor) -> None:
    exc = leaky_credential_error()
    for err in (
        to_llm_error(exc, redactor),
        to_tts_error(exc, redactor),
        to_stt_error(exc, redactor),
    ):
        assert isinstance(err, AwsCredentialError)
        assert err.kind == "aws_credentials"
        assert FAKE_KEY_ID not in err.message
        assert FAKE_SECRET[3:20] not in err.message
        assert REDACTED in err.message
        assert "InvalidSignatureException" in err.message
    assert FAKE_SECRET[3:20] not in describe_aws_error(exc, redactor)


def test_llm_mapping(redactor: Redactor) -> None:
    assert isinstance(to_llm_error(TimeoutError(), redactor), LLMUnavailableError)
    other = to_llm_error(wrap_cause(client_error("ThrottlingException", 429), 1), redactor)
    assert isinstance(other, LLMUnavailableError)
    assert other.kind == "llm_unavailable"
    assert "ThrottlingException" in other.message


@pytest.mark.parametrize(
    ("exc", "tts_kind", "stt_kind"),
    [
        (TimeoutError(), "tts_timeout", "stt_timeout"),
        (ReadTimeoutError(endpoint_url="https://example.invalid"), "tts_timeout", "stt_timeout"),
        (client_error("ServiceFailureException", 500), "tts_failed", "stt_failed"),
    ],
)
def test_speech_mapping(
    exc: BaseException, tts_kind: str, stt_kind: str, redactor: Redactor
) -> None:
    tts = to_tts_error(exc, redactor)
    stt = to_stt_error(exc, redactor)
    assert isinstance(tts, TTSError)
    assert isinstance(stt, STTError)
    assert (tts.kind, stt.kind) == (tts_kind, stt_kind)
