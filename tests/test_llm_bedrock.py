"""Unit tests for ``map_llm_error`` and ``make_chat_client`` (Req 5.2, 5.7, 5.12, 5.13, 12.4).

Complements ``test_aws_errors.py`` (the ``classify_aws_error`` table, bare and wrapped
via ``__cause__``/``__context__``, cycles) and ``test_bedrock_factory.py`` (client
config, signer credentials, conflicting process environment). This module drives the
real ``map_llm_error`` path with errors wrapped in real MAF exceptions, checks that
every Credential is redacted from the resulting messages, and that building the client
reads no ``.env`` and sets up no OpenTelemetry.

Clients are only constructed and inspected; no request is sent. Every credential is an
obviously fake placeholder, and the real ``.env`` is never touched (tests ``chdir`` into
``tmp_path``).
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import agent_framework._settings as maf_settings
import agent_framework.observability as maf_observability
import httpx
import pytest
from agent_framework.exceptions import AgentException, ChatClientException
from botocore.client import BaseClient
from botocore.exceptions import ClientError, ConnectTimeoutError, ReadTimeoutError
from opentelemetry import trace

from friday.aws_errors import CREDENTIAL_CODES
from friday.config import Settings, secret_values
from friday.constants import REDACT_MIN_FRAGMENT
from friday.errors import AwsCredentialError, FridayError, LLMUnavailableError
from friday.llm.bedrock import make_chat_client, map_llm_error
from friday.redact import REDACTED, Redactor

FAKE_KEY_ID = "AKIAFAKEFAKEFAKEFAKE"
FAKE_SECRET = "FAKEsecretVALUEforTESTSonly0123456789xyz"  # noqa: S105 - fake placeholder
FAKE_TOKEN = "FAKEsessionTOKENforTESTSonly0123456789"  # noqa: S105 - fake placeholder
FAKE_FRED_KEY = "fakefredkeyPLACEHOLDERforTESTSonly"
MODEL_ID = "us.openai.gpt-5.6-terra"
REGION = "us-east-2"
CREDENTIALS = (FAKE_KEY_ID, FAKE_SECRET, FAKE_TOKEN, FAKE_FRED_KEY)

# Same names as the conflicting process-environment test, here only in a dummy .env.
DOTENV_VALUES = {
    "BEDROCK_CHAT_MODEL": "dotenv.model-from-file",
    "BEDROCK_MODEL_ID": "dotenv.model-from-file",
    "BEDROCK_REGION": "eu-central-1",
    "BEDROCK_ACCESS_KEY": "AKIADOTENVDOTENVDOT0",
    "BEDROCK_SECRET_KEY": "dotenvSecretFromBedrockVariable000000000",
    "BEDROCK_SESSION_TOKEN": "dotenvTokenFromBedrockVariable0000000",
    "AWS_ACCESS_KEY_ID": "AKIADOTENVDOTENVDOT1",
    "AWS_SECRET_ACCESS_KEY": "dotenvSecretFromAwsVariable0000000000000",
    "AWS_SESSION_TOKEN": "dotenvTokenFromAwsVariable00000000000",
    "AWS_REGION": "eu-central-1",
    "AWS_DEFAULT_REGION": "eu-central-1",
    "AWS_PROFILE": "dotenv-profile-that-does-not-exist",
    "AWS_ENDPOINT_URL": "https://dotenv-override.invalid",
}


@pytest.fixture
def settings() -> Settings:
    return Settings(
        aws_access_key_id=FAKE_KEY_ID,
        aws_secret_access_key=FAKE_SECRET,
        aws_session_token=FAKE_TOKEN,
        aws_region=REGION,
        fred_api_key=FAKE_FRED_KEY,
        bedrock_model_id=MODEL_ID,
    )


@pytest.fixture
def redactor(settings: Settings) -> Redactor:
    """The production Redactor: built from every Credential in Settings."""
    return Redactor(secret_values(settings))


def client_error(code: str, status: int = 400, message: str = "boom") -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "Converse",
    )


# --- Real MAF wrappers ---------------------------------------------------------


def maf_from(inner: Exception) -> Exception:
    """``raise ChatClientException(..., inner_exception=e) from e`` (MAF's own idiom)."""
    try:
        raise ChatClientException("Bedrock call failed.", inner_exception=inner) from inner
    except ChatClientException as wrapper:
        return wrapper


def maf_context(inner: Exception) -> Exception:
    """A MAF exception raised while handling ``inner`` (implicit ``__context__`` only)."""
    try:
        try:
            raise inner
        except Exception:
            raise ChatClientException("Bedrock call failed.")  # noqa: B904 - implicit context
    except ChatClientException as wrapper:
        assert wrapper.__cause__ is None
        return wrapper


def maf_two_levels(inner: Exception) -> Exception:
    """``AgentException`` from ``ChatClientException`` from ``inner``."""
    middle = maf_from(inner)
    try:
        raise AgentException("Agent run failed.", inner_exception=middle) from middle
    except AgentException as outer:
        return outer


def bare(inner: Exception) -> Exception:
    return inner


WRAPS: list[Callable[[Exception], Exception]] = [bare, maf_from, maf_context, maf_two_levels]
WRAP_IDS = ["bare", "maf-from", "maf-context", "maf-two-levels"]


# --- map_llm_error: credential errors -------------------------------------------


@pytest.mark.parametrize("wrap", WRAPS, ids=WRAP_IDS)
@pytest.mark.parametrize("code", sorted(CREDENTIAL_CODES))
def test_every_glossary_code_maps_to_aws_credentials(
    code: str, wrap: Callable[[Exception], Exception], redactor: Redactor
) -> None:
    err = map_llm_error(wrap(client_error(code, 403)), redactor)
    assert isinstance(err, AwsCredentialError)
    assert err.kind == "aws_credentials"
    assert code in err.message  # names the service error, not the MAF wrapper


def http_status_error(status: int, body: bytes) -> httpx.HTTPStatusError:
    """A real httpx error: status and body live on ``exc.response``."""
    request = httpx.Request("POST", "https://bedrock-runtime.us-east-2.amazonaws.com/x")
    response = httpx.Response(status, content=body, request=request)
    return httpx.HTTPStatusError("HTTP error", request=request, response=response)


@pytest.mark.parametrize("wrap", WRAPS, ids=WRAP_IDS)
@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (403, b'{"__type":"com.amazon.coral.service#ExpiredTokenException"}', AwsCredentialError),
        (403, b'{"__type":"ExpiredToken"}', AwsCredentialError),
        (
            401,
            b'{"code":"UnrecognizedClientException:http://internal.invalid/"}',
            AwsCredentialError,
        ),
        (403, b'{"Code":"InvalidSignatureException"}', AwsCredentialError),
        (403, b'{"message":"no code here"}', AwsCredentialError),
        (401, b"", AwsCredentialError),
        (403, b'{"__type":"ThrottlingException"}', LLMUnavailableError),
        (500, b'{"__type":"AccessDeniedException"}', LLMUnavailableError),
    ],
    ids=[
        "403-namespaced-__type",
        "403-ExpiredToken",
        "401-code-with-uri",
        "403-Code",
        "403-no-code",
        "401-empty",
        "403-other-code",
        "500-not-auth",
    ],
)
def test_http_auth_body_variants(
    status: int,
    body: bytes,
    expected: type[FridayError],
    wrap: Callable[[Exception], Exception],
    redactor: Redactor,
) -> None:
    assert isinstance(map_llm_error(wrap(http_status_error(status, body)), redactor), expected)


@pytest.mark.parametrize("wrap", WRAPS, ids=WRAP_IDS)
@pytest.mark.parametrize("code", sorted(CREDENTIAL_CODES))
def test_smithy_style_errors(
    code: str, wrap: Callable[[Exception], Exception], redactor: Redactor
) -> None:
    modeled = type(code, (Exception,), {})("modeled smithy error")
    unmodeled = Exception(
        "Unknown error for operation com.amazonaws.bedrock#Converse"
        f" - status: 403 - id: com.amazonaws.bedrock#{code}"
    )
    for exc in (modeled, unmodeled):
        assert isinstance(map_llm_error(wrap(exc), redactor), AwsCredentialError)


# --- map_llm_error: timeouts and other failures ---------------------------------

ClientTimeoutError = type("ClientTimeoutError", (Exception,), {})  # smithy name, no import

UNAVAILABLE: list[Callable[[], Exception]] = [
    TimeoutError,
    lambda: ReadTimeoutError(endpoint_url="https://bedrock-runtime.us-east-2.amazonaws.com"),
    lambda: ConnectTimeoutError(endpoint_url="https://bedrock-runtime.us-east-2.amazonaws.com"),
    lambda: ClientTimeoutError("A timeout error occurred."),
    lambda: client_error("ThrottlingException", 429),
    lambda: client_error("ValidationException", 400, "The provided model identifier is invalid."),
    lambda: RuntimeError("model failed"),
]
UNAVAILABLE_IDS = [
    "asyncio-timeout",
    "read-timeout",
    "connect-timeout",
    "smithy-timeout",
    "throttling",
    "invalid-model-id",
    "runtime",
]


@pytest.mark.parametrize("wrap", WRAPS, ids=WRAP_IDS)
@pytest.mark.parametrize("make", UNAVAILABLE, ids=UNAVAILABLE_IDS)
def test_timeouts_and_other_errors_map_to_llm_unavailable(
    make: Callable[[], Exception], wrap: Callable[[Exception], Exception], redactor: Redactor
) -> None:
    err = map_llm_error(wrap(make()), redactor)
    assert isinstance(err, LLMUnavailableError)
    assert err.kind == "llm_unavailable"


def test_credential_error_wins_over_timeout_in_maf_chain(redactor: Redactor) -> None:
    timeout = maf_from(TimeoutError())
    timeout.__context__ = client_error("ExpiredTokenException", 403)
    assert isinstance(map_llm_error(timeout, redactor), AwsCredentialError)


def test_cyclic_maf_chain_terminates(redactor: Redactor) -> None:
    a = ChatClientException("a")
    b = AgentException("b")
    a.__cause__ = b
    b.__context__ = a
    assert isinstance(map_llm_error(a, redactor), LLMUnavailableError)
    b.__cause__ = client_error("AccessDeniedException", 403)
    assert isinstance(map_llm_error(a, redactor), AwsCredentialError)


# --- Redaction of every Credential ---------------------------------------------

LEAK = (
    f"key={FAKE_KEY_ID} secret={FAKE_SECRET} token={FAKE_TOKEN} fred={FAKE_FRED_KEY}"
    f" parts={FAKE_KEY_ID[4:14]}|{FAKE_SECRET[5:15]}|{FAKE_TOKEN[10:20]}|{FAKE_FRED_KEY[3:13]}"
    f" url=https://api.stlouisfed.org/fred/series/observations?api_key={FAKE_FRED_KEY}&x=1"
)

LEAKY: list[Callable[[], Exception]] = [
    lambda: client_error("InvalidSignatureException", 403, LEAK),
    lambda: client_error("ThrottlingException", 429, LEAK),
    lambda: ReadTimeoutError(endpoint_url=LEAK),
    lambda: RuntimeError(LEAK),
    lambda: http_status_error(403, f'{{"__type":"ExpiredToken","message":"{LEAK}"}}'.encode()),
]
LEAKY_IDS = ["credential", "throttling", "timeout", "runtime", "http-403"]


def assert_no_credential(text: str) -> None:
    """No Credential appears in full or as any ``REDACT_MIN_FRAGMENT``-character fragment."""
    k = REDACT_MIN_FRAGMENT
    for secret in CREDENTIALS:
        for i in range(len(secret) - k + 1):
            assert secret[i : i + k] not in text, f"fragment of a Credential leaked: {text!r}"


@pytest.mark.parametrize("wrap", WRAPS, ids=WRAP_IDS)
@pytest.mark.parametrize("make", LEAKY, ids=LEAKY_IDS)
def test_every_credential_is_redacted(
    make: Callable[[], Exception], wrap: Callable[[Exception], Exception], redactor: Redactor
) -> None:
    err = map_llm_error(wrap(make()), redactor)
    assert isinstance(err, AwsCredentialError | LLMUnavailableError)
    assert_no_credential(err.message)
    assert_no_credential(str(err))


def test_credentials_in_maf_wrapper_text_are_redacted(redactor: Redactor) -> None:
    # The MAF wrapper's own text (and its inner_exception repr in args) carries the leak.
    try:
        raise ChatClientException(LEAK, inner_exception=RuntimeError(LEAK)) from None
    except ChatClientException as exc:
        err = map_llm_error(exc, redactor)
    assert isinstance(err, LLMUnavailableError)
    assert_no_credential(err.message)
    assert REDACTED in err.message


def test_redacted_message_keeps_the_service_error(redactor: Redactor) -> None:
    err = map_llm_error(maf_from(client_error("InvalidSignatureException", 403, LEAK)), redactor)
    assert isinstance(err, AwsCredentialError)
    assert "InvalidSignatureException" in err.message
    assert REDACTED in err.message


# --- make_chat_client: Settings only -------------------------------------------


def _runtime(client: object) -> BaseClient:
    runtime = client._bedrock_client  # type: ignore[attr-defined]  # MAF stores the client here
    assert isinstance(runtime, BaseClient)
    return runtime


def _signer_credentials(runtime: BaseClient) -> tuple[str, str, str | None]:
    creds = runtime._request_signer._credentials.get_frozen_credentials()  # type: ignore[attr-defined]
    return creds.access_key, creds.secret_key, creds.token


@pytest.mark.parametrize(
    "model_id",
    [
        "anthropic.claude-3-haiku-20240307-v1:0",
        "arn:aws:bedrock:us-east-2:123456789012:inference-profile/us.openai.gpt-5.6-terra",
    ],
    ids=["versioned-no-prefix", "inference-profile-arn"],
)
def test_model_id_is_passed_through_unchanged(settings: Settings, model_id: str) -> None:
    assert make_chat_client(replace(settings, bedrock_model_id=model_id)).model == model_id


def test_region_is_not_hardcoded(settings: Settings) -> None:
    client = make_chat_client(replace(settings, aws_region="us-west-2"))
    runtime = _runtime(client)
    assert client.region == "us-west-2"
    assert runtime.meta.region_name == "us-west-2"
    assert runtime.meta.endpoint_url == "https://bedrock-runtime.us-west-2.amazonaws.com"


@pytest.fixture
def dummy_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[object]:
    """Run from a temp dir holding a conflicting ``.env``; record any dotenv read."""
    (tmp_path / ".env").write_text("".join(f"{k}={v}\n" for k, v in DOTENV_VALUES.items()))
    monkeypatch.chdir(tmp_path)
    for name in DOTENV_VALUES:
        monkeypatch.delenv(name, raising=False)
    calls: list[object] = []

    def spy(*args: object, **kwargs: object) -> dict[str, str]:
        calls.append((args, kwargs))
        return {}

    monkeypatch.setattr(maf_settings, "dotenv_values", spy)
    monkeypatch.setattr(maf_observability, "load_dotenv", spy)
    return calls


@pytest.mark.parametrize("token", [FAKE_TOKEN, None])
def test_dotenv_in_working_directory_is_never_read(
    settings: Settings, dummy_dotenv: list[object], token: str | None
) -> None:
    client = make_chat_client(replace(settings, aws_session_token=token))
    runtime = _runtime(client)
    assert dummy_dotenv == []
    assert client.model == MODEL_ID
    assert client.region == REGION
    assert runtime.meta.region_name == REGION
    assert runtime.meta.endpoint_url == f"https://bedrock-runtime.{REGION}.amazonaws.com"
    assert _signer_credentials(runtime) == (FAKE_KEY_ID, FAKE_SECRET, token)
    assert not set(DOTENV_VALUES) & set(os.environ)  # nothing loaded into the process env


def test_no_opentelemetry_setup(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://otel-collector.invalid:4317")
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    monkeypatch.setenv("ENABLE_INSTRUMENTATION", "true")
    monkeypatch.setenv("ENABLE_CONSOLE_EXPORTERS", "true")
    make_chat_client(settings)
    assert maf_observability.OBSERVABILITY_SETTINGS.is_setup is False
    assert isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider)
    assert importlib.util.find_spec("opentelemetry.sdk") is None
