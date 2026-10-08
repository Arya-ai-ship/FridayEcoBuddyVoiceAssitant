"""Tests for ``llm/bedrock.py`` and ``aws_session.py`` (no network calls, fake credentials).

Clients are only constructed and inspected; no request is ever sent.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from botocore.client import BaseClient
from botocore.exceptions import ClientError, ReadTimeoutError

from friday.aws_session import isolated_boto3_session
from friday.config import Settings
from friday.constants import AWS_MAX_ATTEMPTS, LLM_CONNECT_TIMEOUT_S, LLM_TIMEOUT_S, MAX_TOOL_CALLS
from friday.errors import AwsCredentialError, LLMUnavailableError
from friday.llm.bedrock import make_chat_client, make_llm_error_mapper, map_llm_error
from friday.redact import Redactor

FAKE_KEY_ID = "AKIAFAKEFAKEFAKEFAKE"
FAKE_SECRET = "FAKEsecretVALUEforTESTSonly0123456789xyz"  # noqa: S105 - fake placeholder
FAKE_TOKEN = "FAKEsessionTOKENforTESTSonly0123456789"  # noqa: S105 - fake placeholder
MODEL_ID = "us.openai.gpt-5.6-terra"
REGION = "us-east-2"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        aws_access_key_id=FAKE_KEY_ID,
        aws_secret_access_key=FAKE_SECRET,
        aws_session_token=FAKE_TOKEN,
        aws_region=REGION,
        fred_api_key="fakefredkey000000000000000000000",
        bedrock_model_id=MODEL_ID,
    )


@pytest.fixture
def conflicting_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Set BEDROCK_*/AWS_* values that must not influence the clients (placeholders only)."""
    missing = str(tmp_path / "does-not-exist")
    values = {
        "BEDROCK_CHAT_MODEL": "env.model-from-environment",
        "BEDROCK_REGION": "eu-west-1",
        "BEDROCK_ACCESS_KEY": "AKIAENVENVENVENVENV0",
        "BEDROCK_SECRET_KEY": "envSecretFromBedrockVariable000000000000",
        "BEDROCK_SESSION_TOKEN": "envTokenFromBedrockVariable0000000000",
        "AWS_ACCESS_KEY_ID": "AKIAENVENVENVENVENV1",
        "AWS_SECRET_ACCESS_KEY": "envSecretFromAwsVariable0000000000000000",
        "AWS_SESSION_TOKEN": "envTokenFromAwsVariable00000000000000",
        "AWS_REGION": "ap-south-1",
        "AWS_DEFAULT_REGION": "ap-south-1",
        "AWS_PROFILE": "friday-test-profile-that-does-not-exist",
        "AWS_CONFIG_FILE": missing,
        "AWS_SHARED_CREDENTIALS_FILE": missing,
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _runtime(client: object) -> BaseClient:
    runtime = client._bedrock_client  # type: ignore[attr-defined]  # MAF stores the client here
    assert isinstance(runtime, BaseClient)
    return runtime


def _signer_credentials(runtime: BaseClient) -> tuple[str, str, str | None]:
    creds = runtime._request_signer._credentials.get_frozen_credentials()  # type: ignore[attr-defined]
    return creds.access_key, creds.secret_key, creds.token


# --- make_chat_client --------------------------------------------------------


def test_client_uses_settings_model_and_region(settings: Settings) -> None:
    client = make_chat_client(settings)
    runtime = _runtime(client)
    assert client.model == MODEL_ID
    assert client.region == REGION
    assert runtime.meta.region_name == REGION
    assert runtime.meta.service_model.service_name == "bedrock-runtime"


def test_model_id_without_inference_profile_prefix_is_unchanged(settings: Settings) -> None:
    client = make_chat_client(replace(settings, bedrock_model_id="openai.gpt-5.6-terra"))
    assert client.model == "openai.gpt-5.6-terra"


def test_runtime_config_has_timeouts_and_single_attempt(settings: Settings) -> None:
    config = _runtime(make_chat_client(settings)).meta.config
    assert config.read_timeout == LLM_TIMEOUT_S == 60
    assert config.connect_timeout == LLM_CONNECT_TIMEOUT_S == 5
    assert config.retries is not None
    assert config.retries["total_max_attempts"] == AWS_MAX_ATTEMPTS == 1  # botocore adds "mode"
    assert "max_attempts" not in config.retries


def test_function_invocation_limits(settings: Settings) -> None:
    fic = make_chat_client(settings).function_invocation_configuration
    assert fic["max_iterations"] == MAX_TOOL_CALLS + 1
    assert fic["max_consecutive_errors_per_request"] == MAX_TOOL_CALLS + 1
    assert fic["allow_concurrent_invocation"] is False


def test_signer_uses_settings_credentials(settings: Settings) -> None:
    runtime = _runtime(make_chat_client(settings))
    assert _signer_credentials(runtime) == (FAKE_KEY_ID, FAKE_SECRET, FAKE_TOKEN)


@pytest.mark.usefixtures("conflicting_env")
@pytest.mark.parametrize("token", [FAKE_TOKEN, None])
def test_conflicting_environment_is_ignored(settings: Settings, token: str | None) -> None:
    s = replace(settings, aws_session_token=token)
    client = make_chat_client(s)  # no ProfileNotFound despite the missing AWS_PROFILE
    runtime = _runtime(client)
    assert client.model == MODEL_ID
    assert client.region == REGION
    assert runtime.meta.region_name == REGION
    assert _signer_credentials(runtime) == (FAKE_KEY_ID, FAKE_SECRET, token)


# --- isolated_boto3_session ---------------------------------------------------


@pytest.mark.usefixtures("conflicting_env")
@pytest.mark.parametrize("token", [FAKE_TOKEN, None])
def test_isolated_session_ignores_profile_and_env(settings: Settings, token: str | None) -> None:
    session = isolated_boto3_session(replace(settings, aws_session_token=token))
    creds = session.get_credentials()
    assert creds is not None
    frozen = creds.get_frozen_credentials()
    assert (frozen.access_key, frozen.secret_key, frozen.token) == (FAKE_KEY_ID, FAKE_SECRET, token)
    assert session.region_name == REGION
    assert session.profile_name == "default"  # AWS_PROFILE was not consulted


# --- map_llm_error -----------------------------------------------------------


def _client_error(code: str, message: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": message}, "ResponseMetadata": {"HTTPStatusCode": 400}},
        "Converse",
    )


@pytest.fixture
def redactor() -> Redactor:
    return Redactor([FAKE_SECRET, FAKE_TOKEN])


def test_credential_error_maps_to_aws_credential_error(redactor: Redactor) -> None:
    exc = _client_error("UnrecognizedClientException", f"bad token {FAKE_TOKEN}")
    err = map_llm_error(exc, redactor)
    assert isinstance(err, AwsCredentialError)
    assert err.kind == "aws_credentials"
    assert FAKE_TOKEN[:12] not in err.message


def test_wrapped_credential_error_is_found(redactor: Redactor) -> None:
    inner = _client_error("ExpiredTokenException", "expired")
    try:
        try:
            raise inner
        except ClientError as e:
            raise RuntimeError("harness failed") from e
    except RuntimeError as outer:
        assert isinstance(map_llm_error(outer, redactor), AwsCredentialError)


@pytest.mark.parametrize(
    "exc",
    [
        ReadTimeoutError(endpoint_url="https://bedrock-runtime.us-east-2.amazonaws.com"),
        TimeoutError(),
        _client_error("ThrottlingException", f"slow down, secret={FAKE_SECRET}"),
        RuntimeError(f"model failed with {FAKE_SECRET}"),
    ],
    ids=["read-timeout", "asyncio-timeout", "throttling", "other"],
)
def test_timeout_and_other_map_to_llm_unavailable(redactor: Redactor, exc: BaseException) -> None:
    mapper = make_llm_error_mapper(redactor)
    err = mapper(exc)
    assert isinstance(err, LLMUnavailableError)
    assert err.kind == "llm_unavailable"
    assert FAKE_SECRET[:12] not in err.message
    assert FAKE_SECRET[-12:] not in err.message
