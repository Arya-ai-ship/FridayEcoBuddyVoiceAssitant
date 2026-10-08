"""Tests that ``isolated_boto3_session`` ignores configured endpoint URL overrides.

Clients are only constructed and inspected; no request is ever sent.
"""

from __future__ import annotations

import pytest

from friday.aws_session import isolated_boto3_session
from friday.config import Settings

REGION = "us-east-2"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        aws_access_key_id="AKIAFAKEFAKEFAKEFAKE",
        aws_secret_access_key="FAKEsecretVALUEforTESTSonly0123456789xyz",  # noqa: S106
        aws_session_token=None,
        aws_region=REGION,
        fred_api_key="fakefredkey000000000000000000000",
        bedrock_model_id="us.openai.gpt-5.6-terra",
    )


@pytest.fixture
def endpoint_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every endpoint override at a bogus host (and try to re-enable them)."""
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://global-override.invalid")
    monkeypatch.setenv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "https://bedrock-override.invalid")
    monkeypatch.setenv("AWS_ENDPOINT_URL_POLLY", "https://polly-override.invalid")
    monkeypatch.setenv("AWS_ENDPOINT_URL_STS", "https://sts-override.invalid")
    monkeypatch.setenv("AWS_IGNORE_CONFIGURED_ENDPOINT_URLS", "false")


@pytest.mark.usefixtures("endpoint_env")
@pytest.mark.parametrize(
    ("service", "expected"),
    [
        ("bedrock-runtime", f"https://bedrock-runtime.{REGION}.amazonaws.com"),
        ("polly", f"https://polly.{REGION}.amazonaws.com"),
        ("sts", f"https://sts.{REGION}.amazonaws.com"),
    ],
)
def test_endpoint_overrides_are_ignored(settings: Settings, service: str, expected: str) -> None:
    client = isolated_boto3_session(settings).client(service)  # type: ignore[call-overload]
    assert client.meta.endpoint_url == expected
