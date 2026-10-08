"""MAF Bedrock Converse client factory and LLM error mapping (Req 5.2, 5.7, 5.12, 12.11).

This is the only module (besides ``wiring.py``) that imports the MAF Bedrock package.

:func:`make_chat_client` builds the ``bedrock-runtime`` client itself, from an
isolated boto3 session, and passes it to ``BedrockChatClient`` together with an
explicit ``model`` and ``region`` (design: spike findings 7 and 8). With a prebuilt
``client=``, the ``BEDROCK_*`` credential values MAF still resolves from the
environment are discarded, so the credentials, region, and model are exactly the
``Settings`` values. ``env_file_path`` is never passed, so MAF reads no ``.env``,
and MAF observability setup helpers are never called (spike finding 9).
"""

from __future__ import annotations

from collections.abc import Callable

from agent_framework import FunctionInvocationConfiguration
from agent_framework_bedrock import BedrockChatClient
from botocore.config import Config

from friday.aws_errors import to_llm_error
from friday.aws_session import isolated_boto3_session
from friday.config import Settings
from friday.constants import AWS_MAX_ATTEMPTS, LLM_CONNECT_TIMEOUT_S, LLM_TIMEOUT_S, MAX_TOOL_CALLS
from friday.errors import FridayError
from friday.redact import Redactor

BEDROCK_RUNTIME_SERVICE = "bedrock-runtime"
"""boto3 service name of the Bedrock Converse API."""


def runtime_config() -> Config:
    """botocore config for Bedrock runtime: 60 s read, 5 s connect, a single attempt.

    ``total_max_attempts`` is used because ``max_attempts=1`` still allows one retry
    (spike finding 7). The agent's ``asyncio.wait_for`` stays the authoritative limit.
    """
    return Config(
        read_timeout=LLM_TIMEOUT_S,
        connect_timeout=LLM_CONNECT_TIMEOUT_S,
        retries={"total_max_attempts": AWS_MAX_ATTEMPTS},
    )


def function_invocation_configuration() -> FunctionInvocationConfiguration:
    """MAF tool-loop limits: Friday's own 9th-call termination fires before these."""
    return {
        "max_iterations": MAX_TOOL_CALLS + 1,
        "max_consecutive_errors_per_request": MAX_TOOL_CALLS + 1,
        "allow_concurrent_invocation": False,
    }


def make_chat_client(settings: Settings) -> BedrockChatClient:
    """Build the MAF Bedrock Converse client from explicit Settings values only.

    The model ID is passed through unchanged (Req 5.2). The session token is used
    only when set (Req 12.11).
    """
    # Only the bedrock-runtime/polly/sts stubs are installed; the other overloads are unknown.
    runtime = isolated_boto3_session(settings).client(  # pyright: ignore[reportUnknownMemberType]
        BEDROCK_RUNTIME_SERVICE,
        region_name=settings.aws_region,
        config=runtime_config(),
    )
    return BedrockChatClient(
        client=runtime,
        model=settings.bedrock_model_id,
        region=settings.aws_region,
        function_invocation_configuration=function_invocation_configuration(),
    )


def map_llm_error(exc: BaseException, redactor: Redactor) -> FridayError:
    """Map a harness or Bedrock failure to a redacted Friday error (Req 5.12, 5.13).

    ``CREDENTIAL`` becomes ``AwsCredentialError``; ``TIMEOUT`` and ``OTHER`` become
    ``LLMUnavailableError``.
    """
    return to_llm_error(exc, redactor)


def make_llm_error_mapper(redactor: Redactor) -> Callable[[BaseException], FridayError]:
    """Bind ``redactor`` into :func:`map_llm_error` for injection by ``wiring.py``."""

    def mapper(exc: BaseException) -> FridayError:
        return map_llm_error(exc, redactor)

    return mapper
