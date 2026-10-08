"""Converse contract test: the real BedrockChatClient round-trips tool calls (task 14.4).

Drives the real ``BedrockChatClient`` (from ``make_chat_client`` with placeholder
credentials) against ``botocore.stub.Stubber`` Converse responses that contain a
``toolUse`` block and then text, through the Friday harness with the four Friday tools.
Asserts the tool call executes and its result round-trips to a ``final(outcome="ok")`` in
the same event shapes the ``ScriptedChatClient`` produces, and that the Converse request
carries the configured model ID and the four ``toolSpec`` entries.

**Validates: Requirements 5.1, 5.3, 5.14, 12.11**
"""

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportAttributeAccessIssue=false, reportArgumentType=false

from __future__ import annotations

from typing import Any

import pytest
from agent_framework import AgentSession, InMemoryHistoryProvider
from botocore.stub import Stubber

from fakes import FakeFred, FakeTTS
from friday.agent.harness import build_harness_agent
from friday.agent.loop import Agent
from friday.agent.middleware import FridayMiddleware
from friday.agent.tools import ToolRegistry
from friday.config import Settings
from friday.data.indicators import default_indicator_map
from friday.data.plot import ChartRenderer
from friday.events import FinalEvent
from friday.llm.bedrock import make_chat_client, make_llm_error_mapper
from friday.redact import Redactor
from friday.session import Session

pytestmark = pytest.mark.asyncio

MODEL_ID = "us.openai.gpt-5.6-terra"
FRIDAY_TOOLS = {"fetch_data", "describe_data", "fill_missing", "plot_data"}


def _settings() -> Settings:
    return Settings(
        aws_access_key_id="AKIAFAKEFAKEFAKEFAKE",
        aws_secret_access_key="FAKEsecretVALUEforTESTSonly0123456789xyz",  # noqa: S106
        aws_session_token=None,
        aws_region="us-east-2",
        fred_api_key="fakefredkey000000000000000000000",
        bedrock_model_id=MODEL_ID,
    )


def _tooluse_response() -> dict[str, Any]:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tool-1",
                            "name": "fetch_data",
                            "input": {"indicator": "inflation"},
                        }
                    }
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        "metrics": {"latencyMs": 1},
    }


def _text_response() -> dict[str, Any]:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "text": (
                            "<display>Boss, I pulled inflation.</display>"
                            "<spoken>Pulled inflation, Boss.</spoken>"
                        )
                    }
                ],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        "metrics": {"latencyMs": 1},
    }


async def test_converse_tool_call_round_trips_through_the_harness() -> None:
    client = make_chat_client(_settings())
    runtime = client._bedrock_client  # type: ignore[attr-defined]

    # Record the first Converse request to inspect modelId and toolConfig.
    requests: list[dict[str, Any]] = []
    original = runtime.converse

    def recording_converse(**kwargs: Any) -> Any:
        requests.append(kwargs)
        return original(**kwargs)

    runtime.converse = recording_converse  # type: ignore[method-assign]

    tools = ToolRegistry(FakeFred(), ChartRenderer(), default_indicator_map())
    tts = FakeTTS()
    history = InMemoryHistoryProvider()
    redactor = Redactor(["placeholder-secret-value"])
    middleware = FridayMiddleware.build(tools, tts, make_llm_error_mapper(redactor))
    harness = build_harness_agent(client, tools, middleware, history)
    agent = Agent(harness, tts, history, redactor)
    session = Session("00000000-0000-4000-8000-000000000000", AgentSession(), last_seen=0.0)

    with Stubber(runtime) as stub:
        stub.add_response("converse", _tooluse_response(), expected_params=None)
        stub.add_response("converse", _text_response(), expected_params=None)
        events = [event async for event in agent.run_turn(session, "please pull inflation")]

    types = [e.type for e in events]
    # The tool call round-trips to the same event shapes ScriptedChatClient produces.
    assert "status" in types
    assert "dataset_preview" in types
    assert types[-1] == "final"
    final = events[-1]
    assert isinstance(final, FinalEvent)
    assert final.outcome == "ok"
    assert session.datasets  # the fetch stored a Dataset

    # The request carries the configured model ID and exactly the four Friday toolSpecs.
    first = requests[0]
    assert first["modelId"] == MODEL_ID
    tool_specs = first["toolConfig"]["tools"]
    names = {spec["toolSpec"]["name"] for spec in tool_specs}
    assert names == FRIDAY_TOOLS


async def test_converse_text_only_turn_produces_a_final_ok() -> None:
    client = make_chat_client(_settings())
    runtime = client._bedrock_client  # type: ignore[attr-defined]

    tools = ToolRegistry(FakeFred(), ChartRenderer(), default_indicator_map())
    tts = FakeTTS()
    history = InMemoryHistoryProvider()
    redactor = Redactor(["placeholder-secret-value"])
    middleware = FridayMiddleware.build(tools, tts, make_llm_error_mapper(redactor))
    harness = build_harness_agent(client, tools, middleware, history)
    agent = Agent(harness, tts, history, redactor)
    session = Session("00000000-0000-4000-8000-000000000000", AgentSession(), last_seen=0.0)

    with Stubber(runtime) as stub:
        stub.add_response("converse", _text_response(), expected_params=None)
        events = [event async for event in agent.run_turn(session, "hi Friday")]

    final = events[-1]
    assert isinstance(final, FinalEvent)
    assert final.outcome == "ok"
    assert not session.datasets
