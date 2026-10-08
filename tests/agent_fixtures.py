"""Shared helpers to run the Friday Agent on fakes (used by the agent test modules).

``build_agent`` wires a ``ScriptedChatClient`` (or any MAF chat client), ``FakeFred``,
``FakeTTS``, the real tool registry, middleware, harness, and the ``Agent`` facade exactly
as ``wiring.py`` will, so the tests run the real harness/middleware/tools offline.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_framework import AgentSession, BaseChatClient, InMemoryHistoryProvider

from fakes import FakeFred, FakeTTS
from friday.agent.harness import build_harness_agent
from friday.agent.loop import Agent
from friday.agent.middleware import ChatGuard, FridayMiddleware
from friday.agent.tools import ToolRegistry
from friday.data.indicators import default_indicator_map
from friday.data.plot import ChartRenderer
from friday.errors import AwsCredentialError, FridayError, LLMUnavailableError
from friday.events import Event
from friday.redact import Redactor
from friday.session import Session

SESSION_ID = "00000000-0000-4000-8000-000000000000"


def map_llm_error(exc: BaseException) -> FridayError:
    """Classify a model failure the way ``llm/bedrock.py`` does, without importing it."""
    cause = exc
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if _is_credential(cause):
            return AwsCredentialError()
        cause = cause.__cause__ or cause.__context__
    return LLMUnavailableError("unavailable")


def _is_credential(exc: BaseException) -> bool:
    from botocore.exceptions import ClientError

    if isinstance(exc, AwsCredentialError):
        return True
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return code in {
            "ExpiredTokenException",
            "UnrecognizedClientException",
            "InvalidSignatureException",
            "AccessDeniedException",
            "UnauthorizedException",
        }
    return False


@dataclass
class AgentHarness:
    """A built Agent plus the fakes and the chat client, for inspection in tests."""

    agent: Agent
    client: BaseChatClient
    fred: FakeFred
    tts: FakeTTS
    history: InMemoryHistoryProvider
    session: Session


def build_agent(
    client: BaseChatClient,
    *,
    fred: FakeFred | None = None,
    tts: FakeTTS | None = None,
    chat_timeout_s: float | None = None,
) -> AgentHarness:
    """Build the full Agent on ``client`` and the given (or default) fakes.

    ``chat_timeout_s`` overrides the per-model-call timeout so a hang resolves quickly.
    """
    fred = fred or FakeFred()
    tts = tts or FakeTTS()
    tools = ToolRegistry(fred, ChartRenderer(), default_indicator_map())
    history = InMemoryHistoryProvider()
    redactor = Redactor(["placeholder-secret-value-only"])
    middleware = FridayMiddleware.build(tools, tts, map_llm_error)
    if chat_timeout_s is not None:
        middleware = FridayMiddleware(
            middleware.function, ChatGuard(map_llm_error, timeout_s=chat_timeout_s)
        )
    harness = build_harness_agent(client, tools, middleware, history)
    agent = Agent(harness, tts, history, redactor)
    session = Session(SESSION_ID, AgentSession(), last_seen=0.0)
    return AgentHarness(agent, client, fred, tts, history, session)


async def run_turn(harness: AgentHarness, text: str) -> list[Event]:
    """Run one turn and collect its events."""
    return [event async for event in harness.agent.run_turn(harness.session, text)]


def unpaired_calls(harness: AgentHarness) -> list[str]:
    """Call IDs in the stored history without exactly one matching result (or vice versa).

    Converse rejects a request whose tool calls and tool results do not pair up, so this
    must be empty after every turn.
    """
    from collections import Counter
    from typing import Any, cast

    snapshot = cast("dict[str, Any]", harness.session.agent_session.to_dict())
    state = cast("dict[str, Any]", snapshot.get("state", {}))
    stored = cast("dict[str, Any]", state.get("in_memory", {}))
    messages = cast("list[dict[str, Any]]", stored.get("messages", []))
    calls: Counter[str] = Counter()
    results: Counter[str] = Counter()
    for message in messages:
        for content in cast("list[dict[str, Any]]", message.get("contents", [])):
            if content.get("type") == "function_call":
                calls[str(content.get("call_id"))] += 1
            elif content.get("type") == "function_result":
                results[str(content.get("call_id"))] += 1
    bad = [cid for cid in calls if results[cid] != 1]
    return bad + [cid for cid in results if cid not in calls]
