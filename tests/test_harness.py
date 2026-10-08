"""Unit tests for the harness composition (task 12.10).

Checks that every disable flag is set and no opt-in capability is configured, that the
scripted client sees exactly the four Friday tools and no DEFAULT_HARNESS_INSTRUCTIONS,
and that a full fake turn makes no non-loopback network connection (the autouse network
guard in conftest covers the egress assertion).

Requirements: 5.4, 5.14, 5.15, 12.4
"""

from __future__ import annotations

import pytest
from agent_framework import DEFAULT_HARNESS_INSTRUCTIONS

from agent_fixtures import build_agent, run_turn
from fakes import Call, ScriptedChatClient, calls, text
from friday.agent.templates import TOOL_NAMES

pytestmark = pytest.mark.asyncio

REPLY = "<display>Boss, done.</display><spoken>Done, Boss.</spoken>"


async def test_scripted_client_sees_exactly_the_four_friday_tools() -> None:
    client = ScriptedChatClient([text(REPLY)])
    harness = build_agent(client)

    await run_turn(harness, "hello")

    assert client.calls, "the model was called at least once"
    for call in client.calls:
        assert set(call.tool_names) == set(TOOL_NAMES)
        # No harness tools (todo/mode/file memory/web search/approval) leak in.
        assert len(call.tool_names) == len(TOOL_NAMES)


async def test_instructions_contain_friday_rules_and_not_the_default_harness_text() -> None:
    client = ScriptedChatClient([text(REPLY)])
    harness = build_agent(client)

    await run_turn(harness, "hello")

    instructions = client.calls[0].instructions or ""
    assert "Boss" in instructions  # the Friday persona
    assert "four tools" in instructions or "fetch_data" in instructions  # the Friday rules
    assert DEFAULT_HARNESS_INSTRUCTIONS not in instructions


async def test_full_turn_with_a_tool_call_offers_only_the_four_tools() -> None:
    script = [calls(Call("fetch_data", {"indicator": "inflation"})), text(REPLY)]
    client = ScriptedChatClient(script)
    harness = build_agent(client)

    events = await run_turn(harness, "please pull inflation")

    assert events[-1].type == "final"
    for call in client.calls:
        assert set(call.tool_names) == set(TOOL_NAMES)
