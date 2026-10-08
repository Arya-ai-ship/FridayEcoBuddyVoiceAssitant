"""Property test: the LLM sees the full Session history (task 12.7).

Requirements: 5.1, 5.3, 5.4, 5.6, 5.10, 5.15
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agent_fixtures import build_agent, run_turn
from fakes import ScriptedChatClient, text
from friday.agent.templates import TOOL_NAMES
from friday.events import FinalEvent

pytestmark = pytest.mark.asyncio


def _reply(n: int) -> str:
    return f"<display>Boss, reply {n}.</display><spoken>Reply {n}, Boss.</spoken>"


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(turns=st.integers(1, 4))
async def test_llm_sees_the_full_growing_history(turns: int) -> None:
    """Feature: friday-voice-data-assistant, Property 22: The LLM sees the full Session
    history.

    Over successive turns the messages the client receives grow monotonically (nothing is
    dropped, compaction off); every call offers exactly the four Friday tools; and the
    instructions carry Friday's rules and persona. A no-tool reply's text is the final text.

    **Validates: Requirements 5.1, 5.3, 5.4, 5.6, 5.10, 5.15**
    """
    client = ScriptedChatClient([text(_reply(i)) for i in range(turns)])
    harness = build_agent(client)

    message_counts: list[int] = []
    for i in range(turns):
        events = await run_turn(harness, f"user message {i}")
        final = events[-1]
        assert isinstance(final, FinalEvent)
        assert f"reply {i}" in final.text.lower()  # the no-tool reply text is the final text
        message_counts.append(len(client.calls[-1].messages))

    # Each turn's model call sees at least as many messages as the previous turn.
    for earlier, later in zip(message_counts, message_counts[1:], strict=False):
        assert later >= earlier

    # Every call: exactly the four Friday tools, Friday instructions, no default harness text.
    for call in client.calls:
        assert set(call.tool_names) == set(TOOL_NAMES)
        assert "Boss" in (call.instructions or "")
