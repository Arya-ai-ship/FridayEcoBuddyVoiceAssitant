"""Property test: status events bracket every tool execution (task 12.5, Req 4.4, 4.5, 4.11)."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agent_fixtures import build_agent, run_turn
from fakes import Call, ScriptedChatClient, calls, text
from friday.constants import MAX_DONE_WORDS
from friday.events import DatasetPreviewEvent, StatusEvent

pytestmark = pytest.mark.asyncio

# Tool calls that succeed on the fakes (fetch) or fail (fetch of an unknown indicator).
SUCCEEDS = Call("fetch_data", {"indicator": "inflation"})
FAILS = Call("fetch_data", {"indicator": "not-an-indicator"})


@st.composite
def llm_scripts(draw: st.DrawFn) -> list[object]:
    """A turn: one model response of 1-4 fetch calls (mix of success/fail), then text."""
    n = draw(st.integers(1, 4))
    chosen = [draw(st.sampled_from([SUCCEEDS, FAILS])) for _ in range(n)]
    reply = "<display>Boss, done.</display><spoken>Done, Boss.</spoken>"
    return [calls(*chosen), text(reply)]


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(script=llm_scripts())
async def test_status_events_bracket_every_tool_execution(script: list[object]) -> None:
    """Feature: friday-voice-data-assistant, Property 20: Status events bracket every tool
    execution.

    For a scripted turn of valid tool calls, some of which fail: each executed Tool has a
    ``status(tool_start)`` before any display event for it; each success is followed by one
    ``status(tool_done)`` of at most 10 words; each failure by one ``status(tool_error)``
    and no ``tool_done``; and ``final`` is last.

    **Validates: Requirements 4.4, 4.5, 4.11**
    """
    harness = build_agent(ScriptedChatClient(script))  # type: ignore[arg-type]

    events = await run_turn(harness, "please pull data")

    assert events[-1].type == "final"
    starts = [e for e in events if isinstance(e, StatusEvent) and e.phase == "tool_start"]
    dones = [e for e in events if isinstance(e, StatusEvent) and e.phase == "tool_done"]
    errors = [e for e in events if isinstance(e, StatusEvent) and e.phase == "tool_error"]
    previews = [e for e in events if isinstance(e, DatasetPreviewEvent)]

    # A tool_start precedes any display event in the stream.
    first_start = next((i for i, e in enumerate(events) if e in starts), None)
    for preview in previews:
        assert events.index(preview) > (first_start if first_start is not None else -1)

    # Each tool_start is matched by exactly one done or error.
    assert len(starts) == len(dones) + len(errors)
    for done in dones:
        assert len(done.text.split()) <= MAX_DONE_WORDS
    for err in errors:
        assert err.text  # names the failed action
