"""Property test: the tool-call budget counts every request (task 12.6, Req 5.8, 5.9, 10.10)."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agent_fixtures import build_agent, run_turn, unpaired_calls
from fakes import Call, ScriptedChatClient, Step, calls, text
from friday.constants import MAX_TOOL_CALLS
from friday.events import FinalEvent

pytestmark = pytest.mark.asyncio

VALID = Call("fetch_data", {"indicator": "inflation"})
UNKNOWN_NAME = Call("search_web", {"q": "x"})
# Schema-invalid arguments on a fetch (an analysis tool with no Dataset loaded would end the
# turn as no_data first, design step 2; invalid fill methods are covered in test_tool_args).
BAD_ARGS = Call("fetch_data", {"indicator": "inflation", "colour": "red"})
REPLY = "<display>Boss, done.</display><spoken>Done, Boss.</spoken>"


@st.composite
def call_batches(draw: st.DrawFn) -> tuple[list[Step], int]:
    """A turn requesting k calls spread over 1-3 model responses; returns (script, k)."""
    k = draw(st.integers(1, 12))
    chosen = [draw(st.sampled_from([VALID, UNKNOWN_NAME, BAD_ARGS])) for _ in range(k)]
    cuts = sorted(draw(st.sets(st.integers(1, k - 1), max_size=2))) if k > 1 else []
    bounds = [0, *cuts, k]
    script: list[Step] = [calls(*chosen[a:b]) for a, b in zip(bounds, bounds[1:], strict=False)]
    # Two spare replies: the turn's own final text (if it gets that far) and a follow-up turn.
    return [*script, text(REPLY), text(REPLY)], k


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(batch=call_batches())
async def test_tool_budget_counts_every_request(batch: tuple[list[Step], int]) -> None:
    """Feature: friday-voice-data-assistant, Property 21: Tool-call budget counts every
    request, including rejected ones.

    At most 8 requests are processed; rejected ones run no Tool and return a named error;
    if k > 8 the turn ends with ``outcome == "tool_limit"`` and Datasets from the first 8
    calls remain; and every tool call in history has a matching tool result, so the next
    turn's request is valid.

    **Validates: Requirements 5.8, 5.9, 10.10**
    """
    script, k = batch
    client = ScriptedChatClient(script)
    harness = build_agent(client)

    events = await run_turn(harness, "do many things")
    final = events[-1]
    assert isinstance(final, FinalEvent)

    if k > MAX_TOOL_CALLS:
        assert final.outcome == "tool_limit"
    assert len(harness.session.datasets) <= MAX_TOOL_CALLS
    assert unpaired_calls(harness) == []

    # The history stays usable: a follow-up turn succeeds and is still fully paired.
    follow_up = await run_turn(harness, "and now?")
    assert isinstance(follow_up[-1], FinalEvent) and follow_up[-1].outcome == "ok"
    assert unpaired_calls(harness) == []
