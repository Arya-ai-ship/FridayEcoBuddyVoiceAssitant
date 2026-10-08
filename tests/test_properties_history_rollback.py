"""Property test: LLM failures roll back the turn's history (task 12.8, Req 5.7, 5.12)."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agent_fixtures import build_agent, run_turn
from fakes import Call, Raise, ScriptedChatClient, calls, credential_error, text
from friday.events import FinalEvent

pytestmark = pytest.mark.asyncio

REPLY = "<display>Boss, ok.</display><spoken>Ok, Boss.</spoken>"


def _history_len(harness: object) -> int:
    from typing import Any, cast

    session = cast(Any, harness).session.agent_session
    snapshot = cast("dict[str, Any]", session.to_dict())
    state = cast("dict[str, Any]", snapshot.get("state", {}))
    in_memory = cast("dict[str, Any]", state.get("in_memory", {}))
    messages = cast("list[Any]", in_memory.get("messages", []))
    return len(messages)


@st.composite
def failures(draw: st.DrawFn):
    """A failure to raise: credential (bare/wrapped) or a generic error."""
    kind = draw(st.sampled_from(["credential", "credential_wrapped", "other"]))
    if kind == "credential":
        return Raise(credential_error(wrapped=False)), "aws_credentials"
    if kind == "credential_wrapped":
        return Raise(credential_error(wrapped=True)), "aws_credentials"
    return Raise(RuntimeError("model exploded")), "llm_unavailable"


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(data=st.data(), prior_turns=st.integers(0, 2), fail_on_second_call=st.booleans())
async def test_llm_failures_roll_back_the_turns_history(
    data: st.DataObject, prior_turns: int, fail_on_second_call: bool
) -> None:
    """Feature: friday-voice-data-assistant, Property 23: LLM failures roll back the turn's
    history.

    For any failure point in a turn, the AgentSession history after the turn equals the
    history before the user message, and ``final.outcome`` is aws_credentials or
    llm_unavailable accordingly. A credential error wrapped as a MAF ``__cause__`` is still
    classified as aws_credentials.

    **Validates: Requirements 5.7, 5.12**
    """
    failure, expected = data.draw(failures())

    # Build a script: some successful prior turns, then the failing turn. When the failure
    # is on the second model call, the turn's first call makes a (persisted) tool call.
    script: list[object] = [text(REPLY) for _ in range(prior_turns)]
    if fail_on_second_call:
        script.append(calls(Call("fetch_data", {"indicator": "inflation"})))
    script.append(failure)

    harness = build_agent(ScriptedChatClient(script))  # type: ignore[arg-type]

    for i in range(prior_turns):
        await run_turn(harness, f"prior {i}")

    before = _history_len(harness)
    events = await run_turn(harness, "the failing turn")
    after = _history_len(harness)

    final = events[-1]
    assert isinstance(final, FinalEvent)
    assert final.outcome == expected
    # History is rolled back to exactly what it was before this user message (Req 5.7, 5.12).
    assert after == before
