"""Unit tests for the Agent facade ``run_turn`` (task 12.9).

Drives the real harness/middleware/tools through ``ScriptedChatClient`` and the fakes.
Covers the happy path and event sequence, the fixed outcome texts, that credential errors
make no TTS call, TTS failure/timeout still send text, and a hung model call times out.

Requirements: 4.4, 4.6, 4.8, 5.6, 5.7, 5.9, 5.12, 8.1, 8.6
"""

from __future__ import annotations

import pytest

from agent_fixtures import build_agent, run_turn, unpaired_calls
from fakes import (
    Call,
    Hang,
    Raise,
    ScriptedChatClient,
    calls,
    credential_error,
    text,
)
from friday.constants import MAX_SPOKEN_WORDS, MAX_STATUS_WORDS
from friday.events import FinalEvent, StatusEvent

pytestmark = pytest.mark.asyncio

REPLY = "<display>Boss, I pulled inflation.</display><spoken>Pulled inflation, Boss.</spoken>"


async def test_fetch_happy_path_event_sequence_and_offer() -> None:
    script = [calls(Call("fetch_data", {"indicator": "inflation"})), text(REPLY)]
    harness = build_agent(ScriptedChatClient(script))

    events = await run_turn(harness, "please pull inflation")

    types = [e.type for e in events]
    assert types == ["status", "dataset_preview", "status", "final"]
    start, _, done, final = events
    assert isinstance(start, StatusEvent) and start.phase == "tool_start"
    assert isinstance(done, StatusEvent) and done.phase == "tool_done"
    assert isinstance(final, FinalEvent) and final.outcome == "ok"
    assert "Boss" in final.text
    # The next-step offer after a fetch is appended.
    assert "statistics" in final.text and "plot" in final.text
    assert final.audio is not None  # spoken text was synthesized
    assert harness.session.datasets


async def test_no_data_analysis_request_returns_fixed_text_and_runs_no_tool() -> None:
    script = [calls(Call("describe_data", {}))]
    harness = build_agent(ScriptedChatClient(script))

    events = await run_turn(harness, "run the stats")

    final = events[-1]
    assert isinstance(final, FinalEvent) and final.outcome == "no_data"
    assert "fetch" in final.text.lower()
    assert not harness.session.datasets
    assert not any(e.type == "dataset_preview" for e in events)


async def test_credential_error_makes_no_tts_call() -> None:
    harness = build_agent(ScriptedChatClient([Raise(credential_error(wrapped=True))]))

    events = await run_turn(harness, "please pull inflation")

    final = events[-1]
    assert isinstance(final, FinalEvent) and final.outcome == "aws_credentials"
    assert final.audio is None and final.spoken_text == ""
    assert harness.tts.calls == []  # Req 5.12: no TTS on a credential error


async def test_tts_failure_still_sends_text_with_an_audio_error() -> None:
    script = [text(REPLY)]
    harness = build_agent(ScriptedChatClient(script), tts=_failing_tts("failed"))

    events = await run_turn(harness, "hi")

    final = events[-1]
    assert isinstance(final, FinalEvent) and final.outcome == "ok"
    assert final.text and final.audio is None
    assert final.audio_error == "tts_failed"


async def test_tts_timeout_still_sends_text_with_an_audio_error() -> None:
    script = [text(REPLY)]
    harness = build_agent(ScriptedChatClient(script), tts=_failing_tts("timeout"))

    events = await run_turn(harness, "hi")

    final = events[-1]
    assert isinstance(final, FinalEvent) and final.outcome == "ok"
    assert final.audio_error == "tts_timeout"


async def test_hung_model_call_times_out_as_llm_unavailable() -> None:
    # ChatGuard wraps each model call in wait_for; a short timeout makes the hang resolve.
    harness = build_agent(ScriptedChatClient([Hang()]), chat_timeout_s=0.05)

    events = await run_turn(harness, "please pull inflation")
    final = events[-1]
    assert isinstance(final, FinalEvent) and final.outcome == "llm_unavailable"


def _failing_tts(kind: str):
    from fakes import FakeTTS

    return FakeTTS(failure=kind)  # type: ignore[arg-type]


# --- Regression tests from the readiness sweep -------------------------------------

FETCH = Call("fetch_data", {"indicator": "inflation"})


async def test_start_line_names_the_canonical_indicator_for_an_alias() -> None:
    script = [calls(Call("fetch_data", {"indicator": "  CPI Inflation "})), text(REPLY)]
    harness = build_agent(ScriptedChatClient(script))

    events = await run_turn(harness, "pull cpi inflation")

    start = events[0]
    assert isinstance(start, StatusEvent) and start.phase == "tool_start"
    assert start.text == "Fetching inflation data."


async def test_tool_limit_across_two_model_calls_keeps_history_paired() -> None:
    # 4 calls, then 5 more: the 9th ends the turn. Results of the first batch were already
    # stored by the harness and must not be stored a second time.
    script = [calls(*[FETCH] * 4), calls(*[FETCH] * 5), text(REPLY)]
    harness = build_agent(ScriptedChatClient(script))

    events = await run_turn(harness, "pull it nine times")

    final = events[-1]
    assert isinstance(final, FinalEvent) and final.outcome == "tool_limit"
    assert len(harness.session.datasets) == 8
    assert unpaired_calls(harness) == []
    follow_up = await run_turn(harness, "thanks")
    assert isinstance(follow_up[-1], FinalEvent) and follow_up[-1].outcome == "ok"


async def test_unknown_names_past_the_budget_end_the_turn_without_another_model_call() -> None:
    client = ScriptedChatClient([calls(*[Call("search_web", {"q": "x"})] * 9), text(REPLY)])
    harness = build_agent(client)

    events = await run_turn(harness, "search the web")

    final = events[-1]
    assert isinstance(final, FinalEvent) and final.outcome == "tool_limit"
    assert len(client.calls) == 1  # no model call after the budget ran out
    assert unpaired_calls(harness) == []


async def test_many_tool_calls_stop_speaking_status_lines_at_the_budget() -> None:
    # 8 successful fetches of a long Indicator name: the later lines are shown, not spoken.
    long_name = Call("fetch_data", {"indicator": "10-year treasury yield"})
    harness = build_agent(ScriptedChatClient([calls(*[long_name] * 8), text(REPLY)]))

    events = await run_turn(harness, "pull it eight times")

    statuses = [e for e in events if isinstance(e, StatusEvent)]
    assert len(statuses) == 16  # every line is still shown in the chat
    spoken = sum(len(e.text.split()) for e in statuses if e.audio is not None)
    assert spoken <= MAX_STATUS_WORDS
    final = events[-1]
    assert isinstance(final, FinalEvent) and final.outcome == "ok"
    assert spoken + len(final.spoken_text.split()) <= MAX_SPOKEN_WORDS


async def test_status_lines_count_toward_the_spoken_word_limit() -> None:
    long_reply = (
        "<display>Boss, here is a long summary.</display><spoken>"
        + " ".join(["Inflation rose, Boss."] * 30)
        + "</spoken>"
    )
    harness = build_agent(ScriptedChatClient([calls(FETCH), text(long_reply)]))

    events = await run_turn(harness, "please pull inflation")

    spoken_words = sum(
        len(e.text.split()) for e in events if isinstance(e, StatusEvent) and e.audio is not None
    )
    final = events[-1]
    assert isinstance(final, FinalEvent)
    spoken_words += len(final.spoken_text.split())
    assert spoken_words <= MAX_SPOKEN_WORDS


async def test_fixed_outcome_texts_are_spoken_except_credentials() -> None:
    harness = build_agent(ScriptedChatClient([Raise(RuntimeError("model exploded"))]))
    events = await run_turn(harness, "hi")
    final = events[-1]
    assert isinstance(final, FinalEvent) and final.outcome == "llm_unavailable"
    assert final.spoken_text == final.text and final.audio is not None
