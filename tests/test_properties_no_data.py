"""Property test: analysis requests with no data run no Tool (task 12.4, Req 8.6)."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agent_fixtures import build_agent, run_turn
from fakes import Call, ScriptedChatClient, calls
from friday.agent.templates import NO_DATA_TEXT
from friday.events import FinalEvent

pytestmark = pytest.mark.asyncio

ANALYSIS_TOOLS = ["describe_data", "fill_missing", "plot_data"]


def _args_for(tool: str) -> st.SearchStrategy[dict[str, object]]:
    if tool == "fill_missing":
        return st.fixed_dictionaries(
            {"method": st.sampled_from(["forward_fill", "linear_interpolation"])}
        )
    return st.just({})


@settings(
    max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(data=st.data(), tool=st.sampled_from(ANALYSIS_TOOLS))
async def test_analysis_requests_with_no_data_run_no_tool(data: st.DataObject, tool: str) -> None:
    """Feature: friday-voice-data-assistant, Property 7: Analysis requests with no data run
    no Tool.

    For any describe/fill/plot call requested while the Session has no Datasets: no Tool
    runs, the Session is unchanged, and the turn ends with ``final.outcome == "no_data"``.
    The text and Spoken_Text both ask which Indicator to fetch.

    **Validates: Requirements 8.6**
    """
    args = data.draw(_args_for(tool))
    harness = build_agent(ScriptedChatClient([calls(Call(tool, args))]))

    events = await run_turn(harness, "analyze it")

    final = events[-1]
    assert isinstance(final, FinalEvent)
    assert final.outcome == "no_data"
    # Both the text and the Spoken_Text ask which Indicator to fetch (Property 7).
    assert final.text == NO_DATA_TEXT
    assert final.spoken_text == NO_DATA_TEXT and final.audio is not None
    assert not harness.session.datasets and not harness.session.charts
    # No display events were produced (no Tool ran).
    assert not any(e.type in {"dataset_preview", "stats_table", "chart"} for e in events)
    assert harness.fred.calls == []
