"""Property test for date validation before any FRED call (task 10.3, Req 6.12)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fakes import FakeFred
from friday.agent.tool_args import FetchArgs
from friday.agent.tools import ToolRegistry
from friday.data.indicators import default_indicator_map
from friday.session import Session

pytestmark = pytest.mark.asyncio


def _registry(fred: FakeFred) -> ToolRegistry:
    from friday.data.plot import ChartRenderer

    return ToolRegistry(fred, ChartRenderer(), default_indicator_map())


def _session() -> Session:
    from agent_framework import AgentSession

    return Session("00000000-0000-4000-8000-000000000000", AgentSession(), last_seen=0.0)


@dataclass(frozen=True)
class BadRange:
    """A fetch date range that is invalid, with the piece of input that is wrong."""

    start: str | None
    end: str | None
    bad_input: str


def _malformed() -> st.SearchStrategy[str]:
    """Strings with the YYYY-MM-DD shape that are not real calendar dates.

    The ``FetchArgs`` model already rejects the wrong shape at construction, so the
    Tool's own date check only ever sees shape-valid, non-calendar strings; those are
    what this property exercises.
    """
    return st.sampled_from(
        [
            "2024-13-01",  # month 13
            "2024-00-10",  # month 0
            "2024-02-30",  # Feb 30 is not a real day
            "2024-04-31",  # April has 30 days
            "2023-02-29",  # 2023 is not a leap year
            "0000-01-01",  # year 0 is not a valid date
            "2024-11-31",  # November has 30 days
        ]
    )


@st.composite
def bad_ranges(draw: st.DrawFn) -> BadRange:
    """Either a malformed start/end, or a well-formed start later than end."""
    if draw(st.booleans()):
        bad = draw(_malformed())
        if draw(st.booleans()):
            return BadRange(bad, "2024-12-31", bad)
        return BadRange("2024-01-01", bad, bad)
    # start > end, both real dates
    start = draw(st.dates(min_value=date(1950, 1, 2), max_value=date(2100, 1, 1)))
    end = draw(st.dates(min_value=date(1950, 1, 1), max_value=start - timedelta(days=1)))
    return BadRange(start.isoformat(), end.isoformat(), start.isoformat())


@settings(max_examples=200, deadline=None)
@given(case=bad_ranges(), indicator=st.sampled_from(sorted(default_indicator_map().names)))
async def test_invalid_dates_are_rejected_before_any_fred_call(
    case: BadRange, indicator: str
) -> None:
    """Feature: friday-voice-data-assistant, Property 4: Invalid dates are rejected before
    any FRED call.

    For any start/end pair where a supplied value is not a real YYYY-MM-DD calendar date,
    or start is later than end, ``fetch_data`` returns an error naming the invalid input
    and the fake FRED client records zero calls.

    **Validates: Requirements 6.12**
    """
    fred = FakeFred()
    args = FetchArgs(indicator=indicator, start_date=case.start, end_date=case.end)
    outcome = await _registry(fred).run(_session(), args)

    assert not outcome.ok
    error = outcome.result["error"]
    assert isinstance(error, dict)
    assert error["kind"] == "invalid_date"
    assert case.bad_input in outcome.result_json
    assert fred.calls == []
