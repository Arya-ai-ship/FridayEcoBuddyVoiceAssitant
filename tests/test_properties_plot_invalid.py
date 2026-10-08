"""Property test for invalid plot arguments (task 10.6, Req 11.8, 11.9).

The real middleware validates raw arguments with ``ToolRegistry.validate`` before the
Tool runs, so shape problems (empty list, over 5 IDs, title over 100 chars, malformed
date) surface as a ``ToolValidationError`` and the semantic ones (unknown ID, duplicate,
start > end, no non-missing point in range) surface from ``run``. This property drives raw
argument dicts through the same two-step path and asserts an error with no new chart.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from agent_framework import AgentSession
from hypothesis import given, settings
from hypothesis import strategies as st

from fakes import FakeFred
from friday.agent.tools import ToolRegistry
from friday.constants import MAX_PLOT_SERIES, MAX_TITLE_CHARS
from friday.data.dataset import Dataset
from friday.data.indicators import default_indicator_map
from friday.data.plot import ChartRenderer
from friday.errors import ToolError
from friday.session import Session

pytestmark = pytest.mark.asyncio

PLOT = "plot_data"


def _registry() -> ToolRegistry:
    return ToolRegistry(FakeFred(), ChartRenderer(), default_indicator_map())


def _dataset(dataset_id: str, *, missing: bool = False) -> Dataset:
    dates = tuple(date(2024, m, 1) for m in range(1, 4))
    values = np.array([np.nan, np.nan, np.nan] if missing else [1.0, 2.0, 3.0])
    return Dataset(dataset_id, "cpi", "cpi", "CPIAUCSL", dates, values, None)


def _session(ids: list[str], missing_ids: set[str] | None = None) -> Session:
    session = Session("00000000-0000-4000-8000-000000000000", AgentSession(), last_seen=0.0)
    for i in ids:
        session.add_dataset(_dataset(i, missing=i in (missing_ids or set())))
    return session


def _cases() -> st.SearchStrategy[tuple[dict[str, object], Session]]:
    base = ["ds-1", "ds-2"]
    many = [f"ds-{i}" for i in range(1, MAX_PLOT_SERIES + 2)]
    return st.sampled_from(
        [
            ({"dataset_ids": []}, _session(base)),  # empty list
            ({"dataset_ids": ["ds-9"]}, _session(base)),  # unknown ID
            ({"dataset_ids": ["ds-1", "ds-1"]}, _session(base)),  # duplicate
            ({"dataset_ids": many}, _session(many)),  # more than 5
            ({"dataset_ids": ["ds-1"], "title": "x" * (MAX_TITLE_CHARS + 1)}, _session(base)),
            ({"dataset_ids": ["ds-1"], "start_date": "2024-13-01"}, _session(base)),  # bad date
            (
                {"dataset_ids": ["ds-1"], "start_date": "2024-12-31", "end_date": "2024-01-01"},
                _session(base),
            ),  # start > end
            ({"dataset_ids": ["ds-1"]}, _session(["ds-1"], {"ds-1"})),  # no point in range
        ]
    )


@settings(max_examples=100, deadline=None)
@given(case=_cases())
async def test_invalid_plot_arguments_produce_errors_and_no_chart(
    case: tuple[dict[str, object], Session],
) -> None:
    """Feature: friday-voice-data-assistant, Property 17: Invalid plot arguments produce
    errors and no chart.

    For any ``plot_data`` call with an empty, unknown, duplicate, over-limit, over-long
    title, malformed date, inverted range, or empty-in-range problem, the Plot_Tool returns
    an error identifying the problem and the Session's chart count is unchanged.

    **Validates: Requirements 11.8, 11.9**
    """
    raw, session = case
    registry = _registry()
    before = len(session.charts)

    # Step 1: shape validation (as the middleware does first).
    try:
        args = registry.validate(PLOT, raw)
    except ToolError as exc:
        assert str(exc)  # the error identifies the problem
        assert len(session.charts) == before
        return

    # Step 2: the Tool runs and returns a failed outcome for semantic problems.
    outcome = await registry.run(session, args)
    assert not outcome.ok
    error = outcome.result["error"]
    assert isinstance(error, dict)
    assert isinstance(error.get("message", ""), str) and error["message"]
    assert len(session.charts) == before
