"""Property test for omitted dataset references (task 10.5, Req 8.4)."""

from __future__ import annotations

import pytest
from agent_framework import AgentSession
from hypothesis import given, settings
from hypothesis import strategies as st

from fakes import FakeFred
from friday.agent.tool_args import DescribeArgs, PlotArgs
from friday.agent.tool_runs import StatsDisplay
from friday.agent.tools import ToolRegistry
from friday.data.indicators import default_indicator_map
from friday.data.plot import ChartRenderer
from friday.session import ChartRecord, Session
from strategies import datasets

pytestmark = pytest.mark.asyncio


def _registry() -> ToolRegistry:
    return ToolRegistry(FakeFred(), ChartRenderer(), default_indicator_map())


def _session_with(count: int, data: st.DataObject) -> Session:
    session = Session("00000000-0000-4000-8000-000000000000", AgentSession(), last_seen=0.0)
    for i in range(count):
        from dataclasses import replace

        ds = data.draw(datasets(min_rows=1, max_rows=24, max_magnitude=1e6))
        # Ensure at least one non-missing point so plot has something to draw.
        import numpy as np

        values = ds.values.copy()
        if not np.any(~np.isnan(values)):
            values[0] = 1.0
        session.add_dataset(
            replace(ds, dataset_id=f"ds-{i + 1}", values=values, transformation=None)
        )
    return session


@settings(max_examples=100, deadline=None)
@given(data=st.data(), count=st.integers(1, 5))
async def test_describe_without_id_uses_the_most_recent_dataset(
    data: st.DataObject, count: int
) -> None:
    """Feature: friday-voice-data-assistant, Property 6: Omitted dataset references resolve
    to the most recent Dataset.

    For any Session with at least one Dataset, ``describe_data`` with no ``dataset_id``
    operates on the Dataset inserted most recently.

    **Validates: Requirements 8.4**
    """
    session = _session_with(count, data)
    latest = session.latest_dataset()
    assert latest is not None

    outcome = await _registry().run(session, DescribeArgs())
    assert outcome.ok
    assert isinstance(outcome.display, StatsDisplay)
    assert outcome.display.dataset.dataset_id == latest.dataset_id


@settings(max_examples=100, deadline=None)
@given(data=st.data(), count=st.integers(1, 5))
async def test_plot_without_ids_uses_the_most_recent_dataset(
    data: st.DataObject, count: int
) -> None:
    """Feature: friday-voice-data-assistant, Property 6: Omitted dataset references resolve
    to the most recent Dataset.

    For any Session with at least one Dataset, ``plot_data`` with no ``dataset_ids``
    operates on the Dataset inserted most recently.

    **Validates: Requirements 8.4**
    """
    session = _session_with(count, data)
    latest = session.latest_dataset()
    assert latest is not None

    outcome = await _registry().run(session, PlotArgs())
    assert outcome.ok
    assert isinstance(outcome.display, ChartRecord)
    # The chart's single series must be the most recent Dataset.
    assert latest.dataset_id in outcome.display.dataset_ids
    assert len(outcome.display.dataset_ids) == 1
