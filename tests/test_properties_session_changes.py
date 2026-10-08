"""Property test: tool calls change the Session only through successful fetch/fill
(task 10.4, Req 5.5, 5.11, 6.8, 6.11, 7.7, 9.3, 9.10, 10.1, 10.9, 11.4)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest
from agent_framework import AgentSession
from hypothesis import given, settings
from hypothesis import strategies as st

from fakes import FakeFred
from friday.agent.tool_args import DescribeArgs, FetchArgs, FillArgs, PlotArgs, ToolArgs
from friday.agent.tools import ToolRegistry
from friday.data.dataset import Dataset
from friday.data.indicators import default_indicator_map
from friday.data.plot import ChartRenderer, ChartSpec
from friday.errors import ChartRenderError, FredKind
from friday.session import Session

pytestmark = pytest.mark.asyncio

_NAMES = sorted(default_indicator_map().names)


class FailingRenderer(ChartRenderer):
    """A renderer whose every render fails, to inject a Tool exception on plot."""

    def render(self, spec: ChartSpec) -> bytes:
        raise ChartRenderError()


@dataclass(frozen=True)
class DatasetSnapshot:
    """A Dataset's identity and contents, for the unchanged-datasets invariant."""

    dataset_id: str
    dates: tuple[date, ...]
    values: bytes

    @classmethod
    def of(cls, ds: Dataset) -> DatasetSnapshot:
        return cls(ds.dataset_id, ds.dates, ds.values.tobytes())


def _snapshot(session: Session) -> dict[str, DatasetSnapshot]:
    return {did: DatasetSnapshot.of(ds) for did, ds in session.datasets.items()}


def _tool_args() -> st.SearchStrategy[ToolArgs]:
    """A random Tool call: valid or invalid fetch/stats/fill/plot."""
    fetch = st.builds(
        FetchArgs,
        indicator=st.sampled_from([*_NAMES, "gdp", "nonsense"]),  # some unknown
        start_date=st.sampled_from([None, "2020-01-01", "2024-13-01"]),  # some malformed
        end_date=st.sampled_from([None, "2020-12-31"]),
    )
    describe = st.builds(DescribeArgs, dataset_id=st.sampled_from([None, "ds-1", "ds-99"]))
    fill = st.builds(
        FillArgs,
        dataset_id=st.sampled_from([None, "ds-1", "ds-99"]),
        method=st.sampled_from(["forward_fill", "linear_interpolation"]),
    )
    plot = st.builds(
        PlotArgs,
        dataset_ids=st.sampled_from([None, ["ds-1"], ["ds-99"]]),
    )
    return st.one_of(fetch, describe, fill, plot)


@settings(max_examples=150, deadline=None)
@given(
    calls=st.lists(_tool_args(), min_size=1, max_size=6),
    fred_failure=st.sampled_from([None, "http_error", "timeout", "no_observations"]),
    plot_fails=st.booleans(),
)
async def test_session_changes_only_through_successful_fetch_or_fill(
    calls: list[ToolArgs], fred_failure: FredKind | None, plot_fails: bool
) -> None:
    """Feature: friday-voice-data-assistant, Property 5: Tool calls change the Session only
    through successful fetch/fill.

    For any sequence of Tool invocations (unknown IDs, invalid args, injected FRED failures,
    injected Tool exceptions): each successful fetch/fill adds exactly one Dataset under a
    new ID; every other invocation adds no Dataset; and every pre-existing Dataset keeps its
    ID, dates, and values.

    **Validates: Requirements 5.5, 5.11, 6.8, 6.11, 7.7, 9.3, 9.10, 10.1, 10.9, 11.4**
    """
    fred = FakeFred(failure=fred_failure)
    renderer: ChartRenderer = FailingRenderer() if plot_fails else ChartRenderer()
    registry = ToolRegistry(fred, renderer, default_indicator_map())
    session = Session("00000000-0000-4000-8000-000000000000", AgentSession(), last_seen=0.0)

    for args in calls:
        before = _snapshot(session)
        before_ids = set(before)

        outcome = await registry.run(session, args)
        after = _snapshot(session)

        # Pre-existing Datasets are never mutated or dropped (Req 6.11, 7.7, 9.3, 9.10).
        for did, snap in before.items():
            assert after[did] == snap

        added = set(after) - before_ids
        if outcome.ok and isinstance(args, FetchArgs | FillArgs):
            # Exactly one new Dataset, under a never-before-used ID (Req 6.8, 10.1).
            assert len(added) == 1
            new_id = added.pop()
            assert new_id not in before_ids
        else:
            # Stats, plot, and every failed call add no Dataset (Req 5.5, 5.11, 9.3, 10.9).
            assert added == set()
