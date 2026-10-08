"""Unit tests for ``ToolRegistry.run`` and ``ToolRegistry.function_tools``."""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pytest
from agent_framework import AgentSession, FunctionTool

from fakes import FakeFred, FredCall
from friday.agent.tool_args import DescribeArgs, FetchArgs, FillArgs, PlotArgs, ToolArgs
from friday.agent.tool_runs import StatsDisplay, ToolOutcome
from friday.agent.tools import ToolRegistry
from friday.data.dataset import Dataset
from friday.data.fetch import Observation
from friday.data.indicators import default_indicator_map
from friday.data.plot import ChartRenderer, ChartSpec
from friday.errors import ChartRenderError
from friday.events import ChartEvent, DatasetPreviewEvent, StatsTableEvent, TurnEvents
from friday.ports import FredSource
from friday.session import ChartRecord, Session

PNG_MAGIC = b"\x89PNG"


class FailingRenderer(ChartRenderer):
    """Renderer whose every render fails like a broken matplotlib backend."""

    def render(self, spec: ChartSpec) -> bytes:
        raise ChartRenderError()


@dataclass
class BrokenFred:
    """``FredSource`` raising a non-Friday exception whose text must never leak."""

    calls: int = 0

    async def observations(
        self, series_id: str, start: date | None, end: date | None
    ) -> Sequence[Observation]:
        self.calls += 1
        raise RuntimeError("internal detail api_key=do-not-leak")


@dataclass
class Turn:
    """Minimal ``ToolTurn``."""

    session: Session
    outcomes: list[ToolOutcome] = field(default_factory=list[ToolOutcome])


def new_session() -> Session:
    return Session("00000000-0000-4000-8000-000000000000", AgentSession(), last_seen=0.0)


def make_dataset(dataset_id: str, values: Sequence[float], indicator: str = "cpi") -> Dataset:
    dates = tuple(date(2020 + i // 12, i % 12 + 1, 1) for i in range(len(values)))
    return Dataset(dataset_id, indicator, indicator, "CPIAUCSL", dates, np.array(values), None)


def registry(fred: FredSource | None = None, renderer: ChartRenderer | None = None) -> ToolRegistry:
    return ToolRegistry(fred or FakeFred(), renderer or ChartRenderer(), default_indicator_map())


def snapshot(session: Session) -> tuple[object, ...]:
    return (
        dict(session.datasets),
        dict(session.charts),
        dict(session.fred_cache),
        dict(session.counters),
    )


def error_of(outcome: ToolOutcome) -> dict[str, object]:
    result = outcome.result
    assert result["ok"] is False and result["tool"] == outcome.tool and not outcome.ok
    assert outcome.display is None
    error = result["error"]
    assert isinstance(error, dict)
    return error  # pyright: ignore[reportUnknownVariableType]


# --- Fetch_Tool ---------------------------------------------------------------------


async def test_fetch_stores_dataset_and_reports_it() -> None:
    fred, session = FakeFred(), new_session()
    args = FetchArgs(indicator="  CPI ", start_date="2020-01-01", end_date="2020-12-31")
    outcome = await registry(fred).run(session, args)
    assert outcome.ok and outcome.indicator == "cpi" and outcome.missing_count == 0
    assert fred.calls == [FredCall("CPIAUCSL", date(2020, 1, 1), date(2020, 12, 31))]
    assert outcome.result == {
        "ok": True,
        "dataset_id": "ds-1",
        "indicator": "cpi",
        "series_id": "CPIAUCSL",
        "row_count": 12,
        "first_date": "2020-01-01",
        "last_date": "2020-12-01",
        "missing_count": 0,
        "transformation": None,
    }
    assert session.datasets["ds-1"] is outcome.display
    assert isinstance(outcome.display_event(TurnEvents("s")), DatasetPreviewEvent)


async def test_fetch_applies_yoy_and_counts_missing_values() -> None:
    rows = [{"date": f"2020-{m:02d}-01", "value": "." if m == 3 else "100"} for m in range(1, 13)]
    rows += [{"date": f"2021-{m:02d}-01", "value": "110"} for m in range(1, 13)]
    fred = FakeFred(series={"CPIAUCSL": rows})
    outcome = await registry(fred).run(new_session(), FetchArgs(indicator="inflation"))
    assert outcome.ok and outcome.missing_count == 1
    assert outcome.result["row_count"] == 12 and outcome.result["transformation"] == "yoy"


async def test_fred_cache_avoids_a_second_call_for_the_same_request() -> None:
    fred, session = FakeFred(), new_session()
    tools = registry(fred)
    await tools.run(session, FetchArgs(indicator="cpi", start_date="2020-01-01"))
    await tools.run(session, FetchArgs(indicator="inflation", start_date="2020-01-01"))
    assert len(fred.calls) == 1  # cpi and inflation share CPIAUCSL
    assert list(session.datasets) == ["ds-1", "ds-2"]
    await tools.run(session, FetchArgs(indicator="cpi", start_date="2021-01-01"))
    assert len(fred.calls) == 2


@pytest.mark.parametrize(
    ("args", "kind"),
    [
        (FetchArgs(indicator="gdp"), "unknown_indicator"),
        (FetchArgs(indicator="cpi", start_date="2023-02-30"), "invalid_date"),
        (
            FetchArgs(indicator="cpi", start_date="2024-01-01", end_date="2023-01-01"),
            "invalid_date",
        ),
    ],
)
async def test_fetch_rejects_bad_input_before_fred(args: FetchArgs, kind: str) -> None:
    fred, session = FakeFred(), new_session()
    outcome = await registry(fred).run(session, args)
    assert error_of(outcome)["kind"] == kind
    assert fred.calls == [] and snapshot(session) == snapshot(new_session())


async def test_unknown_indicator_lists_supported_names() -> None:
    outcome = await registry().run(new_session(), FetchArgs(indicator="gdp"))
    assert error_of(outcome)["supported"] == list(default_indicator_map().names)


@pytest.mark.parametrize(
    ("fred", "args", "kind"),
    [
        (FakeFred(failure="timeout"), FetchArgs(indicator="cpi"), "timeout"),
        (
            FakeFred(),
            FetchArgs(indicator="inflation", start_date="2024-01-01"),
            "range_too_short_for_yoy",
        ),
        (BrokenFred(), FetchArgs(indicator="cpi"), "tool_failed"),
    ],
)
async def test_fetch_failure_leaves_session_unchanged(
    fred: FredSource, args: FetchArgs, kind: str
) -> None:
    session = new_session()
    before = snapshot(session)
    outcome = await registry(fred).run(session, args)
    error = error_of(outcome)
    assert (
        error["kind"] == kind
        and outcome.indicator == default_indicator_map().resolve(args.indicator).name
    )
    assert snapshot(session) == before
    assert "do-not-leak" not in outcome.result_json


# --- Stats_Tool ---------------------------------------------------------------------


async def test_describe_defaults_to_most_recent_and_is_read_only() -> None:
    session = new_session()
    session.add_dataset(make_dataset("ds-1", [1.0, 2.0]))
    latest = make_dataset("ds-2", [1.0, math.nan, 3.0], indicator="unemployment rate")
    session.add_dataset(latest)
    before = snapshot(session)
    outcome = await registry().run(session, DescribeArgs())
    assert outcome.ok and outcome.missing_count == 0 and snapshot(session) == before
    stats = outcome.result["stats"]
    assert outcome.result["dataset_id"] == "ds-2" and isinstance(stats, dict)
    assert stats.pop("std") == pytest.approx(math.sqrt(2.0))  # pyright: ignore[reportUnknownMemberType]
    assert stats == {
        "count": 2,
        "missing_count": 1,
        "mean": 2.0,
        "min": 1.0,
        "p25": 1.5,
        "median": 2.0,
        "p75": 2.5,
        "max": 3.0,
        "first_date": "2020-01-01",
        "last_date": "2020-03-01",
    }
    assert isinstance(outcome.display, StatsDisplay) and outcome.display.dataset is latest
    assert isinstance(outcome.display_event(TurnEvents("s")), StatsTableEvent)


async def test_describe_reports_undefined_statistics_as_null() -> None:
    session = new_session()
    session.add_dataset(make_dataset("ds-1", [math.inf, 1.0, math.nan]))
    outcome = await registry().run(session, DescribeArgs(dataset_id="ds-1"))
    assert "NaN" not in outcome.result_json and "Infinity" not in outcome.result_json
    stats = outcome.result["stats"]
    assert isinstance(stats, dict) and stats["mean"] is None and stats["count"] == 2


# --- Fill_Tool ----------------------------------------------------------------------


async def test_fill_stores_a_new_dataset_from_the_most_recent_one() -> None:
    session = new_session()
    source = make_dataset("ds-1", [1.0, math.nan, 3.0, math.nan])
    session.add_dataset(source)
    session.counters["ds"] = 1
    outcome = await registry().run(session, FillArgs())
    assert outcome.result == {
        "ok": True,
        "source_dataset_id": "ds-1",
        "dataset_id": "ds-2",
        "method": "forward_fill",
        "row_count": 4,
        "filled": 2,
        "unfilled": 0,
    }
    filled = session.datasets["ds-2"]
    assert filled is outcome.display and filled.derived_from == "ds-1"
    assert session.datasets["ds-1"] is source and source.missing_count == 2


# --- Plot_Tool ----------------------------------------------------------------------


async def test_plot_renders_and_stores_a_chart_of_the_most_recent_dataset() -> None:
    session = new_session()
    session.add_dataset(make_dataset("ds-1", [1.0, math.nan, 3.0]))
    outcome = await registry().run(session, PlotArgs())
    assert outcome.result == {
        "ok": True,
        "chart_id": "chart-1",
        "dataset_ids": ["ds-1"],
        "first_date": "2020-01-01",
        "last_date": "2020-03-01",
    }
    chart = session.charts["chart-1"]
    assert isinstance(chart, ChartRecord) and chart is outcome.display
    assert chart.png.startswith(PNG_MAGIC)
    event = outcome.display_event(TurnEvents("s"))
    assert isinstance(event, ChartEvent) and event.alt == chart.alt_text


@pytest.mark.parametrize(
    ("args", "renderer", "kind"),
    [
        (PlotArgs(dataset_ids=("ds-9",)), None, "dataset_not_found"),
        (PlotArgs(start_date="2030-01-01"), None, "invalid_date"),
        (PlotArgs(), FailingRenderer(), "chart_render_failed"),
    ],
)
async def test_plot_failure_produces_no_chart(
    args: PlotArgs, renderer: ChartRenderer | None, kind: str
) -> None:
    session = new_session()
    session.add_dataset(make_dataset("ds-1", [1.0, 2.0]))
    before = snapshot(session)
    outcome = await registry(renderer=renderer).run(session, args)
    assert error_of(outcome)["kind"] == kind and snapshot(session) == before


# --- Shared lookups -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "field_name"),
    [(DescribeArgs(), "dataset_id"), (FillArgs(), "dataset_id"), (PlotArgs(), "dataset_ids")],
)
async def test_omitted_id_without_datasets_names_the_field(args: ToolArgs, field_name: str) -> None:
    error = error_of(await registry().run(new_session(), args))
    assert error["kind"] == "invalid_args" and error["field"] == field_name


@pytest.mark.parametrize("args", [DescribeArgs(dataset_id="ds-7"), FillArgs(dataset_id="ds-7")])
async def test_unknown_dataset_id_is_named(args: ToolArgs) -> None:
    session = new_session()
    session.add_dataset(make_dataset("ds-1", [1.0]))
    before = snapshot(session)
    error = error_of(await registry().run(session, args))
    assert error == {
        "kind": "dataset_not_found",
        "message": "dataset 'ds-7' was not found",
        "dataset_id": "ds-7",
    }
    assert snapshot(session) == before


# --- function_tools -----------------------------------------------------------------


def test_function_tools_wrap_the_four_specs() -> None:
    tools = registry().function_tools(lambda: Turn(new_session()))
    specs = ToolRegistry.specs()
    assert [t.name for t in tools] == [s.name for s in specs]
    for tool, spec in zip(tools, specs, strict=True):
        assert isinstance(tool, FunctionTool)
        assert tool.description == spec.description
        assert tool.parameters() == spec.input_schema
        assert tool.approval_mode == "never_require"


async def test_function_tool_runs_on_the_current_turn_and_records_the_outcome() -> None:
    turn = Turn(new_session())
    fetch, describe_tool, _, _ = registry().function_tools(lambda: turn)
    contents = await fetch.invoke(arguments={"indicator": "cpi"})
    assert json.loads(contents[0].text or "") == turn.outcomes[0].result
    assert list(turn.session.datasets) == ["ds-1"]
    contents = await describe_tool.invoke(arguments={"dataset_id": "nope"})
    assert json.loads(contents[0].text or "")["error"]["kind"] == "dataset_not_found"
    assert [o.ok for o in turn.outcomes] == [True, False]


async def test_function_tool_without_a_turn_returns_an_error_result() -> None:
    def no_turn() -> Turn:
        raise LookupError("no active turn")

    tools = registry().function_tools(no_turn)
    contents = await tools[0].invoke(arguments={"indicator": "cpi"})
    result = json.loads(contents[0].text or "")
    assert result["ok"] is False and result["error"]["kind"] == "tool_failed"
    assert "no active turn" not in (contents[0].text or "")
