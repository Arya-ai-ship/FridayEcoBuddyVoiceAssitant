"""Tool bodies and their outcomes (Req 5.5, 5.11, 6.2-6.15, 8.4, 9.1-9.3, 10.1-10.9, 11.1-11.10).

Each ``run_*`` coroutine executes one validated Tool call against a Session. It computes
everything first and changes the Session only in its last step (``_store_dataset`` /
``Session.add_chart``), after every call that can fail. A Tool that raises therefore leaves
the Session exactly as it was: no Dataset, chart, FRED-cache entry, or ID counter changes.
``ToolRegistry.run`` (``agent/tools.py``) turns any exception into a failed ``ToolOutcome``.

Tool results are strict JSON (``allow_nan=False``). The only floats are the statistics,
and ``describe`` already reports a non-finite statistic as undefined; ``_finite`` maps any
NaN/infinity that still reaches a result to ``null`` (undefined), never to a ``NaN`` token.
Dates are ISO ``YYYY-MM-DD`` strings, and a missing date is ``null``.
"""

import asyncio
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Final

from friday.agent.templates import DESCRIBE, FETCH, FILL, PLOT, ToolName
from friday.agent.tool_args import DescribeArgs, FetchArgs, FillArgs, PlotArgs
from friday.constants import MAX_MISSING_DATES_LISTED
from friday.data.dataset import Dataset
from friday.data.dates import parse_date_range
from friday.data.fetch import build_dataset, transform
from friday.data.fill import fill
from friday.data.indicators import IndicatorMap
from friday.data.plot import IDS_FIELD, ChartRenderer, build_chart_spec
from friday.data.stats import Stats, describe
from friday.errors import DatasetNotFound, JsonDict, ToolError, ToolValidationError
from friday.events import ChartEvent, DatasetPreviewEvent, StatsTableEvent, TurnEvents
from friday.ports import FredSource
from friday.session import ChartRecord, Session

DATASET_PREFIX: Final = "ds"
"""Dataset_IDs are ``ds-1``, ``ds-2``, ... per Session (Req 6.8)."""
CHART_PREFIX: Final = "chart"
"""Chart IDs are ``chart-1``, ``chart-2``, ... per Session."""
DATASET_FIELD: Final = "dataset_id"
"""The Stats/Fill argument naming the source Dataset."""
NO_DATASET_REASON: Final = "no dataset is loaded; fetch one first"
"""Why an omitted Dataset_ID cannot be resolved (the middleware normally stops earlier)."""

_PENDING_ID: Final = "pending"
"""Placeholder Dataset_ID until the Dataset is stored; real IDs are assigned on success."""


@dataclass(frozen=True, eq=False)
class StatsDisplay:
    """Display payload of a successful Stats call: the Dataset and its statistics."""

    dataset: Dataset
    stats: Stats


Display = Dataset | StatsDisplay | ChartRecord
"""What the browser shows for a successful call: a Preview_Table (fetch, fill), a stats
table, or a chart."""

DisplayEvent = DatasetPreviewEvent | StatsTableEvent | ChartEvent
"""The NDJSON event built from a ``Display``."""


@dataclass(frozen=True, eq=False)
class ToolOutcome:
    """The result of one Tool call; satisfies ``narration.OutcomeView``.

    ``result_json`` is the strict-JSON Tool result returned to the LLM. ``display`` is set
    only on success. ``indicator`` is the canonical Indicator name when known (for status
    lines). ``missing_count`` is the new Dataset's Missing_Value count for a successful
    fetch and ``0`` otherwise (Req 8.2).
    """

    tool: ToolName
    ok: bool
    result_json: str
    display: Display | None = field(default=None, repr=False)  # a chart holds PNG bytes
    indicator: str | None = None
    missing_count: int = 0

    @classmethod
    def success(
        cls,
        tool: ToolName,
        fields: Mapping[str, object],
        display: Display,
        *,
        indicator: str | None = None,
        missing_count: int = 0,
    ) -> "ToolOutcome":
        """A successful call whose result is ``{"ok": true, **fields}``."""
        result_json = to_json({"ok": True, **fields})
        return cls(tool, True, result_json, display, indicator, missing_count)

    @classmethod
    def failure(
        cls, tool: ToolName, error: ToolError, *, indicator: str | None = None
    ) -> "ToolOutcome":
        """A failed call whose result is ``{"ok": false, "tool", "error"}`` (Req 5.11)."""
        return cls(tool, False, to_json(error.to_result(tool)), None, indicator)

    @property
    def result(self) -> JsonDict:
        """A fresh copy of the Tool result as a dict."""
        parsed: JsonDict = json.loads(self.result_json)
        return parsed

    def display_event(self, events: TurnEvents) -> DisplayEvent | None:
        """Build the display event for this outcome, or ``None`` for a failed call."""
        display = self.display
        if isinstance(display, StatsDisplay):
            return events.stats_table(display.dataset, display.stats)
        if isinstance(display, ChartRecord):
            return events.chart(display.chart_id, display.png, display.alt_text)
        if isinstance(display, Dataset):
            return events.dataset_preview(display)
        return None


def to_json(result: Mapping[str, object]) -> str:
    """Serialize a Tool result as strict JSON (a NaN/infinity raises ``ValueError``)."""
    return json.dumps(result, allow_nan=False, ensure_ascii=False, separators=(",", ":"))


# --- Fetch_Tool -------------------------------------------------------------------


async def run_fetch(
    session: Session, args: FetchArgs, fred: FredSource, indicators: IndicatorMap
) -> ToolOutcome:
    """Resolve, check dates, fetch (or reuse the cache), build, transform, store (Req 6)."""
    entry = indicators.resolve(args.indicator)
    start, end = parse_date_range(args.start_date, args.end_date)  # before any FRED call
    key = (entry.series_id, start, end)
    observations = session.fred_cache.get(key)
    if observations is None:
        observations = tuple(await fred.observations(entry.series_id, start, end))
    built = build_dataset(observations, entry, start, end, dataset_id=_PENDING_ID)
    dataset = _store_dataset(session, transform(built, entry.transformation))
    session.fred_cache[key] = observations
    fields = {
        "dataset_id": dataset.dataset_id,
        "indicator": dataset.indicator,
        "series_id": dataset.series_id,
        "row_count": dataset.row_count,
        "first_date": _iso(dataset.dates[0] if dataset.dates else None),
        "last_date": _iso(dataset.dates[-1] if dataset.dates else None),
        "missing_count": dataset.missing_count,
        "missing_dates": _missing_dates_field(dataset),
        "transformation": dataset.transformation,
    }
    return ToolOutcome.success(
        FETCH, fields, dataset, indicator=entry.name, missing_count=dataset.missing_count
    )


def fetch_indicator(args: FetchArgs, indicators: IndicatorMap) -> str | None:
    """The canonical Indicator a fetch names, or ``None`` when it matches none."""
    try:
        return indicators.resolve(args.indicator).name
    except ToolError:
        return None


# --- Stats_Tool -------------------------------------------------------------------


async def run_describe(session: Session, args: DescribeArgs) -> ToolOutcome:
    """Descriptive_Statistics of a Dataset; read-only (Req 9.1-9.3, 9.10, 8.4)."""
    dataset = source_dataset(session, DESCRIBE, args.dataset_id)
    stats = describe(dataset)
    fields = {"dataset_id": dataset.dataset_id, "stats": _stats_fields(stats)}
    display = StatsDisplay(dataset, stats)
    return ToolOutcome.success(DESCRIBE, fields, display, indicator=dataset.indicator)


def _stats_fields(stats: Stats) -> JsonDict:
    """The ``stats`` object of a Stats result (undefined statistics are ``null``)."""
    return {
        "count": stats.count,
        "missing_count": stats.missing_count,
        "mean": _finite(stats.mean),
        "std": _finite(stats.std),
        "min": _finite(stats.min),
        "p25": _finite(stats.p25),
        "median": _finite(stats.median),
        "p75": _finite(stats.p75),
        "max": _finite(stats.max),
        "first_date": _iso(stats.first_date),
        "last_date": _iso(stats.last_date),
    }


# --- Fill_Tool --------------------------------------------------------------------


async def run_fill(session: Session, args: FillArgs) -> ToolOutcome:
    """Fill Missing_Values into a new stored Dataset (Req 10.1-10.9, 8.4)."""
    source = source_dataset(session, FILL, args.dataset_id)
    result = fill(source, args.method, dataset_id=_PENDING_ID)
    dataset = _store_dataset(session, result.dataset)
    fields = {
        "source_dataset_id": source.dataset_id,
        "dataset_id": dataset.dataset_id,
        "method": args.method,
        "row_count": dataset.row_count,
        "filled": result.filled,
        "unfilled": result.unfilled,
        "missing_dates": _missing_dates_field(dataset),
    }
    return ToolOutcome.success(FILL, fields, dataset, indicator=dataset.indicator)


# --- Plot_Tool --------------------------------------------------------------------


async def run_plot(session: Session, args: PlotArgs, renderer: ChartRenderer) -> ToolOutcome:
    """Build the chart spec, render it off the event loop, store the chart (Req 11, 8.4)."""
    dataset_ids = _plot_ids(session, args.dataset_ids)
    spec = build_chart_spec(
        session.datasets, dataset_ids, args.title, args.start_date, args.end_date
    )
    png = await asyncio.to_thread(renderer.render, spec)
    chart = ChartRecord(
        chart_id=session.next_id(CHART_PREFIX),
        png=png,
        dataset_ids=spec.dataset_ids,
        title=spec.title,
        alt_text=spec.alt_text,
        first_date=spec.first_date,
        last_date=spec.last_date,
    )
    session.add_chart(chart)
    fields = {
        "chart_id": chart.chart_id,
        "dataset_ids": list(chart.dataset_ids),
        "first_date": _iso(chart.first_date),
        "last_date": _iso(chart.last_date),
    }
    return ToolOutcome.success(PLOT, fields, chart)


def _plot_ids(session: Session, dataset_ids: Sequence[str] | None) -> Sequence[str]:
    """The requested IDs, or the most recent Dataset's ID when omitted (Req 8.4)."""
    if dataset_ids is not None:
        return dataset_ids
    latest = session.latest_dataset()
    if latest is None:
        raise ToolValidationError.invalid_args(PLOT, IDS_FIELD, NO_DATASET_REASON)
    return (latest.dataset_id,)


# --- Shared helpers ---------------------------------------------------------------


def source_dataset(session: Session, tool: ToolName, dataset_id: str | None) -> Dataset:
    """The named Dataset, or the most recent one when ``dataset_id`` is omitted (Req 8.4).

    Raises ``DatasetNotFound`` for an unknown ID (Req 9.10, 10.9) and an ``invalid_args``
    error naming ``dataset_id`` when it is omitted and the Session has no Dataset.
    """
    if dataset_id is None:
        latest = session.latest_dataset()
        if latest is None:
            raise ToolValidationError.invalid_args(tool, DATASET_FIELD, NO_DATASET_REASON)
        return latest
    dataset = session.datasets.get(dataset_id)
    if dataset is None:
        raise DatasetNotFound(dataset_id)
    return dataset


def _store_dataset(session: Session, draft: Dataset) -> Dataset:
    """Give ``draft`` a new Dataset_ID and store it; the only Session change (Req 6.8)."""
    dataset = replace(draft, dataset_id=session.next_id(DATASET_PREFIX))
    session.add_dataset(dataset)
    return dataset


def _finite(value: float | None) -> float | None:
    """``value`` for the JSON result; NaN/infinity become ``None`` (undefined)."""
    return value if value is not None and math.isfinite(value) else None


def _iso(value: date | None) -> str | None:
    return None if value is None else value.isoformat()


def _missing_dates_field(dataset: Dataset) -> list[str]:
    """ISO dates of ``dataset``'s Missing_Values, capped at ``MAX_MISSING_DATES_LISTED``.

    Lets the model name where the gaps are without reading every row. The cap keeps the
    result bounded; these FRED series carry at most a few gaps in practice.
    """
    gaps = dataset.missing_dates
    return [day.isoformat() for day in gaps[:MAX_MISSING_DATES_LISTED]]
