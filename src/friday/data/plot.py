"""Plot_Tool chart specs and rendering (Req 11.1-11.10, 14.7, 14.8).

``build_chart_spec`` validates the ``plot_data`` arguments against the Session's Datasets
and produces an immutable ``ChartSpec``: one series per requested Dataset, limited to the
date range, with a Neon_Accent color, labels, title, legend flag, and alt text. Spec
building is pure; ``ChartRenderer`` turns a spec into PNG bytes with the fixed style from
``friday.style``, using matplotlib's OO API on the Agg canvas (never ``pyplot``).
"""

import io
import threading
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Final

import numpy as np

from friday.constants import MAX_PLOT_SERIES, MAX_TITLE_CHARS
from friday.data.dataset import Dataset, FloatArray
from friday.data.dates import END_FIELD, START_FIELD, parse_date_range
from friday.errors import ChartRenderError, DatasetNotFound, DateRangeError, ToolValidationError
from friday.style import (
    CHART_BG,
    CHART_DPI,
    CHART_FORMAT,
    CHART_GRID,
    CHART_HEIGHT_IN,
    CHART_LINE_WIDTH,
    CHART_TEXT,
    CHART_WIDTH_IN,
    NEON_PALETTE,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

PLOT_TOOL: Final = "plot_data"
"""LLM-facing name of the Plot_Tool, used in validation errors."""
IDS_FIELD: Final = "dataset_ids"
"""Tool argument naming the Datasets to plot."""
TITLE_FIELD: Final = "title"
"""Tool argument holding the optional chart title."""

X_LABEL: Final = "Date"
"""Horizontal axis label (Req 11.2)."""
MULTI_Y_LABEL: Final = "Value"
"""Vertical axis label when more than one series is plotted (Req 11.2)."""
NAME_SEPARATOR: Final = ", "
"""Joins Indicator names in the default title and the alt text."""


@dataclass(frozen=True, eq=False)
class ChartSeries:
    """One line series: in-range dates and values (NaN == Missing, drawn as a gap)."""

    dataset_id: str
    indicator: str
    label: str
    color: str
    dates: tuple[date, ...]
    values: FloatArray

    def __post_init__(self) -> None:
        """Store ``values`` as a read-only float64 copy matching ``dates``."""
        values = np.array(self.values, dtype=np.float64, copy=True)
        if values.ndim != 1 or values.shape[0] != len(self.dates):
            raise ValueError("series values must be 1-D and match the dates")
        values.setflags(write=False)
        object.__setattr__(self, "dates", tuple(self.dates))
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, eq=False)
class ChartSpec:
    """Everything the renderer needs for one single-axes line chart."""

    series: tuple[ChartSeries, ...]
    title: str
    x_label: str
    y_label: str
    show_legend: bool
    alt_text: str
    first_date: date
    """Earliest plotted (non-missing) date across all series."""
    last_date: date
    """Latest plotted (non-missing) date across all series."""

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        """The plotted Dataset_IDs, in request order."""
        return tuple(s.dataset_id for s in self.series)


def build_chart_spec(
    available: Mapping[str, Dataset],
    dataset_ids: Sequence[str],
    title: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> ChartSpec:
    """Validate ``plot_data`` arguments and build the ``ChartSpec`` (Req 11.1-11.9, 14.8).

    ``available`` maps the Session's Dataset_IDs to Datasets; ``dataset_ids`` is the
    request (resolving an omitted list to the most recent Dataset is the caller's job).
    A ``None`` or blank ``title`` means the default title listing the Indicator names.
    ``start``/``end`` are optional ``YYYY-MM-DD`` bounds, inclusive.

    Raises ``ToolValidationError`` (field ``dataset_ids`` or ``title``) for an empty,
    oversized, or duplicated ID list or an over-long title; ``DatasetNotFound`` for an
    unknown ID; ``DateRangeError`` for a bad range or a Dataset with no non-missing point
    in range. Input Datasets are never modified.
    """
    datasets = _resolve_datasets(available, dataset_ids)
    _check_title(title)
    start_date, end_date = parse_date_range(start, end)
    series = tuple(
        _series(ds, index, start_date, end_date, _label(ds, datasets))
        for index, ds in enumerate(datasets)
    )
    first, last = _plotted_bounds(series)
    names = _unique_names(datasets)
    return ChartSpec(
        series=series,
        title=title if title is not None and title.strip() else NAME_SEPARATOR.join(names),
        x_label=X_LABEL,
        y_label=datasets[0].indicator if len(datasets) == 1 else MULTI_Y_LABEL,
        show_legend=len(datasets) > 1,
        alt_text=alt_text(names, first, last),
        first_date=first,
        last_date=last,
    )


def alt_text(names: Sequence[str], first: date, last: date) -> str:
    """Chart alt text listing the Indicator names and the plotted date span (Req 11.6)."""
    joined = NAME_SEPARATOR.join(names)
    return f"Line chart of {joined} from {first.isoformat()} to {last.isoformat()}"


def _resolve_datasets(
    available: Mapping[str, Dataset], dataset_ids: Sequence[str]
) -> tuple[Dataset, ...]:
    """Check the ID list (1-5, distinct, known) and look each ID up (Req 11.1, 11.8)."""
    count = len(dataset_ids)
    if count == 0:
        raise _invalid(IDS_FIELD, f"must list 1 to {MAX_PLOT_SERIES} dataset IDs, got none")
    if count > MAX_PLOT_SERIES:
        raise _invalid(IDS_FIELD, f"at most {MAX_PLOT_SERIES} dataset IDs allowed, got {count}")
    seen: set[str] = set()
    for dataset_id in dataset_ids:
        if dataset_id in seen:
            raise _invalid(IDS_FIELD, f"dataset '{dataset_id}' is listed more than once")
        seen.add(dataset_id)
    missing = next((i for i in dataset_ids if i not in available), None)
    if missing is not None:
        raise DatasetNotFound(missing)
    return tuple(available[i] for i in dataset_ids)


def _check_title(title: str | None) -> None:
    """Reject a title longer than ``MAX_TITLE_CHARS`` (Req 11.4, 11.8)."""
    if title is not None and len(title) > MAX_TITLE_CHARS:
        raise _invalid(
            TITLE_FIELD, f"must be at most {MAX_TITLE_CHARS} characters, got {len(title)}"
        )


def _invalid(field: str, reason: str) -> ToolValidationError:
    return ToolValidationError.invalid_args(PLOT_TOOL, field, reason)


def _series(
    ds: Dataset, index: int, start: date | None, end: date | None, label: str
) -> ChartSeries:
    """Slice ``ds`` to ``[start, end]`` (dates are ascending) and color it by ``index``."""
    lo = bisect_left(ds.dates, start) if start is not None else 0
    hi = bisect_right(ds.dates, end) if end is not None else ds.row_count
    values = ds.values[lo:hi]
    if not np.any(~np.isnan(values)):
        raise _empty_range(ds, start, end)
    return ChartSeries(
        dataset_id=ds.dataset_id,
        indicator=ds.indicator,
        label=label,
        color=NEON_PALETTE[index],
        dates=ds.dates[lo:hi],
        values=values,
    )


def _empty_range(ds: Dataset, start: date | None, end: date | None) -> DateRangeError:
    """Error naming a Dataset with no non-missing observation in range (Req 11.9)."""
    if start is not None and end is not None:
        span = f"between {start.isoformat()} and {end.isoformat()}"
    elif start is not None:
        span = f"on or after {start.isoformat()}"
    elif end is not None:
        span = f"on or before {end.isoformat()}"
    else:
        span = "at all"
    field = START_FIELD if start is not None else END_FIELD if end is not None else IDS_FIELD
    message = (
        f"invalid date range: dataset '{ds.dataset_id}' ({ds.indicator}) "
        f"has no non-missing observations {span}"
    )
    return DateRangeError(field, message)


def _label(ds: Dataset, datasets: Sequence[Dataset]) -> str:
    """Legend label: the Indicator name, plus the Dataset_ID when names repeat."""
    shared = sum(1 for other in datasets if other.indicator == ds.indicator) > 1
    return f"{ds.indicator} ({ds.dataset_id})" if shared else ds.indicator


def _unique_names(datasets: Sequence[Dataset]) -> tuple[str, ...]:
    """Indicator names in request order, each listed once."""
    return tuple(dict.fromkeys(ds.indicator for ds in datasets))


def _plotted_bounds(series: Sequence[ChartSeries]) -> tuple[date, date]:
    """First and last non-missing dates across ``series`` (each has at least one)."""
    firsts: list[date] = []
    lasts: list[date] = []
    for s in series:
        present = np.flatnonzero(~np.isnan(s.values))
        firsts.append(s.dates[int(present[0])])
        lasts.append(s.dates[int(present[-1])])
    return min(firsts), max(lasts)


class ChartRenderer:
    """Renders a ``ChartSpec`` to PNG bytes with the fixed chart style (Req 11.4, 14.7).

    One instance is created in ``wiring.py``; its lock serializes renders. ``render`` is
    blocking, so async callers run it via ``asyncio.to_thread``. matplotlib is imported
    on first use, so importing this module stays cheap.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def render(self, spec: ChartSpec) -> bytes:
        """Draw ``spec`` and return the PNG bytes; any failure is a ``ChartRenderError``."""
        try:
            with self._lock:
                return _render_png(spec)
        except ChartRenderError:
            raise
        except Exception as exc:
            raise ChartRenderError() from exc


def _render_png(spec: ChartSpec) -> bytes:
    """Build a fresh Agg-backed figure for ``spec`` and encode it (Req 11.1-11.4, 11.7)."""
    figure = _new_figure()
    axes = figure.add_subplot()
    _style_axes(axes)
    for series in spec.series:
        # NaN values stay in the data, so matplotlib breaks the line there (Req 11.7).
        axes.plot(
            np.array(series.dates, dtype="datetime64[D]"),
            series.values,
            color=series.color,
            linewidth=CHART_LINE_WIDTH,
            label=series.label,
        )
    axes.set_title(spec.title, color=CHART_TEXT)
    axes.set_xlabel(spec.x_label, color=CHART_TEXT)
    axes.set_ylabel(spec.y_label, color=CHART_TEXT)
    if spec.show_legend:
        _add_legend(axes)
    buffer = io.BytesIO()
    figure.savefig(buffer, format=CHART_FORMAT, dpi=CHART_DPI, facecolor=CHART_BG)
    return buffer.getvalue()


def _new_figure() -> "Figure":
    """A standalone figure on an Agg canvas: no pyplot, no backend lookup, no globals."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(
        figsize=(CHART_WIDTH_IN, CHART_HEIGHT_IN),
        dpi=CHART_DPI,
        facecolor=CHART_BG,
        layout="constrained",
    )
    FigureCanvasAgg(figure)
    return figure


def _style_axes(axes: "Axes") -> None:
    """Dark background, light axes/ticks, subtle grid, concise date ticks (Req 14.7)."""
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

    axes.set_facecolor(CHART_BG)
    for spine in axes.spines.values():
        spine.set_color(CHART_TEXT)
    axes.tick_params(colors=CHART_TEXT, labelcolor=CHART_TEXT)
    axes.grid(visible=True, color=CHART_GRID)
    axes.set_axisbelow(True)
    locator = AutoDateLocator()
    axes.xaxis.set_major_locator(locator)
    formatter = ConciseDateFormatter(locator)
    axes.xaxis.set_major_formatter(formatter)


def _add_legend(axes: "Axes") -> None:
    """One legend entry per series, light text on the chart background (Req 11.3)."""
    axes.legend(facecolor=CHART_BG, edgecolor=CHART_GRID, labelcolor=CHART_TEXT)
