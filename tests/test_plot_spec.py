"""Unit tests for ``build_chart_spec`` (Req 11.1-11.3, 11.5-11.9, 14.8)."""

import math
from datetime import date

import numpy as np
import pytest

from friday.constants import MAX_PLOT_SERIES, MAX_TITLE_CHARS
from friday.data.dataset import Dataset
from friday.data.plot import MULTI_Y_LABEL, X_LABEL, build_chart_spec
from friday.errors import DatasetNotFound, DateRangeError, ToolValidationError
from friday.style import NEON_PALETTE

NAN = math.nan


def _ds(dataset_id: str, indicator: str, values: list[float], year: int = 2020) -> Dataset:
    dates = tuple(date(year, month, 1) for month in range(1, len(values) + 1))
    return Dataset(
        dataset_id=dataset_id,
        indicator=indicator,
        value_column=indicator,
        series_id=indicator.upper(),
        dates=dates,
        values=np.array(values, dtype=np.float64),
        transformation=None,
    )


def _session(*datasets: Dataset) -> dict[str, Dataset]:
    return {ds.dataset_id: ds for ds in datasets}


INFLATION = _ds("ds-1", "inflation", [1.0, NAN, 3.0, 4.0, 5.0, 6.0])
UNEMPLOYMENT = _ds("ds-2", "unemployment", [4.0, 4.1, 4.2, NAN, 4.4, 4.5])


def test_palette_covers_the_series_limit() -> None:
    assert len(set(NEON_PALETTE)) >= MAX_PLOT_SERIES


def test_single_series_spec() -> None:
    spec = build_chart_spec(_session(INFLATION), ["ds-1"])
    assert spec.dataset_ids == ("ds-1",)
    (series,) = spec.series
    assert series.label == "inflation"
    assert series.color == NEON_PALETTE[0]
    assert series.dates == INFLATION.dates
    assert np.array_equal(series.values, INFLATION.values, equal_nan=True)
    assert not series.values.flags.writeable
    assert spec.x_label == X_LABEL
    assert spec.y_label == "inflation"
    assert spec.title == "inflation"
    assert not spec.show_legend
    assert spec.alt_text == "Line chart of inflation from 2020-01-01 to 2020-06-01"


def test_multi_series_spec_with_range_and_title() -> None:
    session = _session(INFLATION, UNEMPLOYMENT)
    spec = build_chart_spec(session, ["ds-2", "ds-1"], "Macro", "2020-02-01", "2020-04-01")
    assert spec.dataset_ids == ("ds-2", "ds-1")
    assert [s.color for s in spec.series] == list(NEON_PALETTE[:2])
    assert all(
        s.dates == (date(2020, 2, 1), date(2020, 3, 1), date(2020, 4, 1)) for s in spec.series
    )
    assert np.isnan(spec.series[1].values[0])  # Missing stays NaN (a gap)
    assert spec.y_label == MULTI_Y_LABEL
    assert spec.title == "Macro"
    assert spec.show_legend
    assert (spec.first_date, spec.last_date) == (date(2020, 2, 1), date(2020, 4, 1))
    assert "unemployment" in spec.alt_text and "inflation" in spec.alt_text
    assert "2020-02-01" in spec.alt_text and "2020-04-01" in spec.alt_text


def test_default_title_lists_names_and_repeated_names_get_distinct_labels() -> None:
    filled = _ds("ds-3", "inflation", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    spec = build_chart_spec(_session(INFLATION, UNEMPLOYMENT, filled), ["ds-1", "ds-2", "ds-3"])
    assert spec.title == "inflation, unemployment"
    assert [s.label for s in spec.series] == [
        "inflation (ds-1)",
        "unemployment",
        "inflation (ds-3)",
    ]
    assert len({s.color for s in spec.series}) == 3


def test_plotted_bounds_skip_edge_missing_values() -> None:
    edges = _ds("ds-1", "inflation", [NAN, 2.0, 3.0, NAN])
    spec = build_chart_spec(_session(edges), ["ds-1"])
    assert (spec.first_date, spec.last_date) == (date(2020, 2, 1), date(2020, 3, 1))


def test_inputs_are_unchanged() -> None:
    before = INFLATION.values.copy()
    build_chart_spec(_session(INFLATION), ["ds-1"], start="2020-03-01")
    assert np.array_equal(INFLATION.values, before, equal_nan=True)
    assert INFLATION.row_count == 6


@pytest.mark.parametrize(
    ("ids", "title", "field"),
    [
        ([], None, "dataset_ids"),
        (["ds-1", "ds-1"], None, "dataset_ids"),
        ([f"ds-{i}" for i in range(MAX_PLOT_SERIES + 1)], None, "dataset_ids"),
        (["ds-1"], "x" * (MAX_TITLE_CHARS + 1), "title"),
    ],
)
def test_invalid_arguments_name_the_field(ids: list[str], title: str | None, field: str) -> None:
    with pytest.raises(ToolValidationError) as info:
        build_chart_spec(_session(INFLATION), ids, title)
    assert info.value.field == field
    assert info.value.kind == "invalid_args"


def test_title_at_the_limit_is_accepted() -> None:
    title = "x" * MAX_TITLE_CHARS
    assert build_chart_spec(_session(INFLATION), ["ds-1"], title).title == title


def test_unknown_id_is_named() -> None:
    with pytest.raises(DatasetNotFound) as info:
        build_chart_spec(_session(INFLATION), ["ds-1", "ds-9"])
    assert info.value.dataset_id == "ds-9"


@pytest.mark.parametrize(
    ("start", "end"),
    [("2020-1-01", None), (None, "2020-02-30"), ("2020-05-01", "2020-02-01")],
)
def test_bad_date_ranges_are_rejected(start: str | None, end: str | None) -> None:
    with pytest.raises(DateRangeError):
        build_chart_spec(_session(INFLATION), ["ds-1"], start=start, end=end)


def test_dataset_without_points_in_range_is_named() -> None:
    session = _session(INFLATION, UNEMPLOYMENT)
    with pytest.raises(DateRangeError) as info:
        build_chart_spec(session, ["ds-1", "ds-2"], start="2020-04-01", end="2020-04-01")
    assert "ds-2" in info.value.message
    assert info.value.kind == "invalid_date"


def test_all_missing_dataset_is_rejected() -> None:
    empty = _ds("ds-1", "inflation", [NAN, NAN])
    with pytest.raises(DateRangeError, match="ds-1"):
        build_chart_spec(_session(empty), ["ds-1"])
