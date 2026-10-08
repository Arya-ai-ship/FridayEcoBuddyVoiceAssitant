"""Unit tests for ``data/dataset.py`` and the shared strategies (task 4.1).

Properties 8 (preview) and 9 (CSV round-trip) have their own property-test modules.
"""

import math
from datetime import date

import numpy as np
import pytest
from hypothesis import given, settings

from friday.constants import PREVIEW_ROWS
from friday.data.dataset import (
    Dataset,
    format_display_value,
    parse_csv,
    preview_rows,
    to_csv,
)
from strategies import datasets, fred_observations

NAN = math.nan


def make_dataset(values: list[float], *, column: str = "inflation") -> Dataset:
    dates = tuple(date(2000 + i // 12, i % 12 + 1, 1) for i in range(len(values)))
    return Dataset(
        dataset_id="ds-1",
        indicator=column,
        value_column=column,
        series_id="CPIAUCSL",
        dates=dates,
        values=np.array(values, dtype=np.float64),
        transformation="yoy",
    )


# --- Dataset invariants ------------------------------------------------------


def test_values_are_a_read_only_float64_copy() -> None:
    source = np.array([1.0, 2.0])
    ds = make_dataset(list(source))
    assert ds.values.dtype == np.float64
    assert not ds.values.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        ds.values[0] = 5.0


def test_construction_does_not_alias_the_callers_array() -> None:
    source = np.array([1.0, 2.0])
    ds = Dataset(
        "ds-1", "cpi", "cpi", "CPIAUCSL", (date(2000, 1, 1), date(2000, 2, 1)), source, None
    )
    source[0] = 99.0
    assert ds.values[0] == 1.0


def test_dataset_fields_are_frozen() -> None:
    ds = make_dataset([1.0])
    with pytest.raises(AttributeError):
        ds.indicator = "cpi"  # type: ignore[misc]


def test_counts() -> None:
    ds = make_dataset([1.0, NAN, 3.0, NAN])
    assert ds.row_count == 4
    assert ds.missing_count == 2


@pytest.mark.parametrize(
    "dates",
    [
        (date(2000, 2, 1), date(2000, 1, 1)),  # descending
        (date(2000, 1, 1), date(2000, 1, 1)),  # duplicate
    ],
)
def test_dates_must_be_strictly_ascending(dates: tuple[date, ...]) -> None:
    with pytest.raises(ValueError, match="ascending"):
        Dataset("ds-1", "cpi", "cpi", "CPIAUCSL", dates, np.array([1.0, 2.0]), None)


def test_dates_and_values_must_have_the_same_length() -> None:
    with pytest.raises(ValueError, match="dates but"):
        Dataset("ds-1", "cpi", "cpi", "CPIAUCSL", (date(2000, 1, 1),), np.array([1.0, 2.0]), None)


# --- Preview_Table -----------------------------------------------------------


def test_preview_shows_the_first_ten_rows() -> None:
    ds = make_dataset([float(i) for i in range(25)])
    rows = preview_rows(ds)
    assert len(rows) == PREVIEW_ROWS
    assert rows[0] == ("2000-01-01", "0")
    assert rows[-1] == ("2000-10-01", "9")


def test_preview_shows_every_row_of_a_short_dataset() -> None:
    assert len(preview_rows(make_dataset([1.0, 2.0, 3.0]))) == 3
    assert preview_rows(make_dataset([])) == ()


def test_preview_formats_missing_as_empty_and_rounds_to_four_decimals() -> None:
    rows = preview_rows(make_dataset([9.65634, NAN, 1.5, -0.00001]))
    assert [value for _, value in rows] == ["9.6563", "", "1.5", "0"]


@pytest.mark.parametrize(
    ("value", "text"),
    [(0.0, "0"), (-0.0, "0"), (2.0, "2"), (-1.23456, "-1.2346"), (1e-9, "0"), (NAN, "")],
)
def test_format_display_value(value: float, text: str) -> None:
    assert format_display_value(value) == text


# --- CSV_Export --------------------------------------------------------------


def test_csv_header_rows_and_missing_fields() -> None:
    text = to_csv(make_dataset([9.6563, NAN, 0.1 + 0.2]))
    assert text.splitlines() == [
        "date,inflation",
        "2000-01-01,9.6563",
        "2000-02-01,",
        "2000-03-01,0.30000000000000004",  # repr(float): unrounded
    ]


def test_csv_round_trip_example_keeps_extremes_and_missing() -> None:
    values = [5e-324, -1.7976931348623157e308, -0.0, NAN, 123456789.12345679]
    ds = make_dataset(values, column='odd, "quoted" name')
    table = parse_csv(to_csv(ds))
    assert table.columns == ("date", 'odd, "quoted" name')
    assert table.dates == ds.dates
    assert table.values.tobytes() == ds.values.tobytes()


def test_parse_csv_of_an_empty_dataset() -> None:
    table = parse_csv(to_csv(make_dataset([])))
    assert table.columns == ("date", "inflation")
    assert table.dates == ()
    assert table.values.shape == (0,)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "header"),
        ("day,cpi\n", "header"),
        ("date,cpi\n2000-01-01\n", "line 2"),
        ("date,cpi\n2000-13-01,1.0\n", "line 2"),
        ("date,cpi\n2000-01-01,abc\n", "line 2"),
    ],
)
def test_parse_csv_rejects_malformed_input(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_csv(text)


# --- Shared strategies (sanity checks) ---------------------------------------


@settings(max_examples=50)
@given(datasets())
def test_generated_datasets_are_valid_monthly_datasets(ds: Dataset) -> None:
    assert ds.row_count <= 300
    assert ds.value_column == ds.indicator
    assert all(d.day == 1 for d in ds.dates)
    assert not np.isinf(ds.values).any()


@settings(max_examples=50)
@given(fred_observations())
def test_generated_fred_observations_have_unique_dates(obs: list[dict[str, str]]) -> None:
    dates = [o["date"] for o in obs]
    assert len(dates) == len(set(dates))
    for o in obs:
        date.fromisoformat(o["date"])
        assert o["value"] == "." or math.isfinite(float(o["value"]))
