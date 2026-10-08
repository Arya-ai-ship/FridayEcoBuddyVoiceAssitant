"""Unit tests for strict date-range parsing (task 3.4, Req 6.12, 11.9)."""

from datetime import date

import pytest

from friday.data.dates import END_FIELD, START_FIELD, parse_date, parse_date_range
from friday.errors import DateRangeError


def test_both_bounds_omitted_is_unbounded() -> None:
    assert parse_date_range(None, None) == (None, None)


def test_valid_bounds_are_parsed() -> None:
    assert parse_date_range("2020-01-01", "2024-02-29") == (date(2020, 1, 1), date(2024, 2, 29))


def test_single_bound_leaves_other_side_unbounded() -> None:
    assert parse_date_range("1990-06-15", None) == (date(1990, 6, 15), None)
    assert parse_date_range(None, "1990-06-15") == (None, date(1990, 6, 15))


def test_equal_bounds_are_allowed() -> None:
    assert parse_date_range("2021-03-01", "2021-03-01") == (date(2021, 3, 1), date(2021, 3, 1))


@pytest.mark.parametrize(
    "raw",
    [
        "20240101",  # basic ISO form accepted by date.fromisoformat
        "2024-W01-1",  # ISO week date
        "2024-001",  # ordinal date
        "2024-1-01",
        "2024-01-1",
        "24-01-01",
        "2024/01/01",
        "2024-01-01T00:00:00",
        " 2024-01-01",
        "2024-01-01 ",
        "2024-01-01\n",
        "",
        "yesterday",
        "２０２４-０１-０１",  # full-width (non-ASCII) digits
    ],
)
def test_malformed_shape_is_rejected(raw: str) -> None:
    with pytest.raises(DateRangeError) as info:
        parse_date(raw, START_FIELD)
    err = info.value
    assert err.kind == "invalid_date"
    assert err.field == START_FIELD
    assert repr(raw) in err.message
    assert "YYYY-MM-DD" in err.message


@pytest.mark.parametrize(
    "raw", ["2023-02-29", "2023-02-30", "2024-13-01", "2024-00-10", "0000-01-01"]
)
def test_non_calendar_date_is_rejected(raw: str) -> None:
    with pytest.raises(DateRangeError) as info:
        parse_date(raw, END_FIELD)
    assert info.value.field == END_FIELD
    assert raw in info.value.message
    assert "calendar" in info.value.message


def test_bad_end_date_names_end_field() -> None:
    with pytest.raises(DateRangeError) as info:
        parse_date_range("2020-01-01", "2020-02-30")
    assert info.value.field == END_FIELD
    assert END_FIELD in info.value.message


def test_bad_start_date_names_start_field() -> None:
    with pytest.raises(DateRangeError) as info:
        parse_date_range("20200101", "2020-02-01")
    assert info.value.field == START_FIELD


def test_start_later_than_end_is_rejected() -> None:
    with pytest.raises(DateRangeError) as info:
        parse_date_range("2024-01-02", "2024-01-01")
    err = info.value
    assert err.field == START_FIELD
    assert "2024-01-02" in err.message
    assert "2024-01-01" in err.message
    assert "later than" in err.message


def test_error_renders_as_tool_result() -> None:
    with pytest.raises(DateRangeError) as info:
        parse_date_range("2024-02-30", None)
    result = info.value.to_result("fetch_data")
    assert result["ok"] is False
    assert result["tool"] == "fetch_data"
    assert result["error"] == {
        "kind": "invalid_date",
        "message": info.value.message,
        "field": START_FIELD,
    }
