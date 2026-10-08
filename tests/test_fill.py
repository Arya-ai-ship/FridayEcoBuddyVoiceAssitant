"""Unit tests for ``friday.data.fill`` (Req 10.1-10.8, 10.11).

Properties 13-15 (reference semantics, structure, idempotence) live in their own tests.
"""

from __future__ import annotations

import math
import sys
from datetime import date
from typing import cast

import numpy as np
import pytest

from friday.data.dataset import Dataset
from friday.data.fill import DEFAULT_FILL_METHOD, FILL_METHODS, FillMethod, fill

NAN = math.nan
JAN, FEB, MAR, APR, MAY = (date(2024, m, 1) for m in range(1, 6))
DATES = (JAN, FEB, MAR, APR, MAY)


def make(values: list[float], dates: tuple[date, ...] = DATES) -> Dataset:
    return Dataset(
        dataset_id="ds-1",
        indicator="inflation",
        value_column="inflation",
        series_id="CPIAUCSL",
        dates=dates[: len(values)],
        values=np.array(values, dtype=np.float64),
        transformation="yoy",
    )


def as_list(ds: Dataset) -> list[float]:
    return cast(list[float], ds.values.tolist())


def same(actual: list[float], expected: list[float]) -> bool:
    """Element-wise equality where NaN equals NaN."""
    return len(actual) == len(expected) and all(
        (math.isnan(x) and math.isnan(y)) or x == y for x, y in zip(actual, expected, strict=True)
    )


def test_supported_methods_and_default() -> None:
    assert FILL_METHODS == ("forward_fill", "linear_interpolation")
    assert DEFAULT_FILL_METHOD == "forward_fill"


def test_forward_fill_carries_last_value_and_leaves_leading_gap() -> None:
    result = fill(make([NAN, 1.0, NAN, NAN, 4.0]), "forward_fill", dataset_id="ds-2")
    assert same(as_list(result.dataset), [NAN, 1.0, 1.0, 1.0, 4.0])
    assert (result.filled, result.unfilled) == (2, 1)


def test_default_method_is_forward_fill() -> None:
    result = fill(make([1.0, NAN, 3.0]), dataset_id="ds-2")
    assert as_list(result.dataset) == [1.0, 1.0, 3.0]
    assert result.dataset.fill_method == "forward_fill"


def test_forward_fill_fills_trailing_gap() -> None:
    result = fill(make([2.0, NAN, NAN]), "forward_fill", dataset_id="ds-2")
    assert as_list(result.dataset) == [2.0, 2.0, 2.0]
    assert (result.filled, result.unfilled) == (2, 0)


def test_linear_interpolation_uses_calendar_day_weights() -> None:
    # Jan 1 -> Feb 1 is 31 days, Jan 1 -> Mar 1 is 60 days (2024 is a leap year).
    result = fill(make([0.0, NAN, 60.0]), "linear_interpolation", dataset_id="ds-2")
    assert as_list(result.dataset) == pytest.approx([0.0, 31.0, 60.0])
    assert (result.filled, result.unfilled) == (1, 0)


def test_linear_interpolation_leaves_both_edges_missing() -> None:
    result = fill(make([NAN, 1.0, NAN, 3.0, NAN]), "linear_interpolation", dataset_id="ds-2")
    filled = as_list(result.dataset)
    assert math.isnan(filled[0]) and math.isnan(filled[4])
    assert filled[1:4] == pytest.approx([1.0, 1.0 + 2.0 * 29 / 60, 3.0])
    assert (result.filled, result.unfilled) == (1, 2)


def test_linear_interpolation_stays_finite_for_extreme_neighbours() -> None:
    big = sys.float_info.max
    result = fill(make([-big, NAN, big]), "linear_interpolation", dataset_id="ds-2")
    middle = as_list(result.dataset)[1]
    assert math.isfinite(middle)
    assert -big <= middle <= big


@pytest.mark.parametrize("method", FILL_METHODS)
def test_new_dataset_metadata_and_source_unchanged(method: FillMethod) -> None:
    source = make([1.0, NAN, 3.0])
    before = as_list(source)
    result = fill(source, method, dataset_id="ds-7")
    new = result.dataset
    assert new is not source
    assert new.dataset_id == "ds-7"
    assert new.derived_from == "ds-1"
    assert new.fill_method == method
    assert new.value_column == source.value_column
    assert (new.indicator, new.series_id, new.transformation) == ("inflation", "CPIAUCSL", "yoy")
    assert new.dates == source.dates
    assert not new.values.flags.writeable
    assert same(as_list(source), before)
    assert source.dataset_id == "ds-1"


@pytest.mark.parametrize("method", FILL_METHODS)
def test_no_missing_values_gives_identical_values(method: FillMethod) -> None:
    source = make([1.5, -2.0, 0.0])
    result = fill(source, method, dataset_id="ds-2")
    assert as_list(result.dataset) == as_list(source)
    assert (result.filled, result.unfilled) == (0, 0)


@pytest.mark.parametrize("method", FILL_METHODS)
def test_empty_and_all_missing_datasets(method: FillMethod) -> None:
    empty = fill(make([]), method, dataset_id="ds-2")
    assert empty.dataset.row_count == 0
    assert (empty.filled, empty.unfilled) == (0, 0)
    blank = fill(make([NAN, NAN]), method, dataset_id="ds-3")
    assert blank.dataset.missing_count == 2
    assert (blank.filled, blank.unfilled) == (0, 2)


@pytest.mark.parametrize("method", FILL_METHODS)
def test_refilling_fills_nothing(method: FillMethod) -> None:
    once = fill(make([NAN, 1.0, NAN, 4.0, NAN]), method, dataset_id="ds-2")
    twice = fill(once.dataset, method, dataset_id="ds-3")
    assert same(as_list(twice.dataset), as_list(once.dataset))
    assert twice.filled == 0


def test_invalid_method_names_it_and_lists_supported() -> None:
    with pytest.raises(ValueError, match="invalid method 'spline'; supported: forward_fill, "):
        fill(make([1.0]), cast(FillMethod, "spline"), dataset_id="ds-2")
