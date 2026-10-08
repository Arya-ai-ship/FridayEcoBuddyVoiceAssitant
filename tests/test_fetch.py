"""Unit tests for ``data/fetch.py`` (task 4.4).

Properties 2 (Dataset construction) and 3 (YoY) have their own property-test modules.
"""

import math
from datetime import date

import numpy as np
import pytest

from friday.constants import YOY_LAG_ROWS
from friday.data.fetch import apply_yoy, build_dataset, transform
from friday.data.indicators import IndicatorEntry, default_indicator_map
from friday.errors import RangeTooShortForYoY

NAN = math.nan
ENTRY = IndicatorEntry(name="unemployment rate", aliases=(), series_id="UNRATE")
INFLATION = default_indicator_map().resolve("inflation")


def obs(day: str, value: str) -> dict[str, str]:
    return {"date": day, "value": value}


def monthly(values: list[float]) -> list[dict[str, str]]:
    """Observations from January 2000 onward, ``NaN`` written as FRED's ``.``."""
    return [
        obs(date(2000 + i // 12, i % 12 + 1, 1).isoformat(), "." if math.isnan(v) else repr(v))
        for i, v in enumerate(values)
    ]


def test_build_dataset_sorts_maps_placeholder_and_names_column() -> None:
    raw = [obs("2020-03-01", "4.4"), obs("2020-01-01", "3.5"), obs("2020-02-01", ".")]
    ds = build_dataset(raw, ENTRY, None, None, dataset_id="ds-7")
    assert ds.dataset_id == "ds-7"
    assert ds.indicator == ds.value_column == "unemployment rate"
    assert ds.series_id == "UNRATE"
    assert ds.transformation is None
    assert ds.dates == (date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1))
    np.testing.assert_array_equal(ds.values, [3.5, NAN, 4.4])
    assert not ds.values.flags.writeable


def test_build_dataset_bounds_are_inclusive_and_optional() -> None:
    raw = [obs(f"2020-{m:02d}-01", str(m)) for m in range(1, 7)]
    both = build_dataset(raw, ENTRY, date(2020, 2, 1), date(2020, 4, 1), dataset_id="ds-1")
    assert both.dates == (date(2020, 2, 1), date(2020, 3, 1), date(2020, 4, 1))
    start_only = build_dataset(raw, ENTRY, date(2020, 5, 1), None, dataset_id="ds-1")
    assert start_only.dates == (date(2020, 5, 1), date(2020, 6, 1))
    end_only = build_dataset(raw, ENTRY, None, date(2020, 1, 15), dataset_id="ds-1")
    assert end_only.dates == (date(2020, 1, 1),)


def test_build_dataset_with_nothing_in_range_is_empty() -> None:
    ds = build_dataset([obs("2020-01-01", "1")], ENTRY, date(2021, 1, 1), None, dataset_id="ds-1")
    assert ds.row_count == 0


def test_build_dataset_rejects_duplicate_dates() -> None:
    with pytest.raises(ValueError):
        build_dataset(
            [obs("2020-01-01", "1"), obs("2020-01-01", ".")], ENTRY, None, None, dataset_id="ds-1"
        )


def test_apply_yoy_formula_missing_and_zero_denominator() -> None:
    x = [100.0, 0.0, NAN, 50.0, *[1.0] * 8, 110.0, 5.0, 7.0, NAN]
    result = apply_yoy(x)
    assert result.shape == (len(x) - YOY_LAG_ROWS,)
    assert result[0] == (110.0 / 100.0 - 1.0) * 100.0
    assert math.isnan(result[1])  # zero denominator
    assert math.isnan(result[2])  # missing denominator
    assert math.isnan(result[3])  # missing numerator
    assert not result.flags.writeable


@pytest.mark.parametrize("n", [0, 1, YOY_LAG_ROWS])
def test_apply_yoy_too_short_raises(n: int) -> None:
    with pytest.raises(RangeTooShortForYoY) as info:
        apply_yoy([1.0] * n)
    assert info.value.kind == "range_too_short_for_yoy"


def test_transform_yoy_drops_first_twelve_dates() -> None:
    values = [float(100 + i) for i in range(YOY_LAG_ROWS + 3)]
    raw = build_dataset(monthly(values), INFLATION, None, None, dataset_id="ds-2")
    ds = transform(raw, INFLATION.transformation)
    assert ds.transformation == "yoy"
    assert ds.dataset_id == "ds-2"
    assert ds.row_count == raw.row_count - YOY_LAG_ROWS
    assert ds.dates == raw.dates[YOY_LAG_ROWS:]
    np.testing.assert_array_equal(ds.values, apply_yoy(raw.values))


def test_transform_none_returns_same_dataset() -> None:
    raw = build_dataset(monthly([1.0, 2.0]), ENTRY, None, None, dataset_id="ds-1")
    assert transform(raw, None) is raw
