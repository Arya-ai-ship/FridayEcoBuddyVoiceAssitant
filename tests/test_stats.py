"""Unit tests for ``data/stats.py`` (task 6.1).

Properties 10-12 (reference model, ordering, formatting) have their own property tests.
"""

import math
import statistics
import sys
from datetime import date

import numpy as np
import pytest

from friday.data.dataset import Dataset
from friday.data.stats import (
    STATS_LABELS,
    UNDEFINED_TEXT,
    Stats,
    describe,
    format_stats_rows,
)

NAN = math.nan
FLOAT_MAX = sys.float_info.max


def make_dataset(values: list[float]) -> Dataset:
    dates = tuple(date(2000 + i // 12, i % 12 + 1, 1) for i in range(len(values)))
    return Dataset(
        dataset_id="ds-1",
        indicator="inflation",
        value_column="inflation",
        series_id="CPIAUCSL",
        dates=dates,
        values=np.array(values, dtype=np.float64),
        transformation="yoy",
    )


# --- describe ---------------------------------------------------------------


def test_describe_matches_numpy_and_skips_missing() -> None:
    values = [NAN, 3.0, 1.0, NAN, 4.0, 1.5, 9.0, NAN]
    stats = describe(make_dataset(values))
    present = np.array([v for v in values if not math.isnan(v)])
    assert stats.count == 5
    assert stats.missing_count == 3
    assert stats.mean == pytest.approx(float(np.mean(present)))
    assert stats.std == pytest.approx(float(np.std(present, ddof=1)))
    assert stats.min == 1.0
    assert stats.max == 9.0
    p25, median, p75 = np.percentile(present, [25, 50, 75])
    assert (stats.p25, stats.median, stats.p75) == pytest.approx((p25, median, p75))
    assert stats.first_date == date(2000, 2, 1)  # first non-missing row
    assert stats.last_date == date(2000, 7, 1)  # last non-missing row


def test_single_value_has_undefined_std_only() -> None:
    stats = describe(make_dataset([NAN, 2.5, NAN]))
    assert stats.count == 1
    assert stats.missing_count == 2
    assert stats.std is None
    assert stats.mean == stats.min == stats.p25 == stats.median == stats.p75 == stats.max == 2.5
    assert stats.first_date == stats.last_date == date(2000, 2, 1)


def test_constant_values_have_zero_std() -> None:
    assert describe(make_dataset([2.5, NAN, 2.5])).std == 0.0


@pytest.mark.parametrize("values", [[], [NAN, NAN, NAN]])
def test_no_values_leaves_everything_but_counts_undefined(values: list[float]) -> None:
    undefined = (None,) * 9
    assert describe(make_dataset(values)) == Stats(0, len(values), *undefined)


def test_describe_is_deterministic_and_read_only() -> None:
    ds = make_dataset([1.0, NAN, 2.0, 7.0])
    before = (ds.dataset_id, ds.dates, ds.values.tobytes())
    assert describe(ds) == describe(ds)
    assert (ds.dataset_id, ds.dates, ds.values.tobytes()) == before


def test_mean_is_exact_under_cancellation() -> None:
    stats = describe(make_dataset([1e6, 1e-6, -1e6]))
    assert stats.mean == statistics.mean([1e6, 1e-6, -1e6])


def test_extreme_floats_do_not_overflow() -> None:
    stats = describe(make_dataset([-FLOAT_MAX, FLOAT_MAX]))
    assert stats.mean == 0.0
    assert stats.median == 0.0
    assert stats.p25 == -FLOAT_MAX / 2
    assert stats.std is None  # true value (FLOAT_MAX * sqrt 2) is beyond the float range
    near_max = describe(make_dataset([FLOAT_MAX, FLOAT_MAX / 2]))
    assert near_max.mean == pytest.approx(0.75 * FLOAT_MAX)
    assert near_max.std == pytest.approx(FLOAT_MAX / 2 / math.sqrt(2))


def test_infinite_values_make_numeric_stats_undefined() -> None:
    stats = describe(make_dataset([1.0, math.inf, NAN]))
    assert (stats.count, stats.missing_count) == (2, 1)
    assert stats.mean is stats.std is stats.min is stats.max is stats.median is None
    assert (stats.first_date, stats.last_date) == (date(2000, 1, 1), date(2000, 2, 1))


# --- format_stats_rows ------------------------------------------------------


def test_format_rows_labels_and_values() -> None:
    stats = describe(make_dataset([NAN, 1.0, 2.0, 4.0]))
    rows = format_stats_rows(stats)
    assert tuple(row.label for row in rows) == STATS_LABELS
    assert len(rows) == 11
    assert [row.value for row in rows] == [
        "3",
        "1",
        "2.33",
        "1.53",
        "1.00",
        "1.50",
        "2.00",
        "3.00",
        "4.00",
        "2000-02-01",
        "2000-04-01",
    ]


def test_format_rows_undefined_and_negative_zero() -> None:
    rows = format_stats_rows(describe(make_dataset([NAN])))
    assert [row.value for row in rows] == ["0", "1"] + [UNDEFINED_TEXT] * 9
    tiny = format_stats_rows(describe(make_dataset([-0.001])))
    assert tiny[2].value == "0.00"
    assert tiny[3].value == UNDEFINED_TEXT
