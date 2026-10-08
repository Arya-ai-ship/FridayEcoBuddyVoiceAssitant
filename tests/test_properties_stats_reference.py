"""Property test for ``describe`` against a reference model (task 6.2, design Property 10).

The reference model is independent of ``friday.data.stats``: mean and sample std come from
Python's ``statistics`` module, and the percentiles from a separate linear-interpolation
(numpy ``linear`` method) over the sorted non-missing values, written as the weighted
average ``(1 - w) * lo + w * hi`` of the two closest ranks in exact rational arithmetic.
Values are bounded so that the ``statistics`` reference itself never overflows.
"""

import math
import statistics
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from typing import Final

from hypothesis import given, settings

from friday.data.dataset import Dataset
from friday.data.stats import Stats, describe
from strategies import datasets

REL_TOL: Final = 1e-9
"""Relative tolerance between ``describe`` and the reference (design Property 10)."""

MAX_MAGNITUDE: Final = 1e150
"""Largest |value| generated; squares and sums stay finite for the ``statistics`` model."""

QUARTILES: Final = (25, 50, 75)


@dataclass(frozen=True)
class Reference:
    """Reference Descriptive_Statistics (``None`` means undefined)."""

    count: int
    missing_count: int
    mean: float | None
    std: float | None
    min: float | None
    p25: float | None
    median: float | None
    p75: float | None
    max: float | None
    first_date: date | None
    last_date: date | None


def reference_percentile(ordered: list[float], q: int) -> float:
    """Percentile ``q`` of sorted values by linear interpolation between the closest ranks."""
    rank = Fraction(q, 100) * (len(ordered) - 1)
    lo_index = int(rank)  # rank >= 0, so int() is floor
    hi_index = min(lo_index + 1, len(ordered) - 1)
    weight = rank - lo_index
    lo, hi = Fraction(ordered[lo_index]), Fraction(ordered[hi_index])
    return float((1 - weight) * lo + weight * hi)


def reference_stats(ds: Dataset) -> Reference:
    """Descriptive_Statistics computed with ``statistics`` and plain Python loops."""
    rows = [
        (day, float(value))
        for day, value in zip(ds.dates, ds.values.tolist(), strict=True)
        if not math.isnan(value)
    ]
    present = [value for _, value in rows]
    count = len(present)
    missing_count = ds.row_count - count
    if count == 0:
        return Reference(count, missing_count, *(None,) * 9)
    ordered = sorted(present)
    p25, median, p75 = (reference_percentile(ordered, q) for q in QUARTILES)
    return Reference(
        count=count,
        missing_count=missing_count,
        mean=statistics.mean(present),
        std=statistics.stdev(present) if count >= 2 else None,
        min=ordered[0],
        p25=p25,
        median=median,
        p75=p75,
        max=ordered[-1],
        first_date=rows[0][0],
        last_date=rows[-1][0],
    )


def assert_close(name: str, actual: float | None, expected: float | None) -> None:
    """Both undefined, or both defined and within ``REL_TOL`` of each other."""
    if expected is None:
        assert actual is None, f"{name}: expected undefined, got {actual!r}"
        return
    assert actual is not None, f"{name}: expected {expected!r}, got undefined"
    assert math.isclose(actual, expected, rel_tol=REL_TOL), f"{name}: {actual!r} != {expected!r}"


def snapshot(ds: Dataset) -> tuple[object, ...]:
    """Everything ``describe`` must leave unchanged (Req 9.3)."""
    return (
        ds.dataset_id,
        ds.indicator,
        ds.value_column,
        ds.series_id,
        ds.dates,
        ds.values.tobytes(),
        ds.transformation,
        ds.derived_from,
        ds.fill_method,
    )


@settings(max_examples=200)
@given(datasets(max_magnitude=MAX_MAGNITUDE))
def test_describe_matches_reference_model_and_is_deterministic(ds: Dataset) -> None:
    """Feature: friday-voice-data-assistant, Property 10: Descriptive statistics match a
    reference model and are deterministic.

    For any Dataset, ``describe(ds)`` equals a reference computed with Python's
    ``statistics`` module and an independent linear-interpolation percentile over the
    sorted non-missing values, within 1e-9 relative tolerance. Std is undefined with
    exactly 1 non-missing value; every non-count statistic is undefined with 0. Calling
    ``describe`` twice gives identical results, and the Dataset is unchanged.

    **Validates: Requirements 9.1, 9.2, 9.6, 9.8, 9.9**
    """
    before = snapshot(ds)
    stats = describe(ds)
    expected = reference_stats(ds)

    assert stats.count == expected.count
    assert stats.missing_count == expected.missing_count
    assert stats.first_date == expected.first_date
    assert stats.last_date == expected.last_date
    for name in ("mean", "std", "min", "p25", "median", "p75", "max"):
        assert_close(name, getattr(stats, name), getattr(expected, name))
    if expected.count == 1:
        assert stats.std is None
    if expected.count == 0:
        assert stats == Stats(0, ds.row_count, *(None,) * 9)

    assert describe(ds) == stats
    assert snapshot(ds) == before
