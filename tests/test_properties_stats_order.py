"""Property test for statistics ordering and counts (design: Correctness Property 11)."""

import numpy as np
from hypothesis import given, settings

from friday.data.dataset import Dataset
from friday.data.stats import describe
from strategies import datasets


def _has_non_missing(ds: Dataset) -> bool:
    """True when at least one value is not a Missing_Value (NaN)."""
    return bool(np.any(~np.isnan(ds.values)))


@settings(max_examples=200)
@given(ds=datasets(min_rows=1).filter(_has_non_missing))
def test_stats_are_ordered_and_counts_are_complete(ds: Dataset) -> None:
    """Feature: friday-voice-data-assistant, Property 11: Descriptive statistics are
    ordered and counts are complete.

    For any Dataset with at least one non-missing value: min <= p25 <= median <= p75 <= max,
    and count + missing_count == row_count. Values span the full finite float range, which
    ``describe`` handles with exact arithmetic, so every order statistic and the mean are
    defined (only std may be ``None`` when it is not representable or count < 2).

    **Validates: Requirements 9.7**
    """
    stats = describe(ds)

    assert stats.count >= 1
    assert stats.count + stats.missing_count == ds.row_count

    lo, p25, median, p75, hi = stats.min, stats.p25, stats.median, stats.p75, stats.max
    assert lo is not None and p25 is not None and median is not None
    assert p75 is not None and hi is not None
    assert lo <= p25 <= median <= p75 <= hi, (lo, p25, median, p75, hi)

    assert stats.mean is not None
    assert lo <= stats.mean <= hi, (lo, stats.mean, hi)

    assert stats.first_date is not None
    assert stats.last_date is not None
    assert stats.first_date <= stats.last_date
