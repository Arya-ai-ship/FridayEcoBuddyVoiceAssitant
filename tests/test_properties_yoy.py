"""Property test for the YoY_Transformation (task 4.6, Req 6.6, 6.7, 6.13).

Overflow policy: the design defines a YoY row as ``(x[i+12] / x[i] - 1) * 100`` evaluated in
IEEE-754 doubles, and Missing (NaN) only when an input is Missing or the denominator is 0.
For extreme finite inputs (e.g. ``1e300 / 1e-300``) that expression overflows to ``±inf``;
this is accepted as the formula's float result and is not a Missing_Value. Dataset, the
Preview_Table formatter, and the CSV_Export already represent ``±inf``, and real FRED
monthly series are many orders of magnitude away from overflow. The oracle below evaluates
the same expression with Python floats (also IEEE-754, returning ``inf`` on overflow rather
than raising), so results are compared bit-for-bit, infinities included.
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from friday.constants import YOY_LAG_ROWS
from friday.data.dataset import Dataset
from friday.data.fetch import apply_yoy, transform
from friday.errors import RangeTooShortForYoY
from strategies import finite_floats, missing_masks, monthly_dates

MAX_ROWS = 4 * YOY_LAG_ROWS + 12
"""Long enough for several lagged pairs; short n (<= 12) is drawn often enough to matter."""


def yoy_values() -> st.SearchStrategy[float]:
    """Finite floats with zeros (both signs) and realistic magnitudes drawn often."""
    return st.one_of(
        st.sampled_from((0.0, -0.0)),
        finite_floats(max_magnitude=1_000.0),
        finite_floats(),
    )


@st.composite
def monthly_series(draw: st.DrawFn) -> tuple[list[float], Dataset]:
    """A raw value list (NaN == Missing) and the untransformed monthly Dataset holding it."""
    n = draw(st.integers(0, MAX_ROWS))
    mask = draw(missing_masks(n))
    raw = draw(st.lists(yoy_values(), min_size=n, max_size=n))
    values = [math.nan if missing else v for v, missing in zip(raw, mask, strict=True)]
    dataset = Dataset(
        dataset_id="ds-1",
        indicator="inflation rate",
        value_column="inflation rate",
        series_id="CPIAUCSL",
        dates=draw(monthly_dates(n)),
        values=np.array(values, dtype=np.float64),
        transformation=None,
    )
    return values, dataset


def oracle_yoy(x: list[float]) -> list[float]:
    """Independent reference: the YoY formula row by row with plain Python floats."""
    out: list[float] = []
    for previous, current in zip(x, x[YOY_LAG_ROWS:], strict=False):
        if math.isnan(previous) or math.isnan(current) or previous == 0.0:
            out.append(math.nan)
        else:
            out.append((current / previous - 1.0) * 100.0)
    return out


@settings(max_examples=300)
@given(monthly_series())
def test_yoy_matches_formula_and_drops_twelve_rows(series: tuple[list[float], Dataset]) -> None:
    """Feature: friday-voice-data-assistant, Property 3: YoY transformation matches its
    formula and drops 12 rows.

    For any monthly value sequence ``x`` of length n (Missing values and zeros allowed): if
    n > 12, ``apply_yoy(x)`` has exactly n - 12 rows and row i equals
    ``(x[i+12] / x[i] - 1) * 100`` (bit-for-bit against an independent oracle, overflow to
    ``±inf`` included), and is Missing exactly when ``x[i]`` or ``x[i+12]`` is Missing or
    ``x[i]`` is 0. ``transform(..., "yoy")`` keeps ``dates[12:]`` aligned with those rows.
    If n <= 12, both raise ``RangeTooShortForYoY``.

    **Validates: Requirements 6.6, 6.7, 6.13**
    """
    x, raw = series
    n = len(x)

    if n <= YOY_LAG_ROWS:
        with pytest.raises(RangeTooShortForYoY):
            apply_yoy(x)
        with pytest.raises(RangeTooShortForYoY):
            transform(raw, "yoy")
        return

    result = apply_yoy(x)
    expected = np.array(oracle_yoy(x), dtype=np.float64)
    assert result.shape == (n - YOY_LAG_ROWS,)
    assert not result.flags.writeable

    missing = np.isnan(expected)
    np.testing.assert_array_equal(np.isnan(result), missing)
    # Bit-for-bit on non-missing rows: catches rounding, sign-of-zero, and ±inf mismatches.
    np.testing.assert_array_equal(
        result.view(np.uint64)[~missing], expected.view(np.uint64)[~missing]
    )

    ds = transform(raw, "yoy")
    assert ds.transformation == "yoy"
    assert ds.dataset_id == raw.dataset_id
    assert ds.row_count == raw.row_count - YOY_LAG_ROWS
    assert ds.dates == raw.dates[YOY_LAG_ROWS:]
    np.testing.assert_array_equal(ds.values.view(np.uint64), result.view(np.uint64))
