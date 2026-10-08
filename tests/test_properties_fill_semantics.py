"""Property test for Fill_Tool semantics (task 6.6, Req 10.3, 10.4, 10.5).

The oracle below is an independent, loop-based reference written with plain Python floats
and ``date.toordinal`` day counts. It does not share any helper with ``friday.data.fill``.

Comparison rule:

- ``forward_fill`` copies values, so results must be equal exactly.
- ``linear_interpolation``: the oracle evaluates the spec formula
  ``v_a + (v_b - v_a) * w`` with ``w = days(d_a, d) / days(d_a, d_b)``. Where that is finite,
  the filled value must match it within a relative tolerance of ``1e-12`` (both sides use
  IEEE-754 doubles in the same operation order, so in practice they agree bit-for-bit).
  Where it is not finite, ``v_b - v_a`` overflowed. That only happens for neighbours with
  opposite signs and huge magnitudes. ``fill`` then uses the equivalent convex form
  ``v_a * (1 - w) + v_b * w``. In that case the filled value must be finite, lie between
  the two neighbours, and match the oracle's convex form within the same tolerance.
- Cells without the required neighbour(s) must stay Missing (NaN), and the unfilled count
  must equal the number of cells that stay Missing.
"""

import math
from datetime import date

import numpy as np
from hypothesis import example, given, settings
from hypothesis import strategies as st

from friday.data.dataset import Dataset
from friday.data.fill import FILL_METHODS, FillMethod, fill
from strategies import datasets

REL_TOL = 1e-12
"""Relative tolerance for interpolated cells (see the comparison rule above)."""

OVERFLOW_CASE = Dataset(
    dataset_id="ds-1",
    indicator="inflation rate",
    value_column="inflation rate",
    series_id="CPIAUCSL",
    dates=(date(2000, 1, 1), date(2000, 2, 1), date(2000, 3, 1)),
    values=np.array([-1e308, math.nan, 1e308]),
    transformation=None,
)
"""Opposite-sign huge neighbours, so ``v_b - v_a`` overflows; always exercised."""


def oracle_forward_fill(values: list[float]) -> list[float]:
    """Each Missing cell takes the nearest earlier non-missing value; else stays Missing."""
    out: list[float] = []
    last = math.nan
    for v in values:
        if not math.isnan(v):
            last = v
        out.append(last if math.isnan(v) else v)
    return out


def _nearest(values: list[float], i: int, step: int) -> int | None:
    """Index of the nearest non-missing value from ``i`` in direction ``step``, if any."""
    j = i + step
    while 0 <= j < len(values):
        if not math.isnan(values[j]):
            return j
        j += step
    return None


Neighbours = tuple[float, float, float]
"""``(v_a, v_b, w)`` for a Missing cell between two non-missing neighbours."""


def oracle_neighbours(dates: tuple[date, ...], values: list[float], i: int) -> Neighbours | None:
    """Nearest neighbour values and calendar-day weight for cell ``i``; None at an edge."""
    a, b = _nearest(values, i, -1), _nearest(values, i, +1)
    if a is None or b is None:
        return None
    w = (dates[i].toordinal() - dates[a].toordinal()) / (
        dates[b].toordinal() - dates[a].toordinal()
    )
    return values[a], values[b], w


def assert_interpolated(ds: Dataset, values: list[float], actual: list[float]) -> None:
    """Check every cell of a ``linear_interpolation`` result against the oracle."""
    for i, (source, got) in enumerate(zip(values, actual, strict=True)):
        if not math.isnan(source):
            assert got == source, f"row {i}: non-missing value changed"
            continue
        neighbours = oracle_neighbours(ds.dates, values, i)
        if neighbours is None:
            assert math.isnan(got), f"row {i}: edge gap filled with {got!r}"
            continue
        va, vb, w = neighbours
        expected = va + (vb - va) * w  # the spec formula (Req 10.4)
        if math.isfinite(expected):
            assert math.isclose(got, expected, rel_tol=REL_TOL), f"row {i}: {got!r} != {expected!r}"
        else:  # v_b - v_a overflowed: accept the overflow-free convex form
            convex = va * (1.0 - w) + vb * w
            assert math.isfinite(got), f"row {i}: overflow fallback gave {got!r}"
            assert min(va, vb) <= got <= max(va, vb), f"row {i}: {got!r} outside neighbours"
            assert math.isclose(got, convex, rel_tol=REL_TOL), f"row {i}: {got!r} != {convex!r}"


@settings(max_examples=300)
@example(ds=OVERFLOW_CASE, method="linear_interpolation")
@given(ds=datasets(), method=st.sampled_from(FILL_METHODS))
def test_fill_matches_reference_semantics(ds: Dataset, method: FillMethod) -> None:
    """Feature: friday-voice-data-assistant, Property 13: Fill matches reference semantics.

    For any Dataset and any Fill_Method, each originally Missing cell equals a reference
    loop: ``forward_fill`` takes the nearest earlier non-missing value, and
    ``linear_interpolation`` applies the calendar-day weighted formula between the nearest
    non-missing neighbours. Cells without the required neighbour(s) stay Missing.

    **Validates: Requirements 10.3, 10.4, 10.5**
    """
    values = [float(v) for v in ds.values]
    result = fill(ds, method, dataset_id="ds-new")
    actual = [float(v) for v in result.dataset.values]

    if method == "forward_fill":
        expected = oracle_forward_fill(values)
        assert np.array_equal(np.array(actual), np.array(expected), equal_nan=True)
    else:
        assert_interpolated(ds, values, actual)

    assert result.unfilled == sum(math.isnan(v) for v in actual)
