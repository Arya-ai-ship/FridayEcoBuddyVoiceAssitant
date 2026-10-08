"""Shared Hypothesis generators (design: Testing Strategy, "Key generators").

- ``datasets()``: 0-300 monthly rows from a random start month, finite floats (negatives,
  zeros, huge and tiny magnitudes) with roughly 0-40% Missing_Values, plus explicit
  all-missing, single-non-missing, leading-missing, and trailing-missing shapes.
- ``fred_observations()``: unique monthly FRED observations in shuffled order, as
  ``{"date": "YYYY-MM-DD", "value": "<number>" | "."}`` dicts like the FRED JSON API.

Generators used by only one property stay in that property's test module.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Final

import numpy as np
from hypothesis import strategies as st

from friday.data.dataset import Dataset
from friday.data.fetch import FRED_MISSING
from friday.data.indicators import default_indicator_map

MAX_ROWS: Final = 300

MIN_YEAR: Final = 1900
MAX_YEAR: Final = 2100

SHAPES: Final = (
    "random",
    "all_missing",
    "single_non_missing",
    "leading_missing",
    "trailing_missing",
)
"""Missing-value layouts; ``random`` is first so failures shrink toward it."""

_DEFAULTS = default_indicator_map()
_NAME_ALPHABET: Final = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_.,'\"()%/"


def month_offset(start: date, months: int) -> date:
    """The first day of the month ``months`` after ``start``'s month."""
    index = start.year * 12 + (start.month - 1) + months
    return date(index // 12, index % 12 + 1, 1)


@st.composite
def monthly_dates(draw: st.DrawFn, n: int) -> tuple[date, ...]:
    """``n`` consecutive first-of-month dates from a random start month."""
    last_start_year = MAX_YEAR - math.ceil(MAX_ROWS / 12)
    start = date(draw(st.integers(MIN_YEAR, last_start_year)), draw(st.integers(1, 12)), 1)
    return tuple(month_offset(start, i) for i in range(n))


def finite_floats(max_magnitude: float | None = None) -> st.SearchStrategy[float]:
    """Finite floats; unbounded by default (includes zeros, subnormals, and huge values)."""
    if max_magnitude is None:
        return st.floats(allow_nan=False, allow_infinity=False)
    return st.floats(
        min_value=-max_magnitude, max_value=max_magnitude, allow_nan=False, allow_infinity=False
    )


@st.composite
def missing_masks(draw: st.DrawFn, n: int) -> tuple[bool, ...]:
    """A Missing mask of length ``n`` (True == Missing) in one of the ``SHAPES``."""
    if n == 0:
        return ()
    shape = draw(st.sampled_from(SHAPES))
    rate = draw(st.integers(0, 4))  # tenths: roughly 0-40% Missing
    mask = [d < rate for d in draw(st.lists(st.integers(0, 9), min_size=n, max_size=n))]
    if shape == "all_missing":
        return (True,) * n
    if shape == "single_non_missing":
        keep = draw(st.integers(0, n - 1))
        return tuple(i != keep for i in range(n))
    if shape in ("leading_missing", "trailing_missing") and n >= 2:
        k = draw(st.integers(1, n - 1))  # k missing rows at one edge, then a non-missing row
        if shape == "leading_missing":
            mask[:k] = [True] * k
            mask[k] = False
        else:
            mask[n - k :] = [True] * k
            mask[n - k - 1] = False
    return tuple(mask)


@st.composite
def value_lists(draw: st.DrawFn, n: int, max_magnitude: float | None = None) -> tuple[float, ...]:
    """``n`` values with NaN at the positions of a drawn ``missing_masks`` layout."""
    mask = draw(missing_masks(n))
    raw = draw(st.lists(finite_floats(max_magnitude), min_size=n, max_size=n))
    return tuple(math.nan if missing else value for value, missing in zip(raw, mask, strict=True))


def indicator_names() -> st.SearchStrategy[str]:
    """Default Indicator names, or arbitrary names with CSV-sensitive punctuation."""
    custom = st.text(_NAME_ALPHABET, min_size=1, max_size=30).map(str.strip).filter(bool)
    return st.sampled_from(_DEFAULTS.names) | custom


def series_ids() -> st.SearchStrategy[str]:
    """Default FRED series IDs, or arbitrary upper-case alphanumeric IDs."""
    defaults = tuple(entry.series_id for entry in _DEFAULTS.entries)
    custom = st.text("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", min_size=1, max_size=12)
    return st.sampled_from(defaults) | custom


@st.composite
def datasets(
    draw: st.DrawFn,
    *,
    min_rows: int = 0,
    max_rows: int = MAX_ROWS,
    max_magnitude: float | None = None,
) -> Dataset:
    """A Dataset of monthly rows with Missing_Values in one of the ``SHAPES``.

    ``max_magnitude`` bounds the values for properties whose reference model would
    overflow on extreme floats; by default the full finite range is used.
    """
    n = draw(st.integers(min_rows, max_rows))
    indicator = draw(indicator_names())
    return Dataset(
        dataset_id=f"ds-{draw(st.integers(1, 99))}",
        indicator=indicator,
        value_column=indicator,
        series_id=draw(series_ids()),
        dates=draw(monthly_dates(n)),
        values=np.array(draw(value_lists(n, max_magnitude)), dtype=np.float64),
        transformation=draw(st.sampled_from((None, "yoy"))),
    )


def fred_value_texts() -> st.SearchStrategy[str]:
    """FRED value strings: plain decimals, ``repr`` of finite floats, or the ``.`` placeholder."""
    decimals = st.decimals(
        min_value=-1_000_000, max_value=1_000_000, places=3, allow_nan=False, allow_infinity=False
    ).map(str)
    return st.one_of(st.just(FRED_MISSING), decimals, finite_floats().map(repr))


@st.composite
def fred_observations(draw: st.DrawFn, *, max_size: int = MAX_ROWS) -> list[dict[str, str]]:
    """Unique monthly observations (gaps allowed) in shuffled order, ``.`` for missing."""
    base = date(draw(st.integers(MIN_YEAR, MAX_YEAR - 2 * math.ceil(max_size / 12))), 1, 1)
    offsets = draw(st.sets(st.integers(0, 2 * max_size), max_size=max_size))
    shuffled = draw(st.permutations(sorted(offsets)))
    return [
        {"date": month_offset(base, offset).isoformat(), "value": draw(fred_value_texts())}
        for offset in shuffled
    ]
