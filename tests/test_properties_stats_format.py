"""Property test for ``format_stats_rows`` (task 6.4, design Property 12).

The reference model for numeric rows is independent of ``format_stats_rows``: the exact
decimal value of the float, quantized half-even to ``STATS_DECIMALS`` places. Stats are
drawn both directly (arbitrary counts, finite floats including extremes, ``-0.0``, values
near rounding ties, and ``None``; arbitrary dates or ``None``) and via ``describe``.
"""

import re
from datetime import date
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from typing import Final

from hypothesis import given, settings
from hypothesis import strategies as st

from friday.constants import STATS_DECIMALS
from friday.data.stats import STATS_LABELS, UNDEFINED_TEXT, Stats, describe, format_stats_rows
from strategies import datasets, finite_floats

COUNT_ROWS: Final = 2
"""The first rows (Count, Missing values) are counts."""

DATE_ROWS: Final = 2
"""The last rows (First date, Last date) are dates."""

ISO_DATE: Final = re.compile(r"\d{4}-\d{2}-\d{2}")
PLAIN_INTEGER: Final = re.compile(r"0|[1-9]\d*")
FIXED_NUMBER: Final = re.compile(rf"-?\d+\.\d{{{STATS_DECIMALS}}}")
"""Exactly ``STATS_DECIMALS`` decimals."""

_QUANTUM: Final = Decimal(1).scaleb(-STATS_DECIMALS)
_EXACT_PRECISION: Final = 400  # > 309 integer digits of the largest float64, plus decimals


def expected_display(value: float) -> Decimal:
    """``value`` rounded half-even to ``STATS_DECIMALS`` places, computed exactly."""
    with localcontext() as ctx:
        ctx.prec = _EXACT_PRECISION
        return Decimal(value).quantize(_QUANTUM, rounding=ROUND_HALF_EVEN)


def statistic_values() -> st.SearchStrategy[float | None]:
    """Finite floats (full range), signed zeros, values near 2-decimal ties, or ``None``."""
    near_ties = st.decimals(
        min_value=-10_000, max_value=10_000, places=STATS_DECIMALS + 1, allow_nan=False
    ).map(float)
    return st.one_of(
        st.none(), st.just(-0.0), finite_floats(), finite_floats(max_magnitude=0.01), near_ties
    )


def counts() -> st.SearchStrategy[int]:
    """Non-negative counts, small or very large."""
    return st.integers(min_value=0, max_value=10**15)


@st.composite
def direct_stats(draw: st.DrawFn) -> Stats:
    """A ``Stats`` with every field drawn independently."""
    numbers = [draw(statistic_values()) for _ in range(7)]
    mean, std, low, p25, median, p75, high = numbers
    return Stats(
        count=draw(counts()),
        missing_count=draw(counts()),
        mean=mean,
        std=std,
        min=low,
        p25=p25,
        median=median,
        p75=p75,
        max=high,
        first_date=draw(st.none() | st.dates()),
        last_date=draw(st.none() | st.dates()),
    )


def _numeric_fields(stats: Stats) -> tuple[float | None, ...]:
    return (
        stats.mean,
        stats.std,
        stats.min,
        stats.p25,
        stats.median,
        stats.p75,
        stats.max,
    )


@settings(max_examples=300)
@given(st.one_of(direct_stats(), datasets().map(describe)))
def test_stats_table_formatting(stats: Stats) -> None:
    """Feature: friday-voice-data-assistant, Property 12: Stats table formatting.

    For any Stats result, ``format_stats_rows`` returns 11 rows in the
    Descriptive_Statistics order. Count rows are integer strings. Every other numeric row
    has exactly 2 decimal places and equals the value rounded to 2 decimals. Date rows match
    ``YYYY-MM-DD``. Undefined values are shown as "undefined".

    **Validates: Requirements 9.4**
    """
    rows = format_stats_rows(stats)

    assert tuple(row.label for row in rows) == STATS_LABELS
    texts = [row.value for row in rows]
    count_texts = texts[:COUNT_ROWS]
    number_texts = texts[COUNT_ROWS:-DATE_ROWS]
    date_texts = texts[-DATE_ROWS:]

    for text, count in zip(count_texts, (stats.count, stats.missing_count), strict=True):
        assert PLAIN_INTEGER.fullmatch(text), text
        assert int(text) == count

    for text, value in zip(number_texts, _numeric_fields(stats), strict=True):
        if value is None:
            assert text == UNDEFINED_TEXT
            continue
        assert FIXED_NUMBER.fullmatch(text), text
        expected = expected_display(value)
        assert Decimal(text) == expected
        if expected.is_zero():
            assert not text.startswith("-"), text

    for text, day in zip(date_texts, (stats.first_date, stats.last_date), strict=True):
        if day is None:
            assert text == UNDEFINED_TEXT
        else:
            assert ISO_DATE.fullmatch(text), text
            assert date.fromisoformat(text) == day
