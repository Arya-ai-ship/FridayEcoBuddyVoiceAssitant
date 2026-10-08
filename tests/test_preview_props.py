"""Property test for ``preview_rows`` (task 4.2, design Property 8).

The reference model for display values is independent of ``format_display_value``: the
exact decimal value of the float, quantized half-even to ``PREVIEW_DECIMALS`` places.
"""

import math
import re
from datetime import date
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from typing import Final

from hypothesis import given, settings

from friday.constants import PREVIEW_DECIMALS, PREVIEW_ROWS
from friday.data.dataset import Dataset, preview_rows
from strategies import datasets

ISO_DATE: Final = re.compile(r"\d{4}-\d{2}-\d{2}")
DISPLAY_NUMBER: Final = re.compile(rf"-?\d+(\.\d{{0,{PREVIEW_DECIMALS - 1}}}[1-9])?")
"""At most ``PREVIEW_DECIMALS`` decimals, no trailing zeros, no bare decimal point."""

_QUANTUM: Final = Decimal(1).scaleb(-PREVIEW_DECIMALS)
_EXACT_PRECISION: Final = 1200  # enough digits for any finite float64 at 4 decimals


def expected_display(value: float) -> Decimal:
    """``value`` rounded half-even to ``PREVIEW_DECIMALS`` places, computed exactly."""
    with localcontext() as ctx:
        ctx.prec = _EXACT_PRECISION
        return Decimal(value).quantize(_QUANTUM, rounding=ROUND_HALF_EVEN)


@settings(max_examples=200)
@given(datasets(min_rows=1))
def test_preview_shows_the_first_rows_in_display_format(ds: Dataset) -> None:
    """Feature: friday-voice-data-assistant, Property 8: Preview shows the first rows in
    display format.

    For any Dataset with n >= 1 rows, ``preview_rows`` returns exactly ``min(10, n)`` rows
    equal to the Dataset's first rows in ascending date order. Each date is ISO
    ``YYYY-MM-DD``, and each Missing value is the empty string.

    **Validates: Requirements 7.1, 7.2, 7.3**
    """
    rows = preview_rows(ds)

    assert len(rows) == min(PREVIEW_ROWS, ds.row_count)
    shown_dates = [date.fromisoformat(day) for day, _ in rows]
    assert shown_dates == sorted(shown_dates)
    for i, (day, text) in enumerate(rows):
        assert ISO_DATE.fullmatch(day)
        assert shown_dates[i] == ds.dates[i]
        value = float(ds.values[i])
        if math.isnan(value):
            assert text == ""
        else:
            assert DISPLAY_NUMBER.fullmatch(text), text
            assert text != "-0"
            assert Decimal(text) == expected_display(value)
