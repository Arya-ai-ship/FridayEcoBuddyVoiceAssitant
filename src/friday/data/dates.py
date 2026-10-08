"""Strict ``YYYY-MM-DD`` date-range parsing shared by the Fetch_Tool and Plot_Tool.

Requirements 6.12 and 11.9: a supplied start or end date must be a real calendar date in
exactly ``YYYY-MM-DD`` form, and the start date must not be later than the end date.
``date.fromisoformat`` alone is too lenient (since Python 3.11 it also accepts forms such
as ``20240101`` and ISO week dates), so the shape is checked explicitly first.
This module is pure.
"""

import re
from datetime import date
from typing import Final

from friday.errors import DateRangeError

START_FIELD: Final = "start_date"
"""Tool argument name of the start bound, used to name the bad input in errors."""
END_FIELD: Final = "end_date"
"""Tool argument name of the end bound, used to name the bad input in errors."""

# ASCII-only digits: ``\d`` would otherwise match non-ASCII Unicode digits.
ISO_DATE_PATTERN: Final = "[0-9]{4}-[0-9]{2}-[0-9]{2}"
"""The ``YYYY-MM-DD`` shape (unanchored); the tool argument schemas anchor it with ``^...$``."""
_ISO_DATE: Final = re.compile(ISO_DATE_PATTERN)


def parse_date(raw: str, field: str) -> date:
    """Parse ``raw`` as a strict ``YYYY-MM-DD`` calendar date.

    Raises ``DateRangeError`` naming ``field`` and the offending value when ``raw`` does not
    have the exact ``YYYY-MM-DD`` shape or is not a real calendar date (e.g. ``2023-02-30``).
    """
    if _ISO_DATE.fullmatch(raw) is None:
        raise DateRangeError(field, f"invalid {field} {raw!r}: expected format YYYY-MM-DD")
    try:
        return date.fromisoformat(raw)
    except ValueError:
        raise DateRangeError(field, f"invalid {field} {raw!r}: not a valid calendar date") from None


def parse_date_range(start: str | None, end: str | None) -> tuple[date | None, date | None]:
    """Parse optional start/end bounds; ``None`` means unbounded on that side.

    Returns ``(start_date, end_date)``. Raises ``DateRangeError`` naming the bad input when a
    supplied bound is malformed or not a calendar date, or when start is later than end
    (Req 6.12, 11.9).
    """
    start_date = parse_date(start, START_FIELD) if start is not None else None
    end_date = parse_date(end, END_FIELD) if end is not None else None
    if start_date is not None and end_date is not None and start_date > end_date:
        raise DateRangeError(
            START_FIELD,
            f"invalid date range: {START_FIELD} {start!r} is later than {END_FIELD} {end!r}",
        )
    return start_date, end_date
