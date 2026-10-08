"""Descriptive_Statistics for a Dataset and the stats-table display rows (Req 9.1-9.9).

``describe`` excludes Missing_Values (NaN) from every statistic except the Missing count.
Mean, sample standard deviation, and the linear-interpolation percentiles are computed in
exact rational arithmetic (``fractions.Fraction``) over the non-missing values and rounded
once to float. This keeps them deterministic and accurate under cancellation, and means
extreme finite floats (near ``±1.8e308``) never overflow an intermediate sum, square, or
percentile difference. A statistic whose true value is not representable as a finite float
(for example the std of ``[-1.7e308, 1.7e308]``) is reported as undefined (``None``). If any
non-missing value is infinite, every statistic other than the counts and dates is undefined.

The 11 row labels are table headings, like the Preview_Table's ``date`` column, so they live
here rather than in ``agent/templates.py`` (which holds narration and outcome texts).
This module is pure (stdlib and numpy only).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from typing import Final, NamedTuple

import numpy as np

from friday.constants import STATS_DECIMALS
from friday.data.dataset import Dataset

UNDEFINED_TEXT: Final = "undefined"
"""How an undefined statistic (``None``) appears in the stats table."""

STATS_LABELS: Final = (
    "Count",
    "Missing values",
    "Mean",
    "Std. dev.",
    "Min",
    "25th percentile",
    "Median",
    "75th percentile",
    "Max",
    "First date",
    "Last date",
)
"""Stats-table row labels in the Descriptive_Statistics order."""

_QUARTILES: Final = (25, 50, 75)
"""Percentiles reported as p25, median, and p75."""

_SQRT_GUARD_BITS: Final = 128
"""Minimum bits of the scaled radicand, so the integer square root keeps >= 64 bits."""


@dataclass(frozen=True)
class Stats:
    """Descriptive_Statistics of one Dataset's value column (``None`` means undefined)."""

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


class StatsRow(NamedTuple):
    """One stats-table row: label and display value."""

    label: str
    value: str


def describe(ds: Dataset) -> Stats:
    """Compute the Descriptive_Statistics of ``ds`` without modifying it (Req 9.1-9.3, 9.6-9.9).

    Std is the sample std (ddof = 1) and is ``None`` when fewer than 2 values are
    non-missing. With 0 non-missing values every statistic except the counts is ``None``.
    First and last date are the dates of the earliest and latest non-missing rows.
    """
    present = np.flatnonzero(~np.isnan(ds.values))
    count = int(present.size)
    missing_count = ds.row_count - count
    if count == 0:
        return _undefined_stats(missing_count, None, None)
    first_date, last_date = ds.dates[int(present[0])], ds.dates[int(present[-1])]
    ordered: list[float] = np.sort(ds.values[present]).tolist()
    if not all(math.isfinite(value) for value in ordered):
        return _undefined_stats(missing_count, first_date, last_date, count=count)
    exact = [Fraction(value) for value in ordered]
    mean = sum(exact, Fraction(0)) / count
    p25, median, p75 = (_percentile(exact, q) for q in _QUARTILES)
    return Stats(
        count=count,
        missing_count=missing_count,
        mean=float(mean),
        std=_sample_std(exact, mean),
        min=ordered[0],
        p25=p25,
        median=median,
        p75=p75,
        max=ordered[-1],
        first_date=first_date,
        last_date=last_date,
    )


def format_stats_rows(stats: Stats) -> tuple[StatsRow, ...]:
    """Return the 11 stats-table rows in the Descriptive_Statistics order (Req 9.4).

    Counts are integers, other numbers have exactly ``STATS_DECIMALS`` decimals, dates are
    ISO ``YYYY-MM-DD``, and undefined values are ``"undefined"``.
    """
    values = (
        str(stats.count),
        str(stats.missing_count),
        *(
            _format_number(value)
            for value in (
                stats.mean,
                stats.std,
                stats.min,
                stats.p25,
                stats.median,
                stats.p75,
                stats.max,
            )
        ),
        _format_date(stats.first_date),
        _format_date(stats.last_date),
    )
    return tuple(StatsRow(label, value) for label, value in zip(STATS_LABELS, values, strict=True))


# --- Statistics helpers (exact arithmetic) -----------------------------------


def _undefined_stats(
    missing_count: int, first_date: date | None, last_date: date | None, *, count: int = 0
) -> Stats:
    """Stats whose numeric statistics are all undefined (no or non-finite values)."""
    return Stats(
        count=count,
        missing_count=missing_count,
        mean=None,
        std=None,
        min=None,
        p25=None,
        median=None,
        p75=None,
        max=None,
        first_date=first_date,
        last_date=last_date,
    )


def _percentile(ordered: Sequence[Fraction], q: int) -> float:
    """Linear-interpolation percentile ``q`` of sorted values (numpy's ``linear`` method).

    The virtual index is ``(n - 1) * q / 100``; the result lies between the two closest
    ranks, so it is always a finite float.
    """
    position = Fraction((len(ordered) - 1) * q, 100)
    low = math.floor(position)
    weight = position - low
    if weight == 0:
        return float(ordered[low])
    return float(ordered[low] + (ordered[low + 1] - ordered[low]) * weight)


def _sample_std(exact: Sequence[Fraction], mean: Fraction) -> float | None:
    """Sample std (ddof = 1); ``None`` below 2 values or when it exceeds the float range."""
    if len(exact) < 2:
        return None
    variance = sum(((value - mean) ** 2 for value in exact), Fraction(0)) / (len(exact) - 1)
    return _sqrt(variance)


def _sqrt(radicand: Fraction) -> float | None:
    """Square root of a non-negative rational, accurate to about 1 ulp.

    The radicand is scaled by ``4**k`` so its integer square root has at least 64 bits,
    then divided by ``2**k`` with correctly rounded integer division. Returns ``None``
    when the root is too large for a float.
    """
    numerator, denominator = radicand.numerator, radicand.denominator
    if numerator == 0:
        return 0.0
    magnitude = numerator.bit_length() - denominator.bit_length()
    half_shift = max(0, _SQRT_GUARD_BITS - magnitude + 1) // 2
    root = math.isqrt((numerator << (2 * half_shift)) // denominator)
    try:
        return root / (1 << half_shift)
    except OverflowError:
        return None


# --- Display formatting ------------------------------------------------------


def _format_number(value: float | None) -> str:
    """A non-count statistic with exactly ``STATS_DECIMALS`` decimals, or ``"undefined"``."""
    if value is None:
        return UNDEFINED_TEXT
    text = f"{value:.{STATS_DECIMALS}f}"
    return text.removeprefix("-") if float(text) == 0 else text


def _format_date(value: date | None) -> str:
    """An ISO ``YYYY-MM-DD`` date, or ``"undefined"``."""
    return UNDEFINED_TEXT if value is None else value.isoformat()
