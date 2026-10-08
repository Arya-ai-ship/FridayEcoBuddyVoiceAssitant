"""Datasets, Preview_Table rows, and the CSV_Export format (Req 6.5, 7.1-7.3, 7.5, 7.6).

A ``Dataset`` is immutable: ``dates`` is a tuple and ``values`` is a read-only float64
array in which NaN marks a Missing_Value. Tools never modify a Dataset; they build new ones.
This module is pure (stdlib and numpy only).
"""

import csv
import io
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Final, NamedTuple, Protocol

import numpy as np
import numpy.typing as npt

from friday.constants import PREVIEW_DECIMALS, PREVIEW_ROWS

FloatArray = npt.NDArray[np.float64]
"""A 1-D float64 array; NaN marks a Missing_Value."""

DATE_COLUMN: Final = "date"
"""Name of the date column in previews and CSV_Exports."""

MISSING_TEXT: Final = ""
"""How a Missing_Value appears in a Preview_Table cell and a CSV_Export field."""

PreviewRow = tuple[str, str]
"""One Preview_Table row: ISO date and display value (``""`` for Missing)."""


@dataclass(frozen=True, eq=False)
class Dataset:
    """One immutable table with a ``date`` column and one numeric value column.

    ``values`` is copied on construction into a read-only float64 array, so a Dataset
    never aliases a caller's writable buffer. ``dates`` must be strictly ascending and
    have the same length as ``values``. Equality is identity (arrays have no total ``==``).
    """

    dataset_id: str
    indicator: str
    value_column: str
    series_id: str
    dates: tuple[date, ...]
    values: FloatArray
    transformation: str | None
    derived_from: str | None = None
    fill_method: str | None = None

    def __post_init__(self) -> None:
        """Freeze ``values`` and check the shape and date-order invariants."""
        dates = tuple(self.dates)
        values = np.array(self.values, dtype=np.float64, copy=True)
        if values.ndim != 1:
            raise ValueError(f"values must be 1-D, got {values.ndim} dimensions")
        if len(dates) != values.shape[0]:
            raise ValueError(f"{len(dates)} dates but {values.shape[0]} values")
        if any(earlier >= later for earlier, later in zip(dates, dates[1:], strict=False)):
            raise ValueError("dates must be strictly ascending")
        values.setflags(write=False)
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "values", values)

    @property
    def row_count(self) -> int:
        """Number of rows, Missing_Values included."""
        return len(self.dates)

    @property
    def missing_count(self) -> int:
        """Number of rows whose value is a Missing_Value."""
        return int(np.count_nonzero(np.isnan(self.values)))

    @property
    def missing_dates(self) -> tuple[date, ...]:
        """The dates of the rows whose value is a Missing_Value, in ascending order."""
        gaps = np.isnan(self.values)
        return tuple(
            day for day, is_missing in zip(self.dates, gaps.tolist(), strict=True) if is_missing
        )


class SessionView(Protocol):
    """Read-only view of a Session's Datasets for pure code (narration, prompts).

    ``session.Session`` satisfies it structurally; it is defined here so the domain layer
    can depend on it without importing the application layer.
    """

    @property
    def datasets(self) -> Mapping[str, Dataset]:
        """Datasets by Dataset_ID in insertion order (the last one is the most recent)."""
        ...

    def latest_dataset(self) -> Dataset | None:
        """The most recently inserted Dataset, or ``None`` when there is none (Req 8.4)."""
        ...


# --- Preview_Table -----------------------------------------------------------


def preview_rows(ds: Dataset, n: int = PREVIEW_ROWS) -> tuple[PreviewRow, ...]:
    """Return the first ``min(n, row_count)`` rows in display format (Req 7.1-7.3).

    Dates are ISO ``YYYY-MM-DD``; values are rounded to at most ``PREVIEW_DECIMALS``
    decimals with trailing zeros dropped; Missing_Values are ``""``.
    """
    count = max(0, min(n, ds.row_count))
    return tuple(
        (ds.dates[i].isoformat(), format_display_value(float(ds.values[i]))) for i in range(count)
    )


def format_display_value(value: float) -> str:
    """Format one value for a Preview_Table cell (``""`` for a Missing_Value)."""
    if math.isnan(value):
        return MISSING_TEXT
    if math.isinf(value):
        return repr(value)
    text = f"{value:.{PREVIEW_DECIMALS}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


# --- CSV_Export --------------------------------------------------------------


class CsvTable(NamedTuple):
    """The parsed contents of a CSV_Export: header, dates, and values (NaN == Missing)."""

    columns: tuple[str, str]
    dates: tuple[date, ...]
    values: FloatArray


def to_csv(ds: Dataset) -> str:
    """Render ``ds`` as a CSV_Export (Req 7.5).

    The header is ``date,<value_column>``. Rows are in ascending date order, dates are ISO,
    values use ``repr(float)`` (the shortest form that round-trips, unrounded), and
    Missing_Values are empty fields.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow((DATE_COLUMN, ds.value_column))
    for day, value in zip(ds.dates, ds.values.tolist(), strict=True):
        writer.writerow((day.isoformat(), _csv_field(value)))
    return buffer.getvalue()


def parse_csv(text: str) -> CsvTable:
    """Parse a CSV_Export produced by ``to_csv`` (the inverse used for Req 7.6).

    Raises ``ValueError`` naming the line when the header or a row is malformed.
    """
    reader = csv.reader(io.StringIO(text, newline=""))
    header = next(reader, None)
    if header is None or len(header) != 2 or header[0] != DATE_COLUMN:
        raise ValueError(f"header must be '{DATE_COLUMN},<value column>', got {header!r}")
    dates: list[date] = []
    values: list[float] = []
    for row in reader:
        line = reader.line_num
        if len(row) != 2:
            raise ValueError(f"line {line}: expected 2 fields, got {len(row)}")
        try:
            dates.append(date.fromisoformat(row[0]))
            values.append(math.nan if row[1] == MISSING_TEXT else float(row[1]))
        except ValueError as exc:
            raise ValueError(f"line {line}: {exc}") from None
    columns = (header[0], header[1])
    return CsvTable(columns, tuple(dates), np.array(values, dtype=np.float64))


def _csv_field(value: float) -> str:
    """One CSV value field: empty for a Missing_Value, otherwise ``repr(float)``."""
    return MISSING_TEXT if math.isnan(value) else repr(float(value))
