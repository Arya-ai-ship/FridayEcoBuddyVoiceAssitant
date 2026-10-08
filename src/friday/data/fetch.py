"""Pure Fetch_Tool core: build a Dataset from FRED observations and apply YoY (Req 6.3-6.7, 6.13).

``build_dataset`` turns raw FRED observations (``{"date": "YYYY-MM-DD", "value": "<number>"
| "."}`` mappings, as in the FRED JSON API) into an untransformed Dataset. ``apply_yoy`` is
the YoY_Transformation on a value array, and ``transform`` applies an Indicator's
Transformation to a whole Dataset. The Session assigns Dataset_IDs, so callers pass one in.
This module is pure (stdlib and numpy only).
"""

from collections.abc import Iterable, Mapping
from dataclasses import replace
from datetime import date
from typing import Final

import numpy as np
import numpy.typing as npt

from friday.constants import YOY_LAG_ROWS
from friday.data.dataset import Dataset, FloatArray
from friday.data.indicators import IndicatorEntry, Transformation
from friday.errors import RangeTooShortForYoY

Observation = Mapping[str, str]
"""One FRED observation; only its ``date`` and ``value`` keys are read."""

FRED_MISSING: Final = "."
"""FRED's placeholder for a missing observation value."""


def build_dataset(
    observations: Iterable[Observation],
    entry: IndicatorEntry,
    start: date | None,
    end: date | None,
    *,
    dataset_id: str,
) -> Dataset:
    """Build the untransformed Dataset for ``entry`` from FRED ``observations`` (Req 6.3-6.5).

    Keeps the observations dated on or after ``start`` and on or before ``end`` (``None``
    means unbounded), sorts them by ascending date, maps ``.`` to a Missing_Value (NaN), and
    names the value column after the canonical Indicator. The entry's Transformation is not
    applied here; see ``transform``. Raises ``ValueError`` for a malformed date or value, or
    for duplicate dates.
    """
    parsed = (_parse_observation(obs) for obs in observations)
    rows = sorted((row for row in parsed if _within(row[0], start, end)), key=_row_date)
    return Dataset(
        dataset_id=dataset_id,
        indicator=entry.name,
        value_column=entry.name,
        series_id=entry.series_id,
        dates=tuple(day for day, _ in rows),
        values=np.array([value for _, value in rows], dtype=np.float64),
        transformation=None,
    )


def apply_yoy(values: npt.ArrayLike) -> FloatArray:
    """Return the YoY_Transformation of a monthly value sequence (Req 6.6, 6.7, 6.13).

    Row ``i`` of the result is ``(x[i + 12] / x[i] - 1) * 100`` by row position, so the
    result has ``len(x) - 12`` rows (the first 12 rows of ``x`` are dropped). A row is a
    Missing_Value when either input is Missing or the denominator is 0. The result is a new
    read-only array. Raises ``RangeTooShortForYoY`` when ``x`` has 12 or fewer rows.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"values must be 1-D, got {x.ndim} dimensions")
    if x.shape[0] <= YOY_LAG_ROWS:
        raise RangeTooShortForYoY()
    current = x[YOY_LAG_ROWS:]
    previous = x[:-YOY_LAG_ROWS]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = (current / previous - 1.0) * 100.0
    result: FloatArray = np.where(previous == 0.0, np.nan, ratio)
    result.setflags(write=False)
    return result


def transform(dataset: Dataset, transformation: Transformation | None) -> Dataset:
    """Apply an Indicator's Transformation to an untransformed Dataset (Req 6.6, 6.13).

    With ``None`` the Dataset is returned unchanged. With ``"yoy"`` a new Dataset is built
    from ``apply_yoy(values)`` and the dates of the rows it keeps (the first 12 are dropped),
    under the same Dataset_ID. Raises ``RangeTooShortForYoY`` when no rows remain.
    """
    if transformation is None:
        return dataset
    return replace(
        dataset,
        dates=dataset.dates[YOY_LAG_ROWS:],
        values=apply_yoy(dataset.values),
        transformation=transformation,
    )


def _parse_observation(observation: Observation) -> tuple[date, float]:
    """Parse one observation's date and value (``.`` becomes NaN)."""
    raw_value = observation["value"]
    value = np.nan if raw_value == FRED_MISSING else float(raw_value)
    return date.fromisoformat(observation["date"]), value


def _row_date(row: tuple[date, float]) -> date:
    """Sort key: the row's date (values may be NaN, which does not order)."""
    return row[0]


def _within(day: date, start: date | None, end: date | None) -> bool:
    """True when ``day`` is on or after ``start`` and on or before ``end`` (None = open)."""
    return (start is None or day >= start) and (end is None or day <= end)
