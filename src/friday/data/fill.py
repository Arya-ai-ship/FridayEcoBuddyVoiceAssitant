"""Fill_Tool core: fill Missing_Values into a new Dataset (Req 10.1-10.8, 10.11).

``fill`` never modifies its input. It builds a new ``Dataset`` with the same dates,
indicator, series, transformation, and ``value_column``; ``derived_from`` names the
source and ``fill_method`` records the method. The caller (the Fill_Tool) supplies the
new Dataset_ID, because the Session assigns IDs (Req 6.8, 10.1).

Methods:

- ``forward_fill``: a Missing_Value takes the nearest earlier non-missing value (Req 10.3).
- ``linear_interpolation``: a Missing_Value at ``d`` between non-missing neighbours
  ``(d_a, v_a)`` and ``(d_b, v_b)`` becomes ``v_a + (v_b - v_a) * w`` with the calendar-day
  weight ``w = days(d_a, d) / days(d_a, d_b)`` (Req 10.4).

A Missing_Value with no usable neighbour (a leading gap for ``forward_fill``; a leading or
trailing gap for ``linear_interpolation``) stays Missing and counts as unfilled (Req 10.5).

Numerics: the interpolation uses the spec formula as written so it matches the reference
model. ``v_b - v_a`` can only overflow when the neighbours have opposite signs and huge
magnitudes; for those cells (any non-finite result) the equivalent convex form
``v_a * (1 - w) + v_b * w`` is used instead. Its two terms then have opposite signs, so it
cannot overflow and the filled value stays finite.

This module is pure (stdlib and numpy only).
"""

from dataclasses import replace
from typing import Final, Literal, NamedTuple, get_args

import numpy as np
import numpy.typing as npt

from friday.data.dataset import Dataset, FloatArray

FillMethod = Literal["forward_fill", "linear_interpolation"]
"""A supported Fill_Method."""

FILL_METHODS: Final[tuple[FillMethod, ...]] = get_args(FillMethod)
"""Supported Fill_Methods, in the order error messages list them (Req 10.10)."""

DEFAULT_FILL_METHOD: Final[FillMethod] = "forward_fill"
"""The Fill_Method used when none is given (Req 10.2)."""

IndexArray = npt.NDArray[np.int64]


class FillResult(NamedTuple):
    """The filled Dataset and its gap accounting (``filled + unfilled`` == source Missing)."""

    dataset: Dataset
    filled: int
    unfilled: int


def fill(ds: Dataset, method: FillMethod = DEFAULT_FILL_METHOD, *, dataset_id: str) -> FillResult:
    """Fill ``ds``'s Missing_Values into a new Dataset stored under ``dataset_id``.

    Returns the new Dataset with the counts of filled and still-Missing values. A Dataset
    with no Missing_Values yields identical dates and values with ``filled == unfilled == 0``
    (Req 10.11). Raises ``ValueError`` for an unsupported ``method``; the tool argument
    model rejects those before this is called (Req 10.10).
    """
    if method == "forward_fill":
        values = _forward_fill(ds.values)
    elif method == "linear_interpolation":
        values = _linear_interpolation(ds)
    else:
        raise ValueError(invalid_method_message(method))
    result = replace(
        ds,
        dataset_id=dataset_id,
        values=values,
        derived_from=ds.dataset_id,
        fill_method=method,
    )
    unfilled = result.missing_count
    return FillResult(result, ds.missing_count - unfilled, unfilled)


def invalid_method_message(method: str) -> str:
    """Error text naming an unsupported Fill_Method and listing the supported ones (Req 10.10)."""
    return f"invalid method '{method}'; supported: {', '.join(FILL_METHODS)}"


def _forward_fill(values: FloatArray) -> FloatArray:
    """Carry each non-missing value forward over the Missing_Values after it."""
    prev = _previous_present(~np.isnan(values))
    carried = values[np.maximum(prev, 0)]
    return np.where(prev >= 0, carried, np.nan)


def _linear_interpolation(ds: Dataset) -> FloatArray:
    """Interpolate interior gaps by calendar-day distance between the nearest neighbours."""
    values = ds.values
    present = ~np.isnan(values)
    prev = _previous_present(present)
    nxt = _next_present(present)
    gap = ~present & (prev >= 0) & (nxt < values.shape[0])
    out = values.copy()
    if not gap.any():
        return out
    days = np.array([d.toordinal() for d in ds.dates], dtype=np.int64)
    a, b = prev[gap], nxt[gap]
    w = (days[gap] - days[a]) / (days[b] - days[a])
    out[gap] = _interpolate(values[a], values[b], w)
    return out


def _interpolate(va: FloatArray, vb: FloatArray, w: FloatArray) -> FloatArray:
    """``va + (vb - va) * w``, falling back to the overflow-free convex form where needed."""
    with np.errstate(over="ignore", invalid="ignore"):
        direct = va + (vb - va) * w
        convex = va * (1.0 - w) + vb * w
    return np.where(np.isfinite(direct), direct, convex)


def _previous_present(present: npt.NDArray[np.bool_]) -> IndexArray:
    """Index of the nearest non-missing row at or before each row (-1 when none)."""
    positions = np.where(present, np.arange(present.shape[0], dtype=np.int64), -1)
    return np.maximum.accumulate(positions)


def _next_present(present: npt.NDArray[np.bool_]) -> IndexArray:
    """Index of the nearest non-missing row at or after each row (``len`` when none)."""
    n = present.shape[0]
    positions = np.where(present, np.arange(n, dtype=np.int64), n)
    return np.minimum.accumulate(positions[::-1])[::-1]
