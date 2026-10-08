"""Property test for Dataset construction from FRED observations (design: Property 2)."""

import math
from datetime import date, timedelta

import numpy as np
from hypothesis import event, given, settings
from hypothesis import strategies as st

from friday.data.dataset import DATE_COLUMN
from friday.data.fetch import build_dataset
from friday.data.indicators import IndicatorEntry
from strategies import fred_observations, indicator_names, series_ids

Observations = list[dict[str, str]]
Bounds = tuple[date | None, date | None]

# Bounds may sit this far before or after the observation span, so some exclude everything.
_SPAN_MARGIN = timedelta(days=400)


def _oracle_value(text: str) -> float:
    """FRED's ``.`` placeholder is a Missing_Value (NaN); anything else is a number."""
    return math.nan if text == "." else float(text)


def _oracle_rows(observations: Observations, start: date | None, end: date | None) -> Observations:
    """Observations within the inclusive bounds (``None`` = unbounded), by ascending date.

    Compares the ISO ``YYYY-MM-DD`` strings directly: for four-digit years their
    lexicographic order is the calendar order, independent of ``date`` parsing.
    """
    lo = start.isoformat() if start is not None else None
    hi = end.isoformat() if end is not None else None
    kept = [
        o
        for o in observations
        if (lo is None or o["date"] >= lo) and (hi is None or o["date"] <= hi)
    ]
    return sorted(kept, key=lambda o: o["date"])


@st.composite
def bound_dates(draw: st.DrawFn, observations: Observations) -> date:
    """A date on an observation, one day either side of one, or anywhere near the span."""
    days = [date.fromisoformat(o["date"]) for o in observations]
    if not days:
        return draw(st.dates(date(1900, 1, 1), date(2100, 12, 31)))
    near = st.sampled_from(days)
    off_by_one = st.tuples(near, st.sampled_from((-1, 1))).map(
        lambda pair: pair[0] + timedelta(days=pair[1])
    )
    anywhere = st.dates(min(days) - _SPAN_MARGIN, max(days) + _SPAN_MARGIN)
    return draw(st.one_of(near, off_by_one, anywhere))


@st.composite
def bounds(draw: st.DrawFn, observations: Observations) -> Bounds:
    """Optional start/end bounds; when both are given, start is not later than end."""
    optional = st.none() | bound_dates(observations)
    start, end = draw(optional), draw(optional)
    if start is not None and end is not None and start > end:
        start, end = end, start
    return start, end


@st.composite
def entries(draw: st.DrawFn) -> IndicatorEntry:
    """An Indicator entry; its Transformation must not affect the untransformed Dataset."""
    return IndicatorEntry(
        name=draw(indicator_names()),
        aliases=(),
        series_id=draw(series_ids()),
        transformation=draw(st.sampled_from((None, "yoy"))),
    )


@st.composite
def cases(draw: st.DrawFn) -> tuple[Observations, IndicatorEntry, Bounds]:
    """Shuffled FRED observations, an entry, and bounds drawn relative to the observations."""
    observations = draw(fred_observations())
    return observations, draw(entries()), draw(bounds(observations))


@settings(max_examples=200)
@given(cases())
def test_dataset_construction_filters_sorts_and_maps_placeholders(
    case: tuple[Observations, IndicatorEntry, Bounds],
) -> None:
    """Feature: friday-voice-data-assistant, Property 2: Dataset construction filters, sorts,
    and maps placeholders.

    **Validates: Requirements 6.3, 6.4, 6.5**
    """
    observations, entry, (start, end) = case
    expected = _oracle_rows(observations, start, end)
    kept = "all" if len(expected) == len(observations) else "some" if expected else "none"
    event(f"bounds kept {kept} observations")

    ds = build_dataset(observations, entry, start, end, dataset_id="ds-1")

    assert [d.isoformat() for d in ds.dates] == [o["date"] for o in expected]
    assert all(a < b for a, b in zip(ds.dates, ds.dates[1:], strict=False))
    assert (DATE_COLUMN, ds.value_column) == ("date", entry.name)
    assert ds.indicator == entry.name
    assert ds.series_id == entry.series_id
    assert ds.dataset_id == "ds-1"
    assert ds.transformation is None

    expected_values = np.array([_oracle_value(o["value"]) for o in expected], dtype=np.float64)
    assert np.isnan(ds.values).tolist() == [o["value"] == "." for o in expected]
    np.testing.assert_array_equal(ds.values, expected_values)
