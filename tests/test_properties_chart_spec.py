"""Property test for Plot_Tool chart specs (task 7.3, Req 11.1-11.3, 11.5-11.7, 14.8).

Only the spec is checked (``build_chart_spec``); nothing is rendered. Each requested Dataset
is guaranteed at least one non-missing point inside the drawn date range: one non-missing
"anchor" date is picked per Dataset and the range is drawn to cover every anchor, with each
bound optionally omitted or widened by a few days (so bounds need not be first-of-month).
Indicator names are drawn from a small shared pool part of the time so repeated names (and
their Dataset_ID-suffixed labels) come up regularly.
"""

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from datetime import date, timedelta

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from friday.constants import MAX_PLOT_SERIES, MAX_TITLE_CHARS
from friday.data.dataset import Dataset
from friday.data.plot import MULTI_Y_LABEL, NAME_SEPARATOR, X_LABEL, build_chart_spec
from friday.style import NEON_PALETTE
from strategies import datasets, finite_floats, indicator_names

MAX_TEST_ROWS = 120
"""Rows per Dataset; enough for clipping and gaps while keeping examples fast."""
MAX_DISTRACTORS = 2
"""Extra Session Datasets that are not requested."""
MAX_BOUND_SLACK_DAYS = 400
"""How far a range bound may sit beyond the outermost anchor date."""
SHARED_NAMES = ("inflation", "unemployment")
"""Small name pool so several requested Datasets often share an Indicator name."""


@dataclass(frozen=True)
class PlotCase:
    """A Session, a valid ``plot_data`` request against it, and the raw range bounds."""

    session: dict[str, Dataset]
    ids: tuple[str, ...]
    title: str | None
    start: date | None
    end: date | None


def _with_point(draw: st.DrawFn, ds: Dataset, dataset_id: str) -> Dataset:
    """``ds`` under a new ID and a maybe-shared name, with at least one non-missing value."""
    indicator = draw(st.sampled_from(SHARED_NAMES) | indicator_names())
    values = ds.values.copy()
    if not np.any(~np.isnan(values)):
        values[draw(st.integers(0, ds.row_count - 1))] = draw(finite_floats())
    return replace(
        ds, dataset_id=dataset_id, indicator=indicator, value_column=indicator, values=values
    )


def _bound(draw: st.DrawFn, anchor: date, direction: int) -> date | None:
    """``None`` (unbounded) or a date at or beyond ``anchor`` in ``direction`` (+1/-1)."""
    if draw(st.booleans()):
        return None
    return anchor + timedelta(days=direction * draw(st.integers(0, MAX_BOUND_SLACK_DAYS)))


@st.composite
def plot_cases(draw: st.DrawFn) -> PlotCase:
    """1-5 distinct requested Datasets (plus distractors), an optional title and range."""
    count = draw(st.integers(1, MAX_PLOT_SERIES))
    sources = draw(
        st.lists(datasets(min_rows=1, max_rows=MAX_TEST_ROWS), min_size=count, max_size=count)
    )
    requested = [_with_point(draw, ds, f"ds-{i + 1}") for i, ds in enumerate(sources)]
    distractors = draw(st.lists(datasets(max_rows=MAX_TEST_ROWS), max_size=MAX_DISTRACTORS))
    others = [replace(ds, dataset_id=f"other-{i}") for i, ds in enumerate(distractors)]
    anchors = [
        ds.dates[draw(st.sampled_from(np.flatnonzero(~np.isnan(ds.values)).tolist()))]
        for ds in requested
    ]
    ids = tuple(draw(st.permutations([ds.dataset_id for ds in requested])))
    title = draw(
        st.none() | st.sampled_from(("", "   ")) | st.text(min_size=1, max_size=MAX_TITLE_CHARS)
    )
    return PlotCase(
        session={ds.dataset_id: ds for ds in [*others, *requested]},
        ids=ids,
        title=title,
        start=_bound(draw, min(anchors), -1),
        end=_bound(draw, max(anchors), +1),
    )


def _snapshot(ds: Dataset) -> dict[str, object]:
    """Every field of ``ds``, with ``values`` as raw bytes so NaNs compare equal."""
    snap: dict[str, object] = {f.name: getattr(ds, f.name) for f in fields(ds)}
    snap["values"] = ds.values.tobytes()
    return snap


def _in_range(day: date, start: date | None, end: date | None) -> bool:
    return (start is None or day >= start) and (end is None or day <= end)


def _expected_label(ds: Dataset, requested: Sequence[Dataset]) -> str:
    repeated = sum(other.indicator == ds.indicator for other in requested) > 1
    return f"{ds.indicator} ({ds.dataset_id})" if repeated else ds.indicator


@settings(max_examples=200, deadline=None)
@given(case=plot_cases())
def test_chart_spec_reflects_the_requested_series(case: PlotCase) -> None:
    """Feature: friday-voice-data-assistant, Property 16: Chart spec reflects the requested
    series.

    For any 1-5 distinct Session Datasets and any optional title (at most 100 chars) and
    valid date range, ``build_chart_spec`` produces exactly one series per Dataset_ID in
    request order, each limited to dates within the range with NaN kept at the remaining
    Missing positions; pairwise-distinct Neon_Accent colors assigned by index; labels that
    contain the Indicator name; a legend only for 2+ series; a y label of the Indicator for
    1 series and "Value" otherwise; the supplied title or else the Indicator names joined;
    and alt text naming every Indicator and the first and last plotted dates. The input
    Datasets are left unchanged.

    **Validates: Requirements 11.1, 11.2, 11.3, 11.5, 11.6, 11.7, 14.8**
    """
    before = {key: _snapshot(ds) for key, ds in case.session.items()}
    requested = [case.session[i] for i in case.ids]
    start = case.start.isoformat() if case.start is not None else None
    end = case.end.isoformat() if case.end is not None else None

    spec = build_chart_spec(case.session, case.ids, case.title, start, end)

    # One series per requested ID, in request order (Req 11.1).
    assert spec.dataset_ids == case.ids
    assert len(spec.series) == len(requested)

    plotted: list[date] = []
    for index, (series, ds) in enumerate(zip(spec.series, requested, strict=True)):
        # Range-limited dates; values bit-for-bit, so Missing stays NaN (Req 11.5, 11.7).
        mask = np.array([_in_range(d, case.start, case.end) for d in ds.dates], dtype=bool)
        assert series.dates == tuple(d for d, keep in zip(ds.dates, mask, strict=True) if keep)
        assert all(_in_range(d, case.start, case.end) for d in series.dates)
        np.testing.assert_array_equal(
            series.values.view(np.uint64), ds.values[mask].view(np.uint64)
        )
        assert not series.values.flags.writeable
        present = ~np.isnan(series.values)
        assert present.any()
        plotted.extend(d for d, keep in zip(series.dates, present, strict=True) if keep)

        # Color by index from the Neon_Accent palette (Req 14.8).
        assert series.color == NEON_PALETTE[index]

        # Legend label names the Indicator, plus the ID when names repeat (Req 11.3).
        assert series.indicator == ds.indicator
        assert ds.indicator in series.label
        assert series.label == _expected_label(ds, requested)

    colors = [s.color for s in spec.series]
    assert len(set(colors)) == len(colors)
    assert set(colors) <= set(NEON_PALETTE)

    # Axes labels, legend flag, and title (Req 11.2, 11.3).
    assert spec.x_label == X_LABEL
    assert spec.y_label == (requested[0].indicator if len(requested) == 1 else MULTI_Y_LABEL)
    assert spec.show_legend == (len(requested) >= 2)
    names = list(dict.fromkeys(ds.indicator for ds in requested))
    if case.title is not None and case.title.strip():
        assert spec.title == case.title
    else:
        assert spec.title == NAME_SEPARATOR.join(names)

    # Plotted bounds and alt text (Req 11.6).
    assert (spec.first_date, spec.last_date) == (min(plotted), max(plotted))
    assert _in_range(spec.first_date, case.start, case.end)
    assert _in_range(spec.last_date, case.start, case.end)
    assert all(name in spec.alt_text for name in names)
    assert spec.first_date.isoformat() in spec.alt_text
    assert spec.last_date.isoformat() in spec.alt_text

    # Inputs unchanged (Req 11.4 via the Property's "Session Datasets").
    assert {key: _snapshot(ds) for key, ds in case.session.items()} == before
