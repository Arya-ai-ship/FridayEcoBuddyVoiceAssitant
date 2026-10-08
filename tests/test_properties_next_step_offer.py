"""Property test for the next-step offer (task 11.4, Req 8.1, 8.2, 8.3, 8.5)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from friday.agent import templates as t
from friday.agent.narration import CallOutcome, next_step_offer
from friday.data.dataset import Dataset, SessionView

ACTION_TOOLS = (t.DESCRIBE, t.FILL, t.PLOT)
ALL_TOOLS = (t.FETCH, *ACTION_TOOLS)


def _dataset(dataset_id: str, missing: int, rows: int = 3) -> Dataset:
    dates = tuple(date(2024, m, 1) for m in range(1, rows + 1))
    values = np.arange(rows, dtype=np.float64)
    for i in range(min(missing, rows)):
        values[i] = np.nan
    return Dataset(dataset_id, "cpi", "cpi", "CPIAUCSL", dates, values, None)


@dataclass(frozen=True)
class View:
    items: tuple[Dataset, ...] = field(default_factory=tuple)

    @property
    def datasets(self) -> Mapping[str, Dataset]:
        return {ds.dataset_id: ds for ds in self.items}

    def latest_dataset(self) -> Dataset | None:
        return self.items[-1] if self.items else None


def _outcome(tool: str, ok: bool, missing: int = 0) -> CallOutcome:
    return CallOutcome(tool, ok, missing)


@st.composite
def sessions(draw: st.DrawFn) -> SessionView:
    count = draw(st.integers(0, 3))
    items = tuple(_dataset(f"ds-{i + 1}", draw(st.integers(0, 3))) for i in range(count))
    return View(items)


@st.composite
def outcome_lists(draw: st.DrawFn) -> list[CallOutcome]:
    n = draw(st.integers(0, 4))
    out: list[CallOutcome] = []
    for _ in range(n):
        tool = draw(st.sampled_from(ALL_TOOLS))
        ok = draw(st.booleans())
        missing = draw(st.integers(0, 5)) if tool == t.FETCH else 0
        out.append(_outcome(tool, ok, missing))
    return out


@settings(max_examples=300, deadline=None)
@given(outcomes=outcome_lists(), session=sessions())
def test_next_step_offer_follows_the_turns_outcomes(
    outcomes: list[CallOutcome], session: SessionView
) -> None:
    """Feature: friday-voice-data-assistant, Property 18: Next-step offer follows the turn's
    outcomes.

    If any outcome failed, the offer is None. After a fetch, the offer ends with the
    stats-or-plot question and states the Missing count exactly when that Dataset has gaps.
    After stats/fill/plot, it names exactly the other actions, with fill offered only when
    some Session Dataset has a Missing value.

    **Validates: Requirements 8.1, 8.2, 8.3, 8.5**
    """
    offer = next_step_offer(outcomes, session)

    if not outcomes or any(not o.ok for o in outcomes):
        assert offer is None
        return

    last = outcomes[-1]
    assert offer is not None
    if last.tool == t.FETCH:
        assert offer.endswith(t.FETCH_QUESTION)
        if last.missing_count > 0:
            assert t.missing_values_text(last.missing_count) in offer
        else:
            assert offer == t.FETCH_QUESTION
    else:
        has_missing = any(ds.missing_count > 0 for ds in session.datasets.values())
        expected_actions = [
            tool for tool in ACTION_TOOLS if tool != last.tool and (tool != t.FILL or has_missing)
        ]
        if expected_actions:
            for tool in expected_actions:
                assert t.ACTION_LABELS[tool] in offer
            if t.FILL not in expected_actions:
                assert t.ACTION_LABELS[t.FILL] not in offer
        else:
            assert offer is None or offer == ""
