"""Unit tests for ``friday.agent.prompts``: rules, persona, and the Dataset inventory.

Covers Requirements 4.1 (Friday persona, "Boss"), 4.2/4.3 (reply format and spoken
limits), 5.6 (fixed instructions), 6.9/6.10 (error relaying), and 8.4 (inventory with the
most recent Dataset marked).
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pytest

from friday.agent import prompts as p
from friday.agent import templates as t
from friday.agent.tool_args import DescribeArgs, FillArgs, PlotArgs
from friday.constants import MAX_SPOKEN_SENTENCES, MAX_SPOKEN_WORDS, MAX_TOOL_CALLS
from friday.data.dataset import Dataset
from friday.data.plot import IDS_FIELD
from friday.errors import FredError, UnknownIndicator

PLACEHOLDER = re.compile(r"\{[^{}]*\}")
NAN = float("nan")


def _dataset(
    dataset_id: str,
    values: list[float],
    *,
    indicator: str = "inflation",
    derived_from: str | None = None,
    fill_method: str | None = None,
) -> Dataset:
    dates = tuple(date(2024, month, 1) for month in range(1, len(values) + 1))
    array = np.array(values, dtype=np.float64)
    return Dataset(
        dataset_id, indicator, indicator, "CPIAUCSL", dates, array, None, derived_from, fill_method
    )


@dataclass(frozen=True)
class View:
    """A minimal real ``SessionView`` over a fixed list of Datasets."""

    items: tuple[Dataset, ...] = field(default_factory=tuple[Dataset, ...])

    @property
    def datasets(self) -> Mapping[str, Dataset]:
        return {ds.dataset_id: ds for ds in self.items}

    def latest_dataset(self) -> Dataset | None:
        return self.items[-1] if self.items else None


# --- FRIDAY_RULES (harness_instructions) -------------------------------------


@pytest.mark.parametrize("rule", p.RULES)
def test_every_rule_is_in_friday_rules(rule: str) -> None:
    assert rule in p.FRIDAY_RULES


def test_numbers_rule_forbids_invented_numbers() -> None:
    assert "tool results" in p.NUMBERS_RULE
    assert "Never invent" in p.NUMBERS_RULE


def test_tools_rule_names_exactly_the_four_tools() -> None:
    for name in t.TOOL_NAMES:
        assert name in p.TOOLS_RULE
    assert "exactly four tools" in p.TOOLS_RULE
    assert str(MAX_TOOL_CALLS) in p.TOOLS_RULE


def test_format_rule_gives_the_reply_shape_and_spoken_limits() -> None:
    assert "<display>…</display><spoken>…</spoken>" in p.FORMAT_RULE
    assert f"at most {MAX_SPOKEN_SENTENCES} short sentences" in p.FORMAT_RULE
    assert str(MAX_SPOKEN_WORDS) in p.FORMAT_RULE
    assert p._SPOKEN_TARGET_WORDS < MAX_SPOKEN_WORDS  # pyright: ignore[reportPrivateUsage]
    for banned in ("tables", "CSV contents", "raw data rows"):
        assert banned in p.FORMAT_RULE


def test_format_rule_leaves_status_lines_to_the_app() -> None:
    assert t.start_text(t.FETCH, "inflation") in p.FORMAT_RULE
    assert t.DONE_LINE in p.FORMAT_RULE


def test_no_offers_rule_forbids_next_step_offers() -> None:
    assert "Do not offer next steps" in p.NO_OFFERS_RULE
    assert "retry" in p.NO_OFFERS_RULE


def test_arguments_rule_covers_dates_and_most_recent_dataset() -> None:
    assert "YYYY-MM-DD" in p.ARGUMENTS_RULE
    assert f"Omit {p.DATASET_ID_ARG}" in p.ARGUMENTS_RULE
    assert IDS_FIELD in p.ARGUMENTS_RULE
    assert "most recent dataset" in p.ARGUMENTS_RULE


def test_argument_names_match_the_tool_models() -> None:
    assert p.DATASET_ID_ARG in DescribeArgs.model_fields
    assert p.DATASET_ID_ARG in FillArgs.model_fields
    assert IDS_FIELD in PlotArgs.model_fields


def test_errors_rule_relays_supported_names_and_offers_retry() -> None:
    for kind in UnknownIndicator.KINDS | FredError.KINDS:
        assert kind in p.ERRORS_RULE
    assert p.SUPPORTED_FIELD in p.ERRORS_RULE
    assert p.SUPPORTED_FIELD in UnknownIndicator("x", ["inflation"]).details()
    assert "offer to retry" in p.ERRORS_RULE


def test_rules_are_numbered_in_order() -> None:
    lines = p.FRIDAY_RULES.splitlines()
    assert lines[0] == "Rules:"
    assert lines[1:] == [f"{n}. {rule}" for n, rule in enumerate(p.RULES, start=1)]


# --- FRIDAY_PERSONA (agent_instructions) -------------------------------------


def test_persona_names_friday_and_allows_boss_occasionally() -> None:
    assert "You are Friday" in p.FRIDAY_PERSONA
    assert f'"{t.BOSS}"' in p.FRIDAY_PERSONA
    assert "never in every reply" in p.FRIDAY_PERSONA


@pytest.mark.parametrize("text", [p.FRIDAY_RULES, p.FRIDAY_PERSONA])
def test_no_unfilled_placeholders(text: str) -> None:
    assert PLACEHOLDER.search(text) is None


# --- dataset_inventory (Req 8.4) ---------------------------------------------


def test_empty_inventory_says_no_datasets() -> None:
    assert p.dataset_inventory(View()) == f"{p.INVENTORY_HEADER}\n{p.NO_DATASETS_TEXT}"
    assert "No datasets are loaded yet." in p.dataset_inventory(View())


def test_inventory_lists_each_dataset_and_marks_only_the_most_recent() -> None:
    fetched = _dataset("ds-1", [1.0, NAN, 3.0])
    filled = _dataset("ds-2", [1.0, 1.0, 3.0], derived_from="ds-1", fill_method="forward_fill")
    text = p.dataset_inventory(View((fetched, filled)))
    assert text.splitlines() == [
        p.INVENTORY_HEADER,
        "- ds-1: inflation (CPIAUCSL), 3 rows, 2024-01-01 to 2024-03-01, 1 missing value",
        "- ds-2: inflation (CPIAUCSL), 3 rows, 2024-01-01 to 2024-03-01, 0 missing values, "
        f"derived from ds-1 with forward_fill {p.MOST_RECENT_MARK}",
    ]
    assert text.count(p.MOST_RECENT_MARK) == 1


def test_inventory_singular_row_and_empty_dataset() -> None:
    one = _dataset("ds-1", [2.0])
    empty = _dataset("ds-2", [])
    lines = p.dataset_inventory(View((one, empty))).splitlines()
    assert ", 1 row, 2024-01-01 to 2024-01-01, 0 missing values" in lines[1]
    assert lines[2].endswith(f"0 rows, {p.EMPTY_RANGE_TEXT}, 0 missing values {p.MOST_RECENT_MARK}")


def test_inventory_is_deterministic() -> None:
    view = View((_dataset("ds-1", [1.0, NAN]), _dataset("ds-2", [4.0], indicator="gdp")))
    assert p.dataset_inventory(view) == p.dataset_inventory(view)
    assert PLACEHOLDER.search(p.dataset_inventory(view)) is None
