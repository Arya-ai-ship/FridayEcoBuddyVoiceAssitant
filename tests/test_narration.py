"""Unit tests for ``friday.agent.narration`` status lines, ``parse_reply``, and offers.

Covers Requirements 4.4/4.5/4.11 (status lines) and 8.1/8.2/8.3/8.5 (next-step offers),
plus the design's ``<display>``/``<spoken>`` reply format with its 2-sentence fallback.
"""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

import numpy as np
import pytest

from friday.agent import templates as t
from friday.agent.narration import (
    CallOutcome,
    done_line,
    error_line,
    grounded_numbers,
    guard,
    next_step_offer,
    parse_reply,
    split_sentences,
    start_line,
)
from friday.constants import MAX_DONE_WORDS, MAX_SPOKEN_WORDS, MAX_STATUS_WORDS
from friday.data.dataset import Dataset, SessionView

RAW_TAG = re.compile(r"<\s*/?\s*(display|spoken)\b", re.IGNORECASE)


def _dataset(dataset_id: str, values: list[float]) -> Dataset:
    dates = tuple(date(2024, month, 1) for month in range(1, len(values) + 1))
    array = np.array(values, dtype=np.float64)
    return Dataset(dataset_id, "inflation", "inflation", "CPIAUCSL", dates, array, None)


@dataclass(frozen=True)
class View:
    """A minimal real ``SessionView`` over a fixed list of Datasets."""

    items: tuple[Dataset, ...] = field(default_factory=tuple[Dataset, ...])

    @property
    def datasets(self) -> Mapping[str, Dataset]:
        return {ds.dataset_id: ds for ds in self.items}

    def latest_dataset(self) -> Dataset | None:
        return self.items[-1] if self.items else None


COMPLETE = _dataset("ds-1", [1.0, 2.0, 3.0])
GAPPY = _dataset("ds-2", [1.0, float("nan"), 3.0, float("nan")])
NO_GAPS: SessionView = View((COMPLETE,))
WITH_GAPS: SessionView = View((COMPLETE, GAPPY))


# --- Status lines (Req 4.4, 4.5, 4.11) -------------------------------------------


def test_fetch_start_line_names_the_resolved_indicator() -> None:
    line = start_line("fetch_data", {"indicator": "CPI yoy"}, "inflation")
    assert line == "Fetching inflation data."


def test_fetch_start_line_falls_back_to_the_raw_argument_then_the_data() -> None:
    assert start_line("fetch_data", {"indicator": "jobs"}, None) == "Fetching jobs data."
    assert start_line("fetch_data", {"indicator": 3}, "  ") == "Fetching the data."
    assert start_line("fetch_data") == "Fetching the data."


@pytest.mark.parametrize("tool", ["describe_data", "fill_missing", "plot_data"])
def test_other_start_lines_ignore_the_indicator(tool: t.ToolName) -> None:
    assert start_line(tool, {"indicator": "x"}, "inflation") == t.START_LINES[tool]


@pytest.mark.parametrize("tool", t.TOOL_NAMES)
def test_done_line_is_at_most_ten_words(tool: str) -> None:
    line = done_line(tool)
    assert line == t.DONE_LINE
    assert 1 <= len(line.split()) <= MAX_DONE_WORDS


@pytest.mark.parametrize(
    ("tool", "indicator", "expected"),
    [
        ("fetch_data", "inflation", "Couldn't fetch inflation data, Boss."),
        ("fetch_data", None, "Couldn't fetch the data, Boss."),
        ("describe_data", None, "Couldn't run statistics, Boss."),
        ("fill_missing", None, "Couldn't fill missing values, Boss."),
        ("plot_data", "inflation", "Couldn't plot the data, Boss."),
    ],
)
def test_error_line_names_the_failed_action(
    tool: str, indicator: str | None, expected: str
) -> None:
    assert error_line(tool, indicator) == expected


@pytest.mark.parametrize("call", [start_line, done_line, error_line])
def test_status_lines_reject_unknown_tools(call: Callable[[str], str]) -> None:
    with pytest.raises(ValueError, match="unknown tool 'search_web'"):
        call("search_web")


# --- parse_reply ------------------------------------------------------------------


def test_parse_reply_reads_both_sections() -> None:
    text = "<display>Boss, CPI rose.\n\n| a | b |</display><spoken>CPI rose, Boss.</spoken>"
    assert parse_reply(text) == ("Boss, CPI rose.\n\n| a | b |", "CPI rose, Boss.")


def test_parse_reply_accepts_any_order_case_and_tag_whitespace() -> None:
    text = "  < SPOKEN >  Short   one.\n</spoken >\n< Display>  Long one.  </ display>  "
    assert parse_reply(text) == ("Long one.", "Short one.")


def test_parse_reply_joins_repeated_sections_in_order() -> None:
    text = "<display>A.</display><spoken>One.</spoken><display>B.</display><spoken>Two.</spoken>"
    assert parse_reply(text) == ("A.\n\nB.", "One. Two.")


def test_parse_reply_without_tags_uses_the_text_and_two_sentences() -> None:
    text = "Boss, CPI rose 3.5% in May. It was 3.2% in April! Is that high? Maybe."
    display, spoken = parse_reply(text)
    assert display == text
    assert spoken == "Boss, CPI rose 3.5% in May. It was 3.2% in April!"


def test_parse_reply_without_spoken_speaks_the_first_two_display_sentences() -> None:
    text = "<display>Boss, here it is.\nLine two\nLine three.</display>"
    assert parse_reply(text) == (
        "Boss, here it is.\nLine two\nLine three.",
        "Boss, here it is. Line two",
    )


def test_parse_reply_with_only_spoken_displays_the_untagged_text() -> None:
    assert parse_reply("<spoken>Done, Boss.</spoken>") == ("Done, Boss.", "Done, Boss.")


def test_parse_reply_handles_unclosed_and_nested_tags() -> None:
    assert parse_reply("<display>Shown. <spoken>Said.</spoken> More.</display>") == (
        "Shown.\n\nMore.",
        "Said.",
    )
    assert parse_reply("<display>Shown.</display><spoken>Said, Boss.") == ("Shown.", "Said, Boss.")


@pytest.mark.parametrize(
    "text",
    [
        "<display><display>x</display>",
        "</spoken>stray closer<spoken>",
        "<disp<display>lay>Hi.</display>",
        "<display attr='1'>Hi</display><spoken",
        "<<spoken>spoken>>Hi.<</spoken>/spoken>",
        "<display>",
        "",
        "   \n ",
    ],
)
def test_parse_reply_never_returns_raw_tags(text: str) -> None:
    display, spoken = parse_reply(text)
    assert not RAW_TAG.search(display)
    assert not RAW_TAG.search(spoken)
    assert display == display.strip()
    assert spoken == " ".join(spoken.split())


def test_parse_reply_of_empty_text_is_empty() -> None:
    assert parse_reply("") == ("", "")


def test_split_sentences_keeps_decimals_and_splits_lines() -> None:
    assert split_sentences("Rate was 3.25 now. Next?\nRow 1\n\n Done!") == [
        "Rate was 3.25 now.",
        "Next?",
        "Row 1",
        "Done!",
    ]


# --- next_step_offer (Req 8.1, 8.2, 8.3, 8.5) ------------------------------------


def test_no_calls_means_no_offer() -> None:
    assert next_step_offer([], WITH_GAPS) is None


@pytest.mark.parametrize(
    "outcomes",
    [
        [CallOutcome("fetch_data", False)],
        [CallOutcome.fetched(COMPLETE), CallOutcome("plot_data", False)],
        [CallOutcome("describe_data", False), CallOutcome("plot_data", True)],
    ],
)
def test_any_failed_call_means_no_offer(outcomes: list[CallOutcome]) -> None:
    assert next_step_offer(outcomes, WITH_GAPS) is None


def test_fetch_without_gaps_asks_the_single_question() -> None:
    assert next_step_offer([CallOutcome.fetched(COMPLETE)], WITH_GAPS) == t.FETCH_QUESTION


def test_fetch_with_gaps_states_the_count_then_asks_the_question() -> None:
    offer = next_step_offer([CallOutcome.fetched(GAPPY)], NO_GAPS)
    assert offer == f"This series has 2 missing values; I can fill them. {t.FETCH_QUESTION}"


@pytest.mark.parametrize(
    ("tool", "session", "expected"),
    [
        ("describe_data", WITH_GAPS, "Next, I can fill missing values or plot the data."),
        ("describe_data", NO_GAPS, "Next, I can plot the data."),
        (
            "plot_data",
            WITH_GAPS,
            "Next, I can run descriptive statistics or fill missing values.",
        ),
        ("plot_data", NO_GAPS, "Next, I can run descriptive statistics."),
        (
            "fill_missing",
            WITH_GAPS,
            "Next, I can run descriptive statistics or plot the data.",
        ),
        ("fill_missing", NO_GAPS, "Next, I can run descriptive statistics or plot the data."),
    ],
)
def test_follow_up_offer_names_the_other_actions(
    tool: str, session: SessionView, expected: str
) -> None:
    assert next_step_offer([CallOutcome(tool, True)], session) == expected


def test_offer_follows_the_last_successful_call() -> None:
    outcomes = [CallOutcome.fetched(GAPPY), CallOutcome("describe_data", True)]
    assert next_step_offer(outcomes, WITH_GAPS) == (
        "Next, I can fill missing values or plot the data."
    )
    outcomes = [CallOutcome("plot_data", True), CallOutcome.fetched(COMPLETE)]
    assert next_step_offer(outcomes, WITH_GAPS) == t.FETCH_QUESTION


def test_empty_session_still_offers_without_fill() -> None:
    assert next_step_offer([CallOutcome("plot_data", True)], View()) == (
        "Next, I can run descriptive statistics."
    )


# --- Narration guard grounding (regressions from the readiness sweep) -------------

STATS_RESULT = (
    '{"ok":true,"dataset_id":"ds-1","stats":{"count":12,"missing_count":0,'
    '"mean":3.3875,"std":0.21608605188253552,"min":3.04,"p25":3.245,"median":3.405,'
    '"p75":3.535,"max":3.71,"first_date":"2023-01-01","last_date":"2023-12-01"}}'
)
FETCH_RESULT = (
    '{"ok":true,"dataset_id":"ds-1","indicator":"10-year treasury yield","series_id":"GS10",'
    '"row_count":12,"first_date":"2023-01-01","last_date":"2023-12-01","missing_count":0,'
    '"transformation":null}'
)


def _guarded(sentence: str, session: SessionView = NO_GAPS) -> str:
    grounded = grounded_numbers(session, [STATS_RESULT, FETCH_RESULT])
    return guard("Boss.", sentence, grounded, 0)[1]


@pytest.mark.parametrize(
    "sentence",
    [
        "The mean was 3.39 percent, Boss.",  # stats value rounded to 2 decimals
        "The standard deviation was 0.22, Boss.",  # a stat that is not a Dataset value
        "The first quartile was 3.25, Boss.",  # half-up rounding of 3.245 (stats table)
        "The first quartile was 3.24, Boss.",  # half-even rounding of 3.245
        "That covers 12 months, Boss.",
        "The series runs from 2023-01-01 to 2023-12-01, Boss.",  # ISO dates
        "The 10-year treasury yield peaked at 3.71, Boss.",  # number in an Indicator name
    ],
)
def test_guard_keeps_findings_stated_from_tool_results(sentence: str) -> None:
    assert _guarded(sentence) == sentence


@pytest.mark.parametrize(
    "sentence",
    [
        "It will hit 9.99 next year, Boss.",  # rounds to 10, but 10 is not what was said
        "The mean was 3.388, Boss.",  # three decimals is not an allowed rounding
        "Values were 3.11, 3.27, 3.43, 3.58 and 3.71.",  # a raw value list (Req 4.3)
    ],
)
def test_guard_drops_ungrounded_numbers_and_value_lists(sentence: str) -> None:
    assert _guarded(sentence) == ""


def test_grounded_numbers_include_dataset_values_and_date_parts() -> None:
    grounded = grounded_numbers(WITH_GAPS)
    assert {Decimal("1.0"), Decimal("3.0"), Decimal(2024), Decimal(1)} <= grounded
    assert not any(number.is_nan() for number in grounded)


def test_grounded_numbers_scan_non_json_tool_results_as_text() -> None:
    grounded = grounded_numbers(View(), ['Error: Requested function "x" not found.', "Took 42 s"])
    assert grounded == {Decimal(42)}


def test_status_budget_leaves_room_for_the_longest_offer_and_stop_text() -> None:
    longest_offer = f"{t.missing_values_text(123)} {t.FETCH_QUESTION}"
    longest_next = t.next_actions_text(["describe_data", "fill_missing", "plot_data"])
    for tail in (longest_offer, longest_next, t.TOOL_LIMIT_TEXT, t.NO_DATA_TEXT):
        assert MAX_STATUS_WORDS + len(tail.split()) <= MAX_SPOKEN_WORDS, tail
