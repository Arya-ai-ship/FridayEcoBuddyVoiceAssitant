"""Unit tests for ``friday.agent.templates`` (all user-facing strings).

Covers Requirements 3.5/3.6 (speech and mic messages), 4.1 ("Boss" used sparingly, only
where Friday addresses the user directly or needs their attention), 4.4/4.5/4.11 (status
lines), 5.7/5.9/5.12/8.6 (fixed outcome texts), 7.8 (CSV unavailable), and 8.1–8.3
(next-step offer pieces).
"""

import itertools
import re
import string

import pytest

from friday.agent import templates as t
from friday.constants import MAX_DONE_WORDS, MAX_TOOL_CALLS

PLACEHOLDER = re.compile(r"\{[^}]*\}")


def _words(text: str) -> int:
    return len(text.split())


def _placeholders(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


# --- Tool names ----------------------------------------------------------------


def test_tool_names_match_the_design() -> None:
    assert t.TOOL_NAMES == ("fetch_data", "describe_data", "fill_missing", "plot_data")
    assert set(t.START_LINES) == set(t.TOOL_NAMES)
    assert set(t.ERROR_LINES) == set(t.TOOL_NAMES)


# --- Status lines (Req 4.4, 4.5, 4.11) -------------------------------------------


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        ("fetch_data", "Fetching inflation data."),
        ("describe_data", "Running statistics."),
        ("fill_missing", "Filling missing values."),
        ("plot_data", "Plotting the data."),
    ],
)
def test_start_lines_use_design_wording(tool: t.ToolName, expected: str) -> None:
    assert t.start_text(tool, "inflation") == expected


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        ("fetch_data", "Couldn't fetch inflation data, Boss."),
        ("describe_data", "Couldn't run statistics, Boss."),
        ("fill_missing", "Couldn't fill missing values, Boss."),
        ("plot_data", "Couldn't plot the data, Boss."),
    ],
)
def test_error_lines_name_the_failed_action(tool: t.ToolName, expected: str) -> None:
    assert t.error_text(tool, "inflation") == expected


def test_only_fetch_lines_take_an_indicator_placeholder() -> None:
    for lines in (t.START_LINES, t.ERROR_LINES):
        assert _placeholders(lines["fetch_data"]) == {"indicator"}
        for tool in ("describe_data", "fill_missing", "plot_data"):
            assert _placeholders(lines[tool]) == set()


@pytest.mark.parametrize("indicator", [None, "", "   "])
def test_fetch_lines_fall_back_without_an_indicator(indicator: str | None) -> None:
    assert t.start_text("fetch_data", indicator) == "Fetching the data."
    assert t.error_text("fetch_data", indicator) == "Couldn't fetch the data, Boss."


def test_fetch_lines_keep_braces_in_indicator_verbatim() -> None:
    assert t.start_text("fetch_data", " {x} ") == "Fetching {x} data."


@pytest.mark.parametrize("tool", t.TOOL_NAMES)
def test_start_lines_have_no_placeholders_left(tool: t.ToolName) -> None:
    assert not PLACEHOLDER.search(t.start_text(tool, "cpi"))


@pytest.mark.parametrize("tool", t.TOOL_NAMES)
def test_error_lines_address_boss(tool: t.ToolName) -> None:
    line = t.error_text(tool, "cpi")
    assert t.BOSS in line
    assert not PLACEHOLDER.search(line)


def test_done_line_is_short() -> None:
    assert t.DONE_LINE == "Done."
    assert 1 <= _words(t.DONE_LINE) <= MAX_DONE_WORDS


# --- Next-step offer pieces (Req 8.1, 8.2, 8.3) -----------------------------------


def test_fetch_question_is_a_single_boss_question() -> None:
    assert t.FETCH_QUESTION == (
        "Boss, do you want to run summary statistics, or do you want to plot the data?"
    )
    assert t.FETCH_QUESTION.count("?") == 1
    assert t.FETCH_QUESTION.endswith("?")


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (1, "This series has 1 missing value; I can fill it."),
        (2, "This series has 2 missing values; I can fill them."),
        (37, "This series has 37 missing values; I can fill them."),
    ],
)
def test_missing_values_note_states_the_count(count: int, expected: str) -> None:
    assert t.missing_values_text(count) == expected


@pytest.mark.parametrize("count", [0, -1])
def test_missing_values_note_rejects_non_positive_counts(count: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        t.missing_values_text(count)


def test_fetch_offer_with_gaps_addresses_boss_and_ends_with_the_question() -> None:
    offer = f"{t.missing_values_text(3)} {t.FETCH_QUESTION}"
    assert t.BOSS in offer
    assert offer.endswith(t.FETCH_QUESTION)


@pytest.mark.parametrize(
    ("actions", "expected"),
    [
        (("plot_data",), "Next, I can plot the data."),
        (("describe_data",), "Next, I can run descriptive statistics."),
        (
            ("plot_data", "describe_data"),
            "Next, I can run descriptive statistics or plot the data.",
        ),
        (
            ("fill_missing", "plot_data"),
            "Next, I can fill missing values or plot the data.",
        ),
        (
            ("describe_data", "fill_missing", "plot_data"),
            "Next, I can run descriptive statistics, fill missing values, or plot the data.",
        ),
    ],
)
def test_next_actions_offer_lists_actions_in_order(
    actions: tuple[t.ToolName, ...], expected: str
) -> None:
    assert t.next_actions_text(actions) == expected


REQ_8_3_NAMES = {
    "describe_data": "descriptive statistics",
    "fill_missing": "fill missing values",
    "plot_data": "plot",
}
OFFERABLE: list[t.ToolName] = ["describe_data", "fill_missing", "plot_data"]


@pytest.mark.parametrize(
    "actions",
    [combo for n in (1, 2, 3) for combo in itertools.combinations(OFFERABLE, n)],
)
def test_next_actions_offer_names_exactly_the_given_actions(
    actions: tuple[t.ToolName, ...],
) -> None:
    offer = t.next_actions_text(actions)
    assert not PLACEHOLDER.search(offer)
    for tool, name in REQ_8_3_NAMES.items():
        assert (name in offer) == (tool in actions), (tool, offer)


@pytest.mark.parametrize("actions", [(), ("fetch_data",), ("fetch_data", "plot_data")])
def test_next_actions_offer_rejects_non_offerable_actions(
    actions: tuple[t.ToolName, ...],
) -> None:
    with pytest.raises(ValueError, match="offer needs"):
        t.next_actions_text(actions)


# --- Fixed outcome texts (Req 5.7, 5.9, 5.12, 8.6) --------------------------------


def test_outcome_texts_cover_every_non_ok_outcome() -> None:
    assert set(t.OUTCOME_TEXTS) == {"llm_unavailable", "aws_credentials", "tool_limit", "no_data"}


@pytest.mark.parametrize("kind", ["llm_unavailable", "aws_credentials", "tool_limit", "no_data"])
def test_outcome_texts_address_boss(kind: t.OutcomeKind) -> None:
    text = t.OUTCOME_TEXTS[kind]
    assert t.BOSS in text
    assert not PLACEHOLDER.search(text)


def test_llm_unavailable_text_says_the_model_is_unavailable() -> None:
    assert t.LLM_UNAVAILABLE_TEXT == (
        "Sorry Boss, my language model is unavailable right now. Please try again in a moment."
    )


def test_aws_credentials_text_asks_to_refresh_env_and_restart() -> None:
    text = t.AWS_CREDENTIALS_TEXT
    assert "expired or lack access" in text
    assert ".env" in text
    assert "refresh" in text
    assert "restart" in text


def test_tool_limit_text_uses_the_configured_limit() -> None:
    expected = (
        f"Boss, I couldn't finish that within the {MAX_TOOL_CALLS}-tool-call limit. "
        "Anything already fetched or plotted is still here."
    )
    assert expected == t.TOOL_LIMIT_TEXT


def test_no_data_text_asks_which_indicator_to_fetch() -> None:
    assert t.NO_DATA_TEXT == "Boss, no data is loaded yet. Which indicator should I fetch?"


# --- Backend messages outside the chat stream (Req 3.5, 3.6, 7.8) ---------------


@pytest.mark.parametrize("text", [t.STT_RETRY_TEXT, t.MIC_REQUIRED_TEXT, t.CSV_UNAVAILABLE_TEXT])
def test_backend_messages_address_boss(text: str) -> None:
    assert t.BOSS in text


def test_stt_retry_text_asks_to_repeat_or_type() -> None:
    assert "repeat" in t.STT_RETRY_TEXT
    assert "type" in t.STT_RETRY_TEXT


def test_mic_text_says_mic_is_required_and_typing_works() -> None:
    assert "microphone access is required" in t.MIC_REQUIRED_TEXT
    assert "type" in t.MIC_REQUIRED_TEXT


def test_csv_unavailable_text_asks_to_fetch_again() -> None:
    assert "no longer available" in t.CSV_UNAVAILABLE_TEXT
    assert "fetch it again" in t.CSV_UNAVAILABLE_TEXT


@pytest.mark.parametrize("code", ["stt_failed", "stt_timeout", "stt_empty", "audio_too_large"])
def test_transcribe_errors_ask_to_repeat(code: str) -> None:
    assert t.transcribe_error_text(code) == t.STT_RETRY_TEXT


def test_transcribe_credential_error_uses_the_req_5_12_text() -> None:
    assert t.transcribe_error_text("aws_credentials") == t.AWS_CREDENTIALS_TEXT
