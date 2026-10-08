"""Every user-facing string Friday produces (single source of truth).

Status lines, next-step offer pieces, the fixed turn-outcome texts, and the messages
the Backend returns for speech, microphone, and CSV failures all live here.
``agent/narration.py`` selects and composes them, and ``agent/prompts.py`` reuses the
wording. This module is pure: stdlib and ``friday.constants`` only.

Friday addresses the user as "Boss" only where it reads naturally: a direct question
(``FETCH_QUESTION``) and messages that need the user's attention or a decision (an
expired credential, a missing microphone, a failed transcription, and similar). Routine
status lines (``START_LINES``, ``DONE_LINE``, ``ERROR_LINES``) and the routine next-step
offer (``NEXT_ACTIONS_LINE``) omit it, since those recur multiple times per turn and a
constant "Boss" there reads as a tic rather than a human assistant (Req 4.1).
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, Literal

from friday.constants import MAX_TOOL_CALLS

BOSS: Final = "Boss"
"""How Friday addresses the user, used sparingly rather than in every line (Req 4.1)."""

# --- Tool names --------------------------------------------------------------

ToolName = Literal["fetch_data", "describe_data", "fill_missing", "plot_data"]

FETCH: Final = "fetch_data"
DESCRIBE: Final = "describe_data"
FILL: Final = "fill_missing"
PLOT: Final = "plot_data"

TOOL_NAMES: Final[tuple[ToolName, ...]] = (FETCH, DESCRIBE, FILL, PLOT)
"""The four Friday tools, in offer order after fetch (Req 5.4)."""

# --- Status lines (Req 4.4, 4.5, 4.11) --------------------------------------

UNNAMED_INDICATOR: Final = "the"
"""Stands in for ``{indicator}`` when no Indicator is known ("Fetching the data.")."""

START_LINES: Final[Mapping[ToolName, str]] = MappingProxyType(
    {
        FETCH: "Fetching {indicator} data.",
        DESCRIBE: "Running statistics.",
        FILL: "Filling missing values.",
        PLOT: "Plotting the data.",
    }
)
"""Spoken when a Tool starts. The fetch line names the Indicator (Req 4.4)."""

DONE_LINE: Final = "Done."
"""Spoken when a Tool succeeds; at most ``MAX_DONE_WORDS`` words (Req 4.5)."""

ERROR_LINES: Final[Mapping[ToolName, str]] = MappingProxyType(
    {
        FETCH: "Couldn't fetch {indicator} data, Boss.",
        DESCRIBE: "Couldn't run statistics, Boss.",
        FILL: "Couldn't fill missing values, Boss.",
        PLOT: "Couldn't plot the data, Boss.",
    }
)
"""Spoken instead of the done line when a Tool fails; names the failed action. Errors
address Boss directly since they need the user's attention (Req 4.11)."""


def _with_indicator(template: str, indicator: str | None) -> str:
    """Fill the ``{indicator}`` placeholder, falling back to ``UNNAMED_INDICATOR``."""
    name = indicator.strip() if indicator else ""
    return template.format(indicator=name or UNNAMED_INDICATOR)


def start_text(tool: ToolName, indicator: str | None = None) -> str:
    """Return the start status line for ``tool``; ``indicator`` fills the fetch line."""
    return _with_indicator(START_LINES[tool], indicator)


def error_text(tool: ToolName, indicator: str | None = None) -> str:
    """Return the failure status line for ``tool``; ``indicator`` fills the fetch line."""
    return _with_indicator(ERROR_LINES[tool], indicator)


# --- Next-step offer pieces (Req 8.1, 8.2, 8.3) -----------------------------

FETCH_QUESTION: Final = (
    "Boss, do you want to run summary statistics, or do you want to plot the data?"
)
"""The single question that ends a response after a successful fetch (Req 8.1)."""

MISSING_VALUES_NOTE: Final = "This series has {count} missing values; I can fill them."
MISSING_VALUE_NOTE: Final = "This series has 1 missing value; I can fill it."
"""Stated before ``FETCH_QUESTION`` when the fetched Dataset has gaps (Req 8.2)."""

NEXT_ACTIONS_LINE: Final = "Next, I can {actions}."
"""Ends a response after a successful stats, fill, or plot call (Req 8.3)."""

ACTION_LABELS: Final[Mapping[ToolName, str]] = MappingProxyType(
    {
        DESCRIBE: "run descriptive statistics",
        FILL: "fill missing values",
        PLOT: "plot the data",
    }
)
"""Offerable follow-up actions, in offer order. Each names its Req 8.3 action."""


def missing_values_text(count: int) -> str:
    """Return the Missing_Value note for a fetched Dataset with ``count`` (≥ 1) gaps."""
    if count < 1:
        raise ValueError(f"missing value count must be at least 1, got {count}")
    return MISSING_VALUE_NOTE if count == 1 else MISSING_VALUES_NOTE.format(count=count)


def _join_or(items: Sequence[str]) -> str:
    """Join as "a", "a or b", or "a, b, or c"."""
    if len(items) <= 2:
        return " or ".join(items)
    return f"{', '.join(items[:-1])}, or {items[-1]}"


def next_actions_text(actions: Sequence[ToolName]) -> str:
    """Return the "Next, I can ..." offer naming ``actions`` in offer order."""
    labels = [ACTION_LABELS[tool] for tool in ACTION_LABELS if tool in actions]
    unknown = [tool for tool in actions if tool not in ACTION_LABELS]
    if unknown or not labels:
        raise ValueError(f"offer needs one or more of {list(ACTION_LABELS)}, got {list(actions)}")
    return NEXT_ACTIONS_LINE.format(actions=_join_or(labels))


# --- Fixed turn-outcome texts (Req 5.7, 5.9, 5.12, 8.6) ---------------------

OutcomeKind = Literal["llm_unavailable", "aws_credentials", "tool_limit", "no_data"]

LLM_UNAVAILABLE_TEXT: Final = (
    "Sorry Boss, my language model is unavailable right now. Please try again in a moment."
)
AWS_CREDENTIALS_TEXT: Final = (
    "Boss, the AWS credentials have expired or lack access. "
    "Please refresh the AWS values in .env and restart the Backend."
)
TOOL_LIMIT_TEXT: Final = (
    f"Boss, I couldn't finish that within the {MAX_TOOL_CALLS}-tool-call limit. "
    "Anything already fetched or plotted is still here."
)
NO_DATA_TEXT: Final = "Boss, no data is loaded yet. Which indicator should I fetch?"

OUTCOME_TEXTS: Final[Mapping[OutcomeKind, str]] = MappingProxyType(
    {
        "llm_unavailable": LLM_UNAVAILABLE_TEXT,
        "aws_credentials": AWS_CREDENTIALS_TEXT,
        "tool_limit": TOOL_LIMIT_TEXT,
        "no_data": NO_DATA_TEXT,
    }
)
"""``final`` text per non-``ok`` outcome. ``aws_credentials`` is never spoken (Req 4.8)."""

# --- Messages the Backend returns outside the chat stream --------------------

STT_RETRY_TEXT: Final = "Sorry Boss, I didn't catch that — please repeat or type it."
"""Transcribe error, timeout, or empty transcript (Req 3.6)."""

MIC_REQUIRED_TEXT: Final = (
    "Boss, microphone access is required for voice input. You can still type your request."
)
"""Microphone permission denied or no device (Req 3.5)."""

CSV_UNAVAILABLE_TEXT: Final = "That dataset is no longer available — please fetch it again, Boss."
"""CSV download for a Dataset_ID no longer in the Session (Req 7.8)."""


def transcribe_error_text(code: str) -> str:
    """Return the message for a ``POST /api/transcribe`` error ``code`` (Req 3.6, 5.12)."""
    return AWS_CREDENTIALS_TEXT if code == "aws_credentials" else STT_RETRY_TEXT
