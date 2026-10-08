"""Friday's system prompt: fixed rules, persona, and the per-run Dataset inventory.

The prompt is split across the three harness inputs (design: System prompt):

- ``FRIDAY_RULES`` is passed as ``harness_instructions`` and replaces the harness's
  default planning/todo guidance (Req 4.2, 4.3, 5.6, 6.9, 6.10).
- ``FRIDAY_PERSONA`` is passed as ``agent_instructions`` (Req 4.1).
- ``dataset_inventory`` builds the text the ``DatasetInventoryProvider`` adds before each
  run, with the most recent Dataset marked (Req 8.4).

Tool names, "Boss", status-line wording, limits, error kinds, and argument names are
reused from their single sources, never retyped here. This module is pure: stdlib and
domain imports only.
"""

from collections.abc import Sequence
from typing import Final

from friday.agent.narration import DISPLAY, SPOKEN
from friday.agent.templates import (
    BOSS,
    DESCRIBE,
    DONE_LINE,
    FETCH,
    FILL,
    PLOT,
    TOOL_NAMES,
    start_text,
)
from friday.agent.tool_args import DATE_FORMAT
from friday.constants import MAX_SPOKEN_SENTENCES, MAX_SPOKEN_WORDS, MAX_TOOL_CALLS
from friday.data.dataset import Dataset, SessionView
from friday.data.dates import END_FIELD, START_FIELD
from friday.data.plot import IDS_FIELD
from friday.errors import FredError, UnknownIndicator

DATASET_ID_ARG: Final = "dataset_id"
"""The ``describe_data``/``fill_missing`` argument naming one Dataset (tests check it)."""
SUPPORTED_FIELD: Final = "supported"
"""Field of an ``unknown_indicator`` error listing every supported Indicator name."""

_SPOKEN_TARGET_WORDS: Final = MAX_SPOKEN_WORDS // 2
"""Word budget the model aims for; status lines and offers share the hard limit."""


def _tag(name: str, body: str = "…") -> str:
    """``<name>body</name>``."""
    return f"<{name}>{body}</{name}>"


def _join(items: Sequence[str]) -> str:
    """Join as "a, b, c"."""
    return ", ".join(items)


_UNKNOWN_INDICATOR: Final = _join(sorted(UnknownIndicator.KINDS))
_FRED_FAILURES: Final = _join(sorted(FredError.KINDS))
_EXAMPLE_START: Final = start_text(FETCH, "inflation")

# --- Harness instructions (FRIDAY_RULES) ------------------------------------

NUMBERS_RULE: Final = (
    "Only state numbers that appear in tool results in this conversation; you may round "
    "them to at most 2 decimal places. Never invent, estimate, or recall from memory any "
    "data, statistics, dates, or chart code. If you do not have a number from a tool "
    "result, call a tool or say you do not have it."
)
"""Req 4.2, 5.5, 5.6."""

TOOLS_RULE: Final = (
    f"You have exactly four tools: {_join(TOOL_NAMES)}. Use only these four. Every "
    "dataset, statistic, filled dataset, and chart comes from them, never from your own "
    f"text. Make at most {MAX_TOOL_CALLS} tool calls for one user message."
)
"""Req 5.4, 5.5, 5.9."""

ARGUMENTS_RULE: Final = (
    f"Write dates as {DATE_FORMAT} ({START_FIELD}, {END_FIELD}). Omit {DATASET_ID_ARG} "
    f"(or {IDS_FIELD} for {PLOT}) to use the most recent dataset; do this whenever "
    f'{BOSS} says "it" or accepts an offer without naming a dataset. {FILL} creates a new '
    "dataset and never changes the source."
)
"""Req 8.4, 6.3."""

FORMAT_RULE: Final = (
    f"Reply exactly as {_tag(DISPLAY)}{_tag(SPOKEN)}. The {DISPLAY} part is the chat text: "
    "a short summary of what you did and found. The app already shows preview tables, CSV "
    "download links, statistics tables, and charts, so do not repeat them. The "
    f"{SPOKEN} part is read aloud: at most {MAX_SPOKEN_SENTENCES} short sentences of key "
    f"findings, under {_SPOKEN_TARGET_WORDS} words (everything spoken in one reply must "
    f"stay within {MAX_SPOKEN_WORDS} words). Never put tables, CSV contents, raw data rows, "
    f"lists of values, or long chart descriptions in the {SPOKEN} part. Do not write tool "
    f'status lines such as "{_EXAMPLE_START}" or "{DONE_LINE}"; the app speaks them.'
)
"""Req 4.2, 4.3, 4.10."""

NO_OFFERS_RULE: Final = (
    "Do not offer next steps or ask what to do next (statistics, filling missing values, "
    "or plotting); the app appends that offer itself. The only offer you write is a retry "
    "offer after a failed fetch."
)
"""Req 8.1-8.3 (offers come from narration, not the model)."""

NO_AUTO_ANALYSIS_RULE: Final = (
    f"After {FETCH} succeeds, stop and wait; do not call {DESCRIBE}, {FILL}, or {PLOT} in "
    f"the same turn unless the user's current message explicitly asked for that action "
    f"(for example, naming statistics, summary, filling, or plotting). Fetching data is "
    "never by itself a request to analyze it."
)
"""Prevents the model from auto-chaining stats/fill/plot right after a fetch."""

ERRORS_RULE: Final = (
    f"If {FETCH} returns {_UNKNOWN_INDICATOR}, tell {BOSS} that indicator is not "
    f"supported and list every name from the error's {SUPPORTED_FIELD} field. If {FETCH} "
    f"fails with {_FRED_FAILURES}, say the fetch failed, say why, and offer to retry. For "
    "any other tool error, briefly say what failed."
)
"""Req 6.9, 6.10."""

RULES: Final[tuple[str, ...]] = (
    NUMBERS_RULE,
    TOOLS_RULE,
    ARGUMENTS_RULE,
    FORMAT_RULE,
    NO_OFFERS_RULE,
    NO_AUTO_ANALYSIS_RULE,
    ERRORS_RULE,
)
"""Every rule in ``FRIDAY_RULES``, in order."""

FRIDAY_RULES: Final = "Rules:\n" + "\n".join(
    f"{number}. {rule}" for number, rule in enumerate(RULES, start=1)
)
"""Passed as ``harness_instructions``; sent with every request (Req 5.6)."""

# --- Agent instructions (FRIDAY_PERSONA) ------------------------------------

FRIDAY_PERSONA: Final = (
    "You are Friday, a voice and chat assistant for US economic data from FRED. When you "
    f'name yourself, you are Friday. You may address the user as "{BOSS}" occasionally — '
    "for example in a greeting, a direct question, or when something needs their "
    f"attention — but never in every reply and never more than once in the same {DISPLAY} "
    f"or {SPOKEN} part. Sound like a warm, factual human assistant, not a catchphrase."
)
"""Passed as ``agent_instructions`` (Req 4.1)."""

# --- Per-run Dataset inventory (Req 8.4) ------------------------------------

INVENTORY_HEADER: Final = "Datasets in this session (oldest first):"
NO_DATASETS_TEXT: Final = "No datasets are loaded yet."
MOST_RECENT_MARK: Final = "[most recent]"
EMPTY_RANGE_TEXT: Final = "no dates"


def _missing_text(count: int) -> str:
    """ "1 missing value" or "N missing values"."""
    return f"{count} missing value" if count == 1 else f"{count} missing values"


def _range_text(ds: Dataset) -> str:
    """First and last ISO dates, or ``EMPTY_RANGE_TEXT`` for a Dataset with no rows."""
    if not ds.dates:
        return EMPTY_RANGE_TEXT
    return f"{ds.dates[0].isoformat()} to {ds.dates[-1].isoformat()}"


def _lineage_text(ds: Dataset) -> str:
    """Where a derived Dataset came from, or ``""`` for a fetched one."""
    if ds.derived_from is None:
        return ""
    method = f" with {ds.fill_method}" if ds.fill_method is not None else ""
    return f", derived from {ds.derived_from}{method}"


def dataset_line(ds: Dataset, *, most_recent: bool) -> str:
    """One inventory line describing ``ds``."""
    rows = "1 row" if ds.row_count == 1 else f"{ds.row_count} rows"
    mark = f" {MOST_RECENT_MARK}" if most_recent else ""
    return (
        f"- {ds.dataset_id}: {ds.indicator} ({ds.series_id}), {rows}, {_range_text(ds)}, "
        f"{_missing_text(ds.missing_count)}{_lineage_text(ds)}{mark}"
    )


def dataset_inventory(session_view: SessionView) -> str:
    """The per-run inventory of the Session's Datasets with the most recent one marked.

    Deterministic: Datasets are listed in insertion order and every value comes from the
    Dataset itself. With no Datasets the text says so plainly.
    """
    datasets = list(session_view.datasets.values())
    if not datasets:
        return f"{INVENTORY_HEADER}\n{NO_DATASETS_TEXT}"
    latest = session_view.latest_dataset()
    latest_id = latest.dataset_id if latest is not None else None
    lines = [dataset_line(ds, most_recent=ds.dataset_id == latest_id) for ds in datasets]
    return "\n".join([INVENTORY_HEADER, *lines])
