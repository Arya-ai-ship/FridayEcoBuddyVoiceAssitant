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
from friday.constants import (
    MAX_MISSING_DATES_LISTED,
    MAX_SPOKEN_SENTENCES,
    MAX_SPOKEN_WORDS,
    MAX_TOOL_CALLS,
)
from friday.data.dataset import Dataset, SessionView
from friday.data.dates import END_FIELD, START_FIELD
from friday.data.indicators import IndicatorEntry, IndicatorMap
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

ANSWER_FROM_CONTEXT_RULE: Final = (
    "Think like an economist first, and only reach for a tool when it will actually add "
    "something. Before calling a tool, check what you already have: the fetch results and "
    "the dataset list in this conversation already give you each dataset's indicator, "
    "series, date range, row count, missing-value count, and the exact dates of any "
    "missing values. If the question can be answered from that context or from sound "
    "economic reasoning, answer directly instead of calling a tool. For example, when "
    f"{BOSS} asks where a missing value is, read it off the missing-value dates you "
    "already have rather than running statistics. Call a tool only to fetch new data, "
    "produce statistics, fill gaps, or draw a chart that you cannot already report. You "
    "may interpret and explain what the data means, but the specific numbers and dates you "
    "state must still come from a tool result or the dataset list, never from memory."
)
"""Autonomy: reason over the context already present and skip tool calls that would not add
information, while keeping stated numbers/dates grounded (Req 4.2, 5.5)."""

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

OFFERS_RULE: Final = (
    "End a successful turn with a brief offer of the useful next steps, chosen only from "
    f"running descriptive statistics, filling missing values, and plotting the data. "
    f"Right after a successful {FETCH}, ask whether to run summary statistics or plot the "
    "data; if that dataset has missing values, first say how many it has and that you can "
    f"fill them. After a successful {DESCRIBE}, {FILL}, or {PLOT}, offer the useful "
    "remaining steps for the dataset you just worked on. Use the conversation so far: "
    "never offer an action you have already run on that same series earlier in this "
    "thread, and only offer filling missing values when that dataset still has missing "
    "values. If nothing useful remains, do not offer anything. Keep the offer to one "
    "short sentence. After a failed fetch, instead offer to retry. Offers are the only "
    "place you ask what to do next."
)
"""The model controls the whole next-step offer flow (fetch question included), using the
conversation history so it never repeats an action already run on the current series."""

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
    ANSWER_FROM_CONTEXT_RULE,
    ARGUMENTS_RULE,
    FORMAT_RULE,
    OFFERS_RULE,
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
    "You are Friday, a voice and chat assistant for US economic data from FRED, and you "
    "carry yourself like a seasoned co-economist sitting beside the user: fluent in "
    "macro and econ concepts, quick with a dry, good-natured joke, and genuinely fun to "
    "talk with across a back-and-forth conversation. When you name yourself, you are "
    f'Friday. You may address the user as "{BOSS}" occasionally — for example in a '
    "greeting, a direct question, or when something needs their attention — but never in "
    f"every reply and never more than once in the same {DISPLAY} or {SPOKEN} part. When "
    "you explain an economic idea, use plain, everyday words and reach for a simple "
    "analogy whenever one helps (inflation as air slowly leaking the value out of a "
    "balloon, interest rates as the price tag on borrowing money); skip the jargon unless "
    "the user asks for it. Keep the humor light and warm, never at the user's expense, "
    "and let it ride naturally over a multi-turn chat rather than forcing a joke into "
    "every line. Above all this is tone, not license: the Rules still bind you — only "
    "state numbers that came from tool results, never invent or recall data, and keep "
    "within the reply format and length limits. A good analogy earns its place only when "
    "it fits inside those limits."
)
"""Passed as ``agent_instructions`` (Req 4.1)."""

# --- Supported-indicator catalogue (injected once per session start) --------

INDICATOR_CATALOGUE_HEADER: Final = (
    "Supported indicators you can fetch (name | aliases | FRED series ID | transformation):"
)
"""Header line for the indicator catalogue injected into the model's instructions."""


def _indicator_line(entry: IndicatorEntry) -> str:
    """One catalogue line: name, aliases, series ID, and optional transformation."""
    aliases = ", ".join(entry.aliases) if entry.aliases else "—"
    transform = entry.transformation if entry.transformation else "none"
    return f"- {entry.name} | {aliases} | {entry.series_id} | {transform}"


def indicator_catalogue(indicators: IndicatorMap) -> str:
    """The full indicator catalogue injected as a context-provider instruction.

    Lists every supported indicator with its canonical name, aliases, FRED series ID, and
    transformation so the model knows exactly what it can fetch and how to refer to it.
    Deterministic: reflects the loaded ``IndicatorMap`` exactly.
    """
    lines = [_indicator_line(e) for e in indicators.entries]
    return "\n".join([INDICATOR_CATALOGUE_HEADER, *lines])


# --- Per-run Dataset inventory (Req 8.4) ------------------------------------

INVENTORY_HEADER: Final = "Datasets in this session (oldest first):"
NO_DATASETS_TEXT: Final = "No datasets are loaded yet."
MOST_RECENT_MARK: Final = "[most recent]"
EMPTY_RANGE_TEXT: Final = "no dates"


def _missing_text(ds: Dataset) -> str:
    """ "1 missing value (2026-08-01)" / "N missing values (dates...)" / "0 missing values".

    Names the gap dates (up to ``MAX_MISSING_DATES_LISTED``) so the model can say where the
    gaps are from the inventory alone, without re-running a tool.
    """
    count = ds.missing_count
    noun = "missing value" if count == 1 else "missing values"
    if count == 0:
        return f"0 {noun}"
    dates = ds.missing_dates
    listed = [day.isoformat() for day in dates[:MAX_MISSING_DATES_LISTED]]
    shown = ", ".join(listed)
    if len(dates) > len(listed):
        shown = f"{shown}, and more"
    return f"{count} {noun} ({shown})"


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
        f"{_missing_text(ds)}{_lineage_text(ds)}{mark}"
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
