"""Deterministic narration: status lines, reply parsing, and next-step offers.

Every function here is pure (stdlib and friday domain modules only). The wording comes
from ``agent/templates.py``; this module only selects and composes it.

- ``start_line`` / ``done_line`` / ``error_line``: Tool status lines (Req 4.4, 4.5, 4.11).
- ``parse_reply``: splits the LLM's ``<display>``/``<spoken>`` reply (design: Narration).

The next-step offer is no longer composed here: the model writes its own offer, driven by
the ``OFFERS_RULE`` system prompt, using the conversation history.

- ``grounded_numbers`` / ``guard``: the narration guard that enforces number grounding,
  no data rows, and the spoken-word limit (Req 4.2, 4.3, 4.10, 9.5).
"""

import contextlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from types import MappingProxyType
from typing import Final, cast

from friday.agent.templates import (
    DONE_LINE,
    TOOL_NAMES,
    ToolName,
    error_text,
    start_text,
)
from friday.constants import MAX_SPOKEN_WORDS
from friday.data.dataset import SessionView

_TOOLS: Final[Mapping[str, ToolName]] = MappingProxyType({name: name for name in TOOL_NAMES})

# --- Status lines (Req 4.4, 4.5, 4.11) ------------------------------------------


def _tool_name(tool: str) -> ToolName:
    """Narrow ``tool`` to a known Tool name, raising ``ValueError`` otherwise."""
    name = _TOOLS.get(tool)
    if name is None:
        raise ValueError(f"unknown tool {tool!r}; expected one of {list(TOOL_NAMES)}")
    return name


def start_line(
    tool: str,
    args: Mapping[str, object] | None = None,
    resolved_indicator: str | None = None,
) -> str:
    """Return the status line spoken when ``tool`` starts (Req 4.4).

    The fetch line names the canonical ``resolved_indicator``, falling back to the raw
    ``indicator`` argument and then to "the data".
    """
    indicator = resolved_indicator
    if not (indicator and indicator.strip()) and args is not None:
        raw = args.get("indicator")
        indicator = raw if isinstance(raw, str) else None
    return start_text(_tool_name(tool), indicator)


def done_line(tool: str) -> str:
    """Return the completion status line for a successful ``tool`` call (Req 4.5)."""
    _tool_name(tool)
    return DONE_LINE


def error_line(tool: str, indicator: str | None = None) -> str:
    """Return the status line naming the failed action of ``tool`` (Req 4.11)."""
    return error_text(_tool_name(tool), indicator)


# --- Reply parsing ----------------------------------------------------------------

DISPLAY: Final = "display"
SPOKEN: Final = "spoken"
FALLBACK_SENTENCES: Final = 2
"""Sentences of display text spoken when the reply has no usable ``<spoken>`` part."""

_TAG: Final = re.compile(r"<\s*(/?)\s*(display|spoken)\s*>", re.IGNORECASE)
# Any leftover tag fragment: "<display", "</ spoken", "<display x='1'>", ...
_TAG_FRAGMENT: Final = re.compile(
    r"<\s*/?\s*(?:display|spoken)\b(?:[^<>\n]{0,40}>)?", re.IGNORECASE
)
_SENTENCE_BREAK: Final = re.compile(r"(?<=[.!?])\s+|\s*\n\s*")
_TRAILING_SPACE: Final = re.compile(r"[ \t]+\n")
_BLANK_LINES: Final = re.compile(r"\n{3,}")
_WHITESPACE: Final = re.compile(r"\s+")


def _sections(text: str) -> dict[str, list[str]]:
    """Split ``text`` by tags into display, spoken, and untagged ("") parts.

    Tags may appear in any order, repeat, nest, or be left unclosed: text belongs to the
    innermost open tag, a closing tag closes its most recent open tag (stray closers are
    ignored), and an unclosed tag runs to the end of the text.
    """
    parts: dict[str, list[str]] = {DISPLAY: [], SPOKEN: [], "": []}
    stack: list[str] = []
    pos = 0
    for match in _TAG.finditer(text):
        parts[stack[-1] if stack else ""].append(text[pos : match.start()])
        pos = match.end()
        name = match.group(2).lower()
        if not match.group(1):
            stack.append(name)
        elif name in stack:
            del stack[len(stack) - 1 - stack[::-1].index(name) :]
    parts[stack[-1] if stack else ""].append(text[pos:])
    return parts


def _scrub(text: str) -> str:
    """Remove tag fragments until none are left (removal can join a new one)."""
    while True:
        cleaned = _TAG_FRAGMENT.sub("", text)
        if cleaned == text:
            return text
        text = cleaned


def _clean_display(parts: Sequence[str]) -> str:
    """Join display parts with blank lines, keeping line breaks but trimming whitespace."""
    joined = "\n\n".join(p for p in (_scrub(part).strip() for part in parts) if p)
    return _BLANK_LINES.sub("\n\n", _TRAILING_SPACE.sub("\n", joined)).strip()


def _clean_spoken(parts: Sequence[str]) -> str:
    """Join spoken parts into one line of single-spaced text."""
    return _WHITESPACE.sub(" ", " ".join(_scrub(part) for part in parts)).strip()


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences at ``.``/``!``/``?`` + whitespace and at line breaks.

    Decimal points ("3.5") do not split. Empty pieces are dropped.
    """
    return [s for s in (piece.strip() for piece in _SENTENCE_BREAK.split(text)) if s]


def parse_reply(text: str) -> tuple[str, str]:
    """Split an LLM reply into ``(display, spoken)`` text.

    The reply should be ``<display>…</display><spoken>…</spoken>``. Repeated sections are
    joined in order. Without display content, the display text is the whole reply with
    tags removed; without spoken content, the spoken text is the first
    ``FALLBACK_SENTENCES`` sentences of the display text. Neither result contains a
    ``<display>``/``<spoken>`` tag, and spoken text is a single whitespace-normalized line.
    """
    parts = _sections(text)
    display = _clean_display(parts[DISPLAY])
    if not display:
        display = _clean_display([_TAG.sub(" ", text)])
    spoken = _clean_spoken(parts[SPOKEN])
    if not spoken:
        spoken = _clean_spoken(split_sentences(display)[:FALLBACK_SENTENCES])
    return display, spoken


# --- Narration guard (Req 4.2, 4.3, 4.10, 9.5) ------------------------------

_GROUNDING_DECIMALS: Final = (0, 1, 2)
"""Decimal places a grounded value may be rounded to when the LLM restates it (Req 9.5)."""

# A numeric token: digits with optional comma thousands-groups and an optional fraction.
# Tokens are unsigned (magnitudes are compared), so the hyphens in "2023-05-01" or
# "10-year" never turn "05" or "1" into negative numbers.
_NUMBER_TOKEN: Final = re.compile(r"(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?")

_MIN_LISTED_VALUES: Final = 4
"""A sentence with this many numeric words, making up most of it, is a value list."""

# A line that looks like a markdown table row or a bare data row (two or more cells split
# by ``|`` or by runs of two or more spaces / tabs), or a markdown table rule (``|---|``).
_TABLE_RULE: Final = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
_PIPE_ROW: Final = re.compile(r"\S\s*\|\s*\S")
_COLUMN_GAP: Final = re.compile(r"\S(?: {2,}|\t+)\S")


def _decimal(text: str) -> Decimal | None:
    """Parse a numeric token or number text into a finite ``Decimal``, or ``None``."""
    try:
        number = Decimal(text.replace(",", ""))
    except ArithmeticError:  # decimal.InvalidOperation: not a number
        return None
    return number if number.is_finite() else None


def numbers_in_text(text: str) -> list[Decimal]:
    """The unsigned numeric tokens in ``text``; an ISO date gives its year, month, and day."""
    numbers = (_decimal(token) for token in _NUMBER_TOKEN.findall(text))
    return [number for number in numbers if number is not None]


def _collect_json_numbers(value: object, found: list[Decimal]) -> None:
    """Add the numbers in a parsed JSON value; strings add their numeric tokens."""
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, int | float):
        number = _decimal(repr(value))
        if number is not None:
            found.append(number)
    elif isinstance(value, str):
        found.extend(numbers_in_text(value))
    elif isinstance(value, Mapping):
        for item in cast(Mapping[object, object], value).values():
            _collect_json_numbers(item, found)
    elif isinstance(value, list):
        for item in cast(list[object], value):
            _collect_json_numbers(item, found)


def numbers_in_tool_result(text: str) -> list[Decimal]:
    """Every number in one Tool result: JSON numbers exactly, plus tokens inside strings.

    A result that is not JSON (MAF's plain-text "function not found" error) is scanned as
    text.
    """
    try:
        parsed: object = json.loads(text)
    except ValueError:
        return numbers_in_text(text)
    found: list[Decimal] = []
    _collect_json_numbers(parsed, found)
    return found


def grounded_numbers(session: SessionView, tool_results: Iterable[str] = ()) -> set[Decimal]:
    """Return the numbers Friday may restate, from the Session's Tool results and Datasets.

    The set holds every number in ``tool_results`` (the Session's Tool result texts,
    including the dates and Indicator names inside them), every non-missing Dataset value,
    and the year, month, and day of every Dataset date (Req 4.2, 9.5). ``guard`` accepts a
    spoken number that equals one of these or one of its 0-, 1-, or 2-decimal roundings.
    """
    grounded: set[Decimal] = set()
    for dataset in session.datasets.values():
        for value in dataset.values.tolist():
            number = _decimal(repr(float(value)))  # a Missing (NaN) value gives None
            if number is not None:
                grounded.add(number)
        for day in dataset.dates:
            grounded.update(Decimal(part) for part in (day.year, day.month, day.day))
    for text in tool_results:
        grounded.update(numbers_in_tool_result(text))
    return grounded


def _spoken_forms(grounded: Iterable[Decimal]) -> set[Decimal]:
    """The magnitudes a grounded number may be spoken as: itself or a 0-2-decimal rounding.

    Half-up and half-even roundings are both accepted, plus the rounding of the number's
    exact binary value (what ``f"{x:.2f}"`` shows in the stats table), so 3.245 may be
    spoken as 3.24 or 3.25.
    """
    forms: set[Decimal] = set()
    for number in grounded:
        magnitude = number.copy_abs()
        forms.add(magnitude)
        binary = Decimal(float(magnitude))
        for places in _GROUNDING_DECIMALS:
            step = Decimal(1).scaleb(-places)
            for source, mode in (
                (magnitude, ROUND_HALF_UP),
                (magnitude, ROUND_HALF_EVEN),
                (binary, ROUND_HALF_EVEN),
            ):
                with contextlib.suppress(ArithmeticError):  # too many digits to round
                    forms.add(source.quantize(step, rounding=mode))
    return forms


def _is_data_row(sentence: str) -> bool:
    """Whether ``sentence`` looks like a markdown table row, table rule, or data row."""
    stripped = sentence.strip()
    if not stripped:
        return False
    if _TABLE_RULE.match(stripped):
        return True
    if _PIPE_ROW.search(stripped):
        return True
    return bool(_COLUMN_GAP.search(stripped))


def _is_value_list(sentence: str) -> bool:
    """Whether ``sentence`` is mostly a list of values (a raw data dump, Req 4.3)."""
    words = sentence.split()
    numeric = sum(1 for word in words if any(char.isdigit() for char in word))
    return numeric >= _MIN_LISTED_VALUES and 2 * numeric > len(words)


def _sentence_is_allowed(sentence: str, spoken_forms: set[Decimal]) -> bool:
    """Whether a spoken sentence is not table data and states only grounded numbers."""
    if _is_data_row(sentence) or _is_value_list(sentence):
        return False
    return all(number in spoken_forms for number in numbers_in_text(sentence))


def _word_count(text: str) -> int:
    """Number of whitespace-separated words in ``text``."""
    return len(text.split())


def guard(display: str, spoken: str, grounded: set[Decimal], fixed_words: int) -> tuple[str, str]:
    """Return guarded ``(display, spoken)`` text (Req 4.1, 4.2, 4.3, 4.10, 9.5).

    Display text is passed through as the model wrote it; the persona prompt and the
    fixed templates are where "Boss" appears, used sparingly rather than forced into
    every reply. Spoken text keeps only sentences that are not table rows or value lists
    and whose every number equals a ``grounded`` number or one of its 0-, 1-, or
    2-decimal roundings. It is then truncated at a sentence boundary so that
    ``fixed_words`` (the turn's spoken status-line words plus the offer the caller
    appends) plus the spoken words stay within ``MAX_SPOKEN_WORDS``. ``fixed_words`` is
    clamped to zero.
    """
    spoken_forms = _spoken_forms(grounded)
    budget = MAX_SPOKEN_WORDS - max(0, fixed_words)
    kept: list[str] = []
    used = 0
    for sentence in split_sentences(spoken):
        if not _sentence_is_allowed(sentence, spoken_forms):
            continue
        words = _word_count(sentence)
        if used + words > budget:
            break
        kept.append(sentence)
        used += words
    return display, " ".join(kept)
