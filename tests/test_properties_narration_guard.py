"""Property test for the narration guard (task 11.5, Req 4.2, 4.3, 4.10, 9.5)."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from friday.agent.narration import guard
from friday.constants import MAX_SPOKEN_WORDS

# Generated tokens never use thousands separators, so plain digit runs suffice here.
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_PLACES = (0, 1, 2)


def _oracle_forms(grounded: set[Decimal]) -> set[Decimal]:
    """Independent model of the allowed spoken magnitudes: each grounded number, or that
    number rounded to 0-2 decimals (half-up, half-even, or from its binary float value)."""
    forms: set[Decimal] = set()
    for number in grounded:
        magnitude = abs(number)
        forms.add(magnitude)
        for places in _PLACES:
            step = Decimal(1).scaleb(-places)
            forms.add(magnitude.quantize(step, rounding=ROUND_HALF_UP))
            forms.add(magnitude.quantize(step, rounding=ROUND_HALF_EVEN))
            forms.add(Decimal(float(magnitude)).quantize(step, rounding=ROUND_HALF_EVEN))
    return forms


def _grounded_numbers() -> st.SearchStrategy[set[Decimal]]:
    """Grounded values with up to 4 decimals, so 0-2-decimal roundings differ from them."""
    return st.sets(
        st.decimals(min_value=-10000, max_value=10000, places=4, allow_nan=False),
        min_size=1,
        max_size=8,
    )


def _spoken_form(number: Decimal, places: int) -> str:
    """How the model might say ``number``: rounded half-up to ``places`` decimals."""
    return str(abs(number).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


@st.composite
def _sentences(draw: st.DrawFn, grounded: list[Decimal]) -> str:
    """A sentence of words, grounded numbers (exact or rounded), or ungrounded numbers."""
    words = st.sampled_from(["CPI", "rose", "fell", "Boss", "the", "value", "percent", "year"])
    exact = st.sampled_from([str(abs(g)) for g in grounded])
    rounded = st.builds(_spoken_form, st.sampled_from(grounded), st.sampled_from(_PLACES))
    ungrounded = st.sampled_from(["999999", "12345.6789", "7777.77"])
    tokens = draw(st.lists(st.one_of(words, exact, rounded, ungrounded), min_size=1, max_size=8))
    return " ".join(tokens) + "."


@st.composite
def spoken_texts(draw: st.DrawFn, grounded: list[Decimal]) -> str:
    sentences = draw(st.lists(_sentences(grounded), min_size=0, max_size=6))
    if draw(st.booleans()):
        sentences.append("| date | value | 3.5 |")  # a table row
    return " ".join(sentences)


@settings(max_examples=300, deadline=None)
@given(data=st.data(), display=st.text(max_size=120), fixed_words=st.integers(0, 80))
def test_narration_guard_enforces_grounding_and_length(
    data: st.DataObject, display: str, fixed_words: int
) -> None:
    """Feature: friday-voice-data-assistant, Property 19: Narration guard enforces
    grounding and length.

    The guarded display text is passed through unchanged; every numeric token in the
    Spoken_Text is a grounded number or one of its 0-2-decimal roundings; the Spoken_Text
    has no table-row lines; and the status-line words plus the spoken words stay within 60.

    **Validates: Requirements 4.1, 4.2, 4.3, 4.10, 9.5**
    """
    grounded = data.draw(_grounded_numbers())
    spoken = data.draw(spoken_texts(sorted(grounded)))

    out_display, out_spoken = guard(display, spoken, grounded, fixed_words)

    # The guard does not alter display text; "Boss" comes from the persona prompt and
    # fixed templates, not from the guard (Req 4.1).
    assert out_display == display

    # Grounding: every numeric token in the spoken text is grounded (Req 4.2, 9.5).
    allowed = _oracle_forms(grounded)
    for token in _NUMBER.findall(out_spoken):
        assert Decimal(token) in allowed, f"ungrounded token {token!r} in {out_spoken!r}"

    # No table-row lines survive (Req 4.3).
    assert "|" not in out_spoken

    # Length: status-line words + spoken words within the hard limit (Req 4.10).
    assert not out_spoken or fixed_words + len(out_spoken.split()) <= MAX_SPOKEN_WORDS


@settings(max_examples=200, deadline=None)
@given(data=st.data(), places=st.sampled_from(_PLACES))
def test_guard_keeps_grounded_findings_and_drops_ungrounded_ones(
    data: st.DataObject, places: int
) -> None:
    """A short finding stating a grounded value (exact or rounded) is kept; a finding with
    a number outside the grounded forms is dropped (Req 4.2, 9.5)."""
    grounded = data.draw(_grounded_numbers())
    target = data.draw(st.sampled_from(sorted(grounded)))
    stated = data.draw(st.sampled_from([str(abs(target)), _spoken_form(target, places)]))
    kept = f"The mean was {stated} percent, Boss."
    assert guard("Boss.", kept, grounded, 0)[1] == kept

    other = data.draw(st.decimals(min_value=0, max_value=100000, places=2, allow_nan=False))
    assume(other not in _oracle_forms(grounded))
    assert guard("Boss.", f"It will reach {other} soon, Boss.", grounded, 0)[1] == ""


@settings(max_examples=100, deadline=None)
@given(display=st.text(max_size=50))
def test_guard_leaves_display_text_untouched(display: str) -> None:
    """``guard`` passes display text through unchanged; it no longer injects "Boss"
    (Req 4.1 — "Boss" now comes only from the persona prompt and fixed templates)."""
    out_display, _ = guard(display, "", set(), 0)
    assert out_display == display
