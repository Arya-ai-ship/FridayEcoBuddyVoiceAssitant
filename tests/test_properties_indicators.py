"""Property test for Indicator_Map resolution (design: Correctness Property 1)."""

from typing import Final

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from friday.data.indicators import IndicatorEntry, IndicatorMap
from friday.errors import UnknownIndicator

# Unicode whitespace that str.strip() removes, beyond plain spaces.
_WHITESPACE: Final = " \t\n\r\x0b\x0c\u00a0\u2003\u3000"

# Letters (including ones with non-trivial case folding such as ß, İ, Σ/ς), digits,
# punctuation, and inner spaces.
_KEY_CHARS: Final = st.characters(categories=("L", "N", "P", "Zs"), codec="utf-8")


def _oracle_key(text: str) -> str:
    """The design's normalization, written independently of ``indicators.normalize``."""
    return text.strip().casefold()


def raw_keys() -> st.SearchStrategy[str]:
    """Names or aliases whose normalized form is non-empty (the loader rejects blanks)."""
    return st.text(_KEY_CHARS, min_size=1, max_size=20).filter(lambda t: _oracle_key(t) != "")


@st.composite
def indicator_maps(draw: st.DrawFn) -> IndicatorMap:
    """1-8 entries whose normalized names and aliases are unique across the whole map."""
    sizes = draw(st.lists(st.integers(0, 3), min_size=1, max_size=8))  # aliases per entry
    keys = draw(
        st.lists(
            raw_keys(),
            min_size=len(sizes) + sum(sizes),
            max_size=len(sizes) + sum(sizes),
            unique_by=_oracle_key,
        )
    )
    entries: list[IndicatorEntry] = []
    pos = 0
    for i, n_aliases in enumerate(sizes):
        name, aliases = keys[pos], tuple(keys[pos + 1 : pos + 1 + n_aliases])
        pos += 1 + n_aliases
        entries.append(IndicatorEntry(name=name, aliases=aliases, series_id=f"S{i}"))
    return IndicatorMap(tuple(entries))


@st.composite
def query_variants(draw: st.DrawFn, base: str) -> str:
    """``base`` with random per-character case and random surrounding whitespace."""
    cased = "".join(draw(st.sampled_from((c, c.upper(), c.lower(), c.swapcase()))) for c in base)
    pad = st.text(st.sampled_from(_WHITESPACE), max_size=3)
    return draw(pad) + cased + draw(pad)


@st.composite
def map_and_query(draw: st.DrawFn) -> tuple[IndicatorMap, str]:
    """A map plus a query that is a variant of one of its keys, or arbitrary text."""
    imap = draw(indicator_maps())
    keys = [k for e in imap.entries for k in (e.name, *e.aliases)]
    known = st.sampled_from(keys).flatmap(query_variants)
    unknown = st.text(max_size=20) | raw_keys().flatmap(query_variants)
    return imap, draw(known | unknown)


@settings(max_examples=200)
@given(map_and_query())
def test_resolution_is_case_and_whitespace_insensitive_and_complete(
    case: tuple[IndicatorMap, str],
) -> None:
    """Feature: friday-voice-data-assistant, Property 1: Indicator resolution is case- and
    whitespace-insensitive and complete.

    **Validates: Requirements 6.2, 6.9**
    """
    imap, query = case
    target = _oracle_key(query)
    expected = [e for e in imap.entries if target in {_oracle_key(k) for k in (e.name, *e.aliases)}]
    assert len(expected) <= 1  # generator guarantees unique normalized keys

    if expected:
        assert imap.resolve(query) is expected[0]
        return

    with pytest.raises(UnknownIndicator) as info:
        imap.resolve(query)
    assert info.value.supported == tuple(e.name for e in imap.entries)
    assert all(e.name in info.value.message for e in imap.entries)
